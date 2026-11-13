import copy
import json
import os
from pathlib import Path
import subprocess
import torch
from torch import nn
import pytest
from test_whole_neuron import TinySwiGLU
from sparsifier.prune.whole_neuron import SwiGLUTriplet, WholeNeuronConfig, prune_triplet, apply_candidate
from sparsifier.prune.whole_neuron_io import export_compact, load_compact, save_stage, load_stage, compact_mlp, file_hash


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinySwiGLU(3, 7)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.embed = nn.Embedding(10, 3)
        self.head = nn.Linear(3, 10, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.model.layers:
            x = x+layer.mlp(x)
        return self.head(x)

    def generate(self, ids, steps=3):
        for _ in range(steps):
            ids = torch.cat((ids, self(ids)[:, -1].argmax(-1, keepdim=True)), 1)
        return ids


def test_mixed_width_compact_forward_generation_and_clean_process_reload(tmp_path):
    torch.manual_seed(1)
    model = TinyModel().double().eval()
    masks = {}
    for b, fraction in enumerate((.3, .6)):
        triplet = SwiGLUTriplet(model.model.layers[b].mlp)
        candidate = prune_triplet(triplet, torch.randn(30, 7, dtype=torch.double), WholeNeuronConfig(fraction))
        x = torch.randn(2, 4, 3, dtype=torch.double)
        apply_candidate(triplet, candidate)
        torch.testing.assert_close(compact_mlp(candidate)(x), triplet.mlp(x))
        masks[b] = candidate.keep
    ids = torch.tensor([[1, 3, 4]])
    expected, generation = model(ids).detach(), model.generate(ids)
    parameter_count = sum(p.numel() for p in model.parameters())
    manifest = export_compact(model, tmp_path, masks)
    assert manifest["parameter_count"] < parameter_count
    loaded, _ = load_compact(tmp_path, model_factory=TinyModel)
    torch.testing.assert_close(loaded(ids), expected)
    assert torch.equal(loaded.generate(ids), generation)
    checkpoint = tmp_path/"expected.pt"
    torch.save({"ids": ids, "output": expected, "generation": generation}, checkpoint)
    script = """import sys, torch
from test_whole_neuron_export import TinyModel
from sparsifier.prune.whole_neuron_io import load_compact
m,_=load_compact(sys.argv[1],model_factory=TinyModel)
e=torch.load(sys.argv[2],weights_only=True)
torch.testing.assert_close(m(e['ids']), e['output'])
assert torch.equal(m.generate(e['ids']), e['generation'])
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="src:.:tests")
    subprocess.run([os.sys.executable, "-c", script, str(tmp_path), str(checkpoint)], check=True, env=env)
    assert export_compact(model, tmp_path, masks) == manifest
    (tmp_path/"model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_compact(tmp_path, model_factory=TinyModel)


def test_triplet_checkpoint_hashes_and_upstream_resume(tmp_path):
    triplet = SwiGLUTriplet(TinySwiGLU())
    candidate = prune_triplet(triplet, torch.randn(30, 7), WholeNeuronConfig(.4))
    path = tmp_path/"stage.pt"
    checksum = save_stage(path, triplet, candidate, config_hash="cfg", upstream="dense", block=1)
    restored = load_stage(path, triplet, expected_hash=checksum, config_hash="cfg", upstream="dense", block=1)
    assert torch.equal(restored.keep, candidate.keep)
    with pytest.raises(ValueError, match="provenance"):
        load_stage(path, triplet, expected_hash=checksum, config_hash="cfg", upstream="stale", block=1)
    candidate.gate[~candidate.keep] = 1
    originals = tuple(t.clone() for t in triplet.weights)
    with pytest.raises(ValueError, match="invariant"):
        apply_candidate(triplet, candidate)
    assert all(torch.equal(a, b) for a, b in zip(originals, triplet.weights))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_stage(path, triplet, expected_hash=checksum, config_hash="cfg", upstream="dense", block=1)


def test_hf_qwen_self_contained_loader_generation_and_runtime(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from sparsifier.eval.gsm8k_whole_neuron import reference_backend
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    from sparsifier.prune.whole_neuron_io import benchmark_compact
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    reference_backend()
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=["linear_attention","full_attention"],
        rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.5},
        eos_token_id=31, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    kept = {}
    for b, fraction in enumerate((.25, .5)):
        triplet = SwiGLUTriplet(model.model.layers[b].mlp)
        candidate = prune_triplet(triplet, torch.randn(40, 24), WholeNeuronConfig(fraction))
        apply_candidate(triplet, candidate)
        kept[b] = candidate.keep
    ids = torch.tensor([[1, 3, 5, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        generated = model.generate(ids, do_sample=False, max_new_tokens=3)
    export_compact(model, tmp_path, kept)
    loaded, _ = load_compact(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)
        assert torch.equal(loaded.generate(ids, do_sample=False, max_new_tokens=3), generated)
    runtime = benchmark_compact(loaded, ids, repeats=2, decode_tokens=2)
    assert len(runtime["prefill_seconds"]) == len(runtime["decode_seconds"]) == 2
    assert all(value > 0 for value in runtime["decode_seconds"])
