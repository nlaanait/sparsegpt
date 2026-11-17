import copy
import json
import numpy as np
import pytest
import torch
from test_whole_neuron import TinySwiGLU
from test_whole_neuron_export import TinyModel
from sparsifier.prune.whole_neuron import SwiGLUTriplet, WholeNeuronConfig, prune_triplet, apply_candidate
from sparsifier.prune.whole_neuron_io import save_stage, file_hash
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k_whole_neuron import (
    triplet_sites, calibrate, reference_backend, validate_backward, calibration_positions,
    cast_finite_check, restore_chain, memory_estimate,
)
from sparsifier.eval.gsm8k_downstream import AnchorPrefix


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def encode(self, *args, **kwargs):
        return [(i % 9)+1 for i in range(128)]


class Model(TinyModel):
    def forward(self, input_ids, **kwargs):
        return super().forward(input_ids)


def rows():
    return [{"id": str(i), "family": str(i%3), "question": "q",
             "split": "geometry_fit" if i < 65 else "pruning_calibration"} for i in range(130)]


def test_triplet_scope_memory_and_ambient_budgets():
    assert triplet_sites([0, 2], ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"], 3) == [(0,"mlp_triplet"),(2,"mlp_triplet")]
    with pytest.raises(ValueError, match="all three"):
        triplet_sites([0], ["mlp.down_proj"], 3)
    with pytest.raises(ValueError):
        triplet_sites([1, 0], ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"], 3)
    c4 = np.tile(np.arange(1, 129) % 9, (32, 1))
    provenance = calibration_positions(Tokenizer(), rows(), c4, 0)
    assert len(provenance["math"]) == len(provenance["c4"]) == 4096
    assert len(provenance["geometric_readouts"]) == 130
    assert memory_estimate(3, 7, .5)["retained_factor_width"] == 4


def test_sequential_calibration_reflects_complete_upstream_triplet_and_alpha_zero_bypasses_geometry(tmp_path, monkeypatch):
    import sparsifier.eval.gsm8k_whole_neuron as module
    def no_geometry(*args, **kwargs):
        raise AssertionError("alpha-zero must not fit charts or choose probes")
    monkeypatch.setattr(module, "FactorAtlas", no_geometry)
    monkeypatch.setattr(module, "probe_rows", no_geometry)
    torch.manual_seed(3)
    model = Model().requires_grad_(False)
    c4 = np.tile(np.arange(1, 129) % 9, (32, 1))
    config = WholeNeuronConfig(.5, "downstream_weight_diagonal", alpha=0)
    _, before, k, _, _ = calibrate(model, Tokenizer(), 1, rows(), c4, config, tmp_path)
    assert k is None and before.shape == (8192, 7)
    first = SwiGLUTriplet(model.model.layers[0].mlp)
    candidate = prune_triplet(first, torch.randn(40, 7), WholeNeuronConfig(.6))
    apply_candidate(first, candidate)
    _, after, _, _, _ = calibrate(model, Tokenizer(), 1, rows(), c4, config, tmp_path)
    assert not torch.equal(before, after)
    assert cast_finite_check(candidate, torch.randn(1, 8, 3))["finite"]
    with pytest.raises(ValueError, match="held-out"):
        calibrate(model, Tokenizer(), 1, [{"split":"validation"}], c4, config, tmp_path)


@pytest.mark.parametrize("anchor_type", ["linear_attention", "full_attention"])
def test_actual_reference_backend_tiny_qwen_downstream_finite_changes(anchor_type, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    identities = reference_backend()
    assert len(identities) == 4
    assert all(not isinstance(getattr(modeling, name), torch.nn.Module) for name in identities)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=["linear_attention",anchor_type],
        rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.5})
    config._attn_implementation = "sdpa"
    torch.manual_seed(41)
    model = modeling.Qwen3_5TextModel(config).eval().requires_grad_(False)
    # Validate actual MLP graph as well as the down-output backward path.
    triplet = SwiGLUTriplet(model.layers[0].mlp)
    anchor = model.layers[1]
    original = triplet.down.weight.clone()
    report = validate_backward(AnchorPrefix(model, anchor), triplet.down, anchor,
                              torch.tensor([[1, 4, 7, 2]]), rtol=.03, atol=1e-6)
    assert report["passed"]
    torch.testing.assert_close(triplet.down.weight, original)
    assert not triplet.down._forward_hooks and not anchor._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_transaction_chain_requires_all_sites_and_original_weight_hashes(tmp_path):
    model = TinyModel()
    from sparsifier.prune.whole_neuron import VERSION
    config = {"method_version": VERSION, "sites": [[0,"mlp_triplet"],[1,"mlp_triplet"]]}
    (tmp_path/"config.json").write_text(json.dumps(config))
    upstream = digest(config)
    stages = []
    original = copy.deepcopy(model)
    for n in range(2):
        triplet = SwiGLUTriplet(model.model.layers[n].mlp)
        candidate = prune_triplet(triplet, torch.randn(40, 7), WholeNeuronConfig(.5))
        checksum = save_stage(tmp_path/f"stage-{n:03d}.pt", triplet, candidate,
            config_hash=digest(config), upstream=upstream, block=n)
        apply_candidate(triplet, candidate)
        stages.append({"site":[n,"mlp_triplet"],"sha256":checksum})
        upstream = checksum
    (tmp_path/"report.json").write_text(json.dumps({"complete":True,"config_sha256":digest(config),"stages":stages}))
    _, _, kept = restore_chain(original, tmp_path)
    assert len(kept) == 2
    for a, b in zip(original.parameters(), model.parameters()):
        torch.testing.assert_close(a, b)
    # Reapplying to an already pruned reference is a provenance error.
    with pytest.raises(ValueError, match="hash"):
        restore_chain(original, tmp_path)


def test_paired_report_uses_actual_removal_and_rejects_protocol_drift(tmp_path):
    from sparsifier.eval.gsm8k_whole_neuron_report import compare
    paths = []
    for index in range(2):
        path = tmp_path/str(index)
        path.mkdir()
        config = {"model":"toy","revision":"fixed","manifest_sha256":"manifest",
                  "max_new_tokens":2048,"evaluation_ids":["a","b"], "arm":"whole_neuron_ambient"}
        (path/"config.json").write_text(json.dumps(config))
        (path/"report.json").write_text(json.dumps({"complete":True,"config_sha256":digest(config),
            "removed_weights":30+index,"total_parameters":100,"eligible_parameters":90}))
        (path/"answers.jsonl").write_text("\n".join(json.dumps({"id":i,"config_sha256":digest(config),
            "correct":index == 0 or i == "a", "generated_tokens":3}) for i in ("a","b")))
        paths.append(path)
    result = compare(paths, draws=50)
    assert result["comparisons"][1]["paired_accuracy_difference"] == -.5
    assert not result["comparisons"][1]["matched_removed_weight_budget"]
    config["max_new_tokens"] = 512
    (paths[1]/"config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatched"):
        compare(paths)


def test_original_validation_checks_bound_manifest_and_controlled_parent_sources():
    from sparsifier.eval.gsm8k_whole_neuron_evaluate import validate_source_separation
    calibration = {'manifest_sha256': 'source-hash', 'rows': [
        {'id': 'controlled/42/variant', 'source_index': 42, 'split': 'geometry_fit'},
        {'id': 'controlled/43/variant', 'source_index': 43, 'split': 'pruning_calibration'}]}
    source = {'manifest_sha256': 'source-hash', 'calibration_ids': ['controlled/42/variant', 'controlled/43/variant']}
    validation = {'rows': [{'id': 'gsm8k/train/1001', 'source_index': 1001, 'split': 'validation'}]}
    assert validate_source_separation(source, calibration, validation)['disjoint_parent_sources']
    # A new manifest hash/validation label cannot make a used parent disjoint.
    leaked = {'manifest_sha256': 'independently-hashed', 'rows': [
        {'id': 'gsm8k/train/42', 'source_index': 42, 'split': 'validation'}]}
    with pytest.raises(ValueError, match='overlap'):
        validate_source_separation(source, calibration, leaked)
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_source_separation(source, dict(calibration, manifest_sha256='wrong'), validation)
    with pytest.raises(ValueError, match='source_index'):
        validate_source_separation(source, calibration, {'rows': [{'id': 'gsm8k/train/1001'}]})
    with pytest.raises(ValueError, match='disagree'):
        validate_source_separation(source, calibration,
            {'rows': [{'id': 'gsm8k/train/1001', 'source_index': 1002}]})


def test_leaked_validation_is_rejected_before_model_or_cuda_loading(tmp_path, monkeypatch):
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.eval import gsm8k_whole_neuron_evaluate as module
    from sparsifier.eval.gsm8k import MODEL, REVISION
    cal = {'rows': [{'id':'controlled/42/variant', 'source_index':42, 'split':'geometry_fit'}]}
    cal['manifest_sha256'] = digest(cal)
    val = {'source_split':'train', 'rows':[{'id':'gsm8k/train/42', 'source_index':42, 'split':'validation'}]}
    val['manifest_sha256'] = digest(val)
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text(json.dumps({'model':MODEL, 'revision':REVISION,
        'manifest_sha256':cal['manifest_sha256'], 'calibration_ids':['controlled/42/variant']}))
    (tmp_path/'cal.json').write_text(json.dumps(cal))
    (tmp_path/'val.json').write_text(json.dumps(val))
    def forbidden(*args, **kwargs):
        raise AssertionError('leaked validation must fail before model/CUDA loading')
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', forbidden)
    monkeypatch.setattr(module, 'configure_cuda', forbidden)
    monkeypatch.setattr(sys, 'argv', ['evaluate','--source',str(source),'--manifest',str(tmp_path/'val.json'),
        '--calibration-manifest',str(tmp_path/'cal.json'),'--output',str(tmp_path/'output')])
    with pytest.raises(ValueError, match='overlap'):
        module.main()
    assert not (tmp_path/'output').exists()
