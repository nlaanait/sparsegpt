import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.eval.gsm8k_batching import pack, policy, trim_generated
from sparsifier.eval.gsm8k_capture import capture_batch, capture_one
from sparsifier.eval.gsm8k_pruning import answer_likelihoods, behavior


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        if text.startswith(" "):
            return [3] if text.strip() == "1" else [4, 4]
        question = int(text.rsplit("\n\n", 1)[-1])
        return [1]*(question % 3+1)+[question+1]

    def decode(self, tokens, **kwargs):
        return "Final answer: 1"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 4)
        self.layer = nn.Linear(4, 3)
        self.head = nn.Linear(3, 32)
        self.generation_config = SimpleNamespace(eos_token_id=[2, 7])
        self.generated_batches = []

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kwargs):
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        hidden = (self.embedding(input_ids)*mask[..., None]).cumsum(1)
        logits = self.head(self.layer(hidden))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **kwargs):
        assert max_new_tokens == 4
        self.generated_batches.append(input_ids[:, -1].tolist())
        sequences = [[3, 2, 0, 0], [3, 4, 5, 6], [3, 7, 0, 0]]
        result = [sequences[(int(i)-1) % 3] for i in input_ids[:, -1]]
        return torch.cat((input_ids, torch.tensor(result)), dim=1)


def rows(count=8):
    return [{"id": str(i), "question": str(i), "family": "test", "answer": "1"} for i in range(count)]


def test_capture_uses_each_unpadded_sequence_and_unchanged_seeds():
    torch.manual_seed(1)
    model = Model().eval()
    tokenizer = Tokenizer()
    sequences = [tokenizer.encode(str(i)) for i in range(8)]
    seeds = list(range(11, 19))
    captured = capture_batch(model, model.layer, pack(sequences, tokenizer, "cpu"),
                             list(map(len, sequences)), seeds, token_samples=2)
    for sequence, seed, actual in zip(sequences, seeds, captured):
        expected = capture_one(model, model.layer, torch.tensor([sequence]), seed=seed, token_samples=2)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        assert actual["readout_position"] == len(sequence)-1
        assert actual["positions"].max() == len(sequence)-1
    assert not model.layer._forward_hooks


def test_teacher_forcing_aligns_variable_target_lengths_and_excludes_padding():
    torch.manual_seed(8)
    model, tokenizer = Model().eval(), Tokenizer()
    selected = rows()
    selected[1]["answer"] = "22"
    actual = answer_likelihoods(model, tokenizer, selected)
    expected = [answer_likelihoods(model, tokenizer, [r])[0] for r in selected]
    assert actual == pytest.approx(expected, abs=1e-6)


def test_generation_eos_cap_tail_and_partial_resume_keep_batch_membership(tmp_path):
    model, tokenizer = Model(), Tokenizer()
    selected = rows(37)
    behavior(model, tokenizer, selected, tmp_path, "config", max_new_tokens=4)
    assert model.generated_batches == [list(range(1, 33)), list(range(33, 38))]
    path = tmp_path/"answers.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["generated_tokens"] == 2 and not records[0]["truncated"]
    assert records[1]["generated_tokens"] == 4 and records[1]["truncated"]
    assert not records[1]["correct"] and records[1]["parsed_answer"] is None
    assert records[2]["generated_tokens"] == 2 and not records[2]["truncated"]
    assert all(r["batching"] == policy() for r in records)
    path.write_text("".join(json.dumps(r)+"\n" for r in records[:33]))
    model.generated_batches.clear()
    behavior(model, tokenizer, selected, tmp_path, "config", max_new_tokens=4)
    assert model.generated_batches == [list(range(33, 38))]
    resumed = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(resumed) == len({r["id"] for r in resumed}) == 37
    assert [r["text"] for r in resumed] == [r["text"] for r in records]
    model.generated_batches.clear()
    behavior(model, tokenizer, selected, tmp_path, "config", max_new_tokens=4)
    assert not model.generated_batches


def test_pack_and_eos_validation():
    with pytest.raises(ValueError):
        pack([[]], Tokenizer(), "cpu")
    assert trim_generated([3, 2, 0, 0], {2}) == [3, 2]
    assert trim_generated([3, 4], {2}) == [3, 4]


def test_resume_rejects_legacy_singleton_records_even_with_matching_outer_hash(tmp_path):
    (tmp_path/"answers.jsonl").write_text(json.dumps({"id": "0", "config_sha256": "same"})+"\n")
    with pytest.raises(ValueError, match="batching provenance"):
        behavior(Model(), Tokenizer(), rows(), tmp_path, "same", max_new_tokens=4)


def test_rollout_has_matching_dense_and_keeps_canceled_local_pilots_excluded():
    from scripts.propagate_gsm8k_batch8 import rollout_plan
    from test_gsm8k_decision import manifests
    plan = rollout_plan(manifests())
    assert plan["batching"] == policy(size=8)  # Explicit historical batch-eight rollout.
    assert len(plan["tasks"]) == 12 and len(plan["replacements"]) == 9
    names = {t["name"] for t in plan["tasks"]}
    assert all("batch8" in n and "-local-" not in n for n in names)
    assert all(r["new_name"] in names for r in plan["replacements"].values())
    assert all(plan["dense"] in t["depends_on"] for t in plan["tasks"] if t["phase"] == "evaluate")


@pytest.mark.parametrize("reference_batch_size", [None, 8, 16])
def test_dense_reference_rejects_a_different_execution_policy(tmp_path, reference_batch_size):
    from sparsifier.data.gsm8k import digest
    from sparsifier.eval.gsm8k import MODEL, REVISION
    from sparsifier.eval.gsm8k_decision import dense_answers
    from test_gsm8k_decision import manifests
    manifest = manifests()
    config = {"arm": "dense", "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_ids": [r["id"] for r in manifest["rows"]], "backend": {}, "packages": {},
        "model": MODEL, "revision": REVISION, "max_new_tokens": 2048}
    if reference_batch_size is not None:
        config["batching"] = policy(size=reference_batch_size)
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"report.json").write_text(json.dumps({"complete": True, "config_sha256": digest(config)}))
    with pytest.raises(ValueError, match="protocol/runtime mismatch"):
        dense_answers(tmp_path, manifest, {}, {})


@pytest.mark.parametrize("anchor_type", ["linear_attention", "full_attention"])
def test_distinct_padded_examples_on_actual_qwen_reference_backend(anchor_type, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from sparsifier.eval.gsm8k_batching import reference_backend
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    reference_backend()
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=["linear_attention", anchor_type],
        rope_parameters={"rope_type":"default", "rope_theta":10000., "partial_rotary_factor":.5})
    config._attn_implementation = "sdpa"
    torch.manual_seed(41)
    model = modeling.Qwen3_5TextModel(config).eval().requires_grad_(False)
    sequences = [[1, 4, i+2, 2]+[i+2]*(i % 4) for i in range(8)]
    lengths, seeds = list(map(len, sequences)), list(range(8))
    target = model.layers[1].mlp.down_proj
    actual = capture_batch(model, target, pack(sequences, Tokenizer(), "cpu"), lengths, seeds)
    for sequence, seed, item in zip(sequences, seeds, actual):
        expected = capture_one(model, target, torch.tensor([sequence]), seed=seed)
        for key in expected:
            np.testing.assert_allclose(item[key], expected[key], rtol=5e-4, atol=1e-5)
