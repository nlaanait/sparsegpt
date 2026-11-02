"""Shared catcher, mask slicing, and the wandb guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sparsifier.models import init_wandb, log_perplexity
from sparsifier.models.loop import (
    Catcher,
    capture_cache,
    excluded,
    hessian_hooks,
    leading_example,
    report_perplexity,
)


def test_leading_example_slices_batch_not_mrope_axis():
    rope = torch.zeros(8, 4, 16)
    rope_batch4 = torch.zeros(4, 2048, 64)
    ids = torch.zeros(4, 8, 4, dtype=torch.long)
    cos, sin = leading_example((rope, rope))
    assert cos.shape == (1, 4, 16)
    assert sin.shape == (1, 4, 16)
    assert leading_example(rope_batch4).shape == (1, 2048, 64)
    assert leading_example(ids).shape == (4, 1, 4)


def test_catcher_stores_a_packed_batch():
    store = torch.zeros(4, 2, 3)
    cache = capture_cache()
    catcher = Catcher(nn.Identity(), store, cache, batched=True)
    rope = torch.ones(2, 2, 3)
    ids = torch.arange(12, dtype=torch.long).reshape(3, 2, 2)
    with pytest.raises(ValueError):
        catcher.forward(
            torch.ones(2, 2, 3),
            attention_mask=None,
            position_embeddings=(rope, rope),
            position_ids=ids,
        )
    assert cache["i"] == 2
    assert torch.equal(store[:2], torch.ones(2, 2, 3))
    assert cache["position_embeddings"][0].shape == (1, 2, 3)
    assert cache["position_ids"].shape == (3, 1, 2)


def test_catcher_keeps_one_squeezed_row():
    store = torch.zeros(1, 2, 3)
    hidden = torch.arange(6, dtype=torch.float).reshape(1, 2, 3)
    cache = capture_cache()
    catcher = Catcher(nn.Identity(), store, cache, squeeze=True)
    with pytest.raises(ValueError):
        catcher.forward(hidden)
    assert torch.equal(store[0], hidden[0])


def test_catcher_requires_the_llama_mask():
    store = torch.zeros(2, 2, 3)
    cache = capture_cache(positions=False)
    mask = torch.ones(1, 2)
    catcher = Catcher(
        nn.Identity(),
        store,
        cache,
        fields=("attention_mask",),
        leading=False,
        required=("attention_mask",),
    )
    with pytest.raises(ValueError):
        catcher.forward(torch.ones(2, 3), attention_mask=mask)
    assert cache["attention_mask"] is mask
    with pytest.raises(KeyError):
        catcher.forward(torch.ones(2, 3))


def test_hessian_hook_sees_the_linear_input():
    layer = nn.Linear(3, 2, bias=False)
    seen = []

    class _Bound:
        def add_batch(self, inputs, outputs):
            seen.append(inputs.clone())

    handles = hessian_hooks({"weight": _Bound()}, {"weight": layer})
    layer(torch.ones(1, 4, 3))
    for handle in handles:
        handle.remove()
    assert seen[0].shape == (1, 4, 3)


def test_excluded_matches_the_layer_filter():
    args = SimpleNamespace(minlayer=1, maxlayer=3, prune_only="mlp", invert=False)
    assert excluded(args, 0, "mlp.down_proj")
    assert not excluded(args, 1, "mlp.down_proj")
    assert excluded(args, 1, "self_attn.q_proj")


def test_report_perplexity_prints_without_wandb(capsys):
    nlls = [torch.zeros(()) for _ in range(2)]
    score = report_perplexity(nlls, 2, 1, "wikitext2", False)
    assert score == pytest.approx(1.0)
    assert "Perplexity:" in capsys.readouterr().out


def test_wandb_logging_is_off_unless_asked():
    init_wandb(SimpleNamespace(log_wandb=False))
    log_perplexity("wikitext2", 1.0, False)


def test_wandb_logging_refuses_when_missing(monkeypatch):
    import sparsifier.models as models

    monkeypatch.setattr(models, "_wandb", None)
    with pytest.raises(RuntimeError, match="wandb"):
        models.init_wandb(SimpleNamespace(log_wandb=True))
    with pytest.raises(RuntimeError, match="wandb"):
        models.log_perplexity("wikitext2", 1.0, True)
