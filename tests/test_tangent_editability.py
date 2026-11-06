"""Contracts for the tangent_editability ablation pruner."""

from __future__ import annotations

import torch
from torch import nn

from sparsifier.prune.editability import EditabilitySparseGPT
from sparsifier.prune.knockout import KnockoutSparseGPT
from sparsifier.prune.pruners import get_pruner


def test_editability_scores_omit_weight_magnitude_vs_knockout():
    """Editability is G·E[φ²]; knockout multiplies by W²."""
    torch.manual_seed(0)
    layer_small = nn.Linear(4, 3, bias=False)
    layer_large = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        base = torch.randn(3, 4)
        layer_small.weight.copy_(base)
        layer_large.weight.copy_(10.0 * base)
    layer_knock = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer_knock.weight.copy_(layer_small.weight)
    kwargs = dict(charts=2, rank=1, calibration_samples=32, seed=0)
    edit_small = EditabilitySparseGPT(layer_small, **kwargs)
    edit_large = EditabilitySparseGPT(layer_large, **kwargs)
    knock = KnockoutSparseGPT(layer_knock, **kwargs)
    x = torch.randn(4, 8, 4).abs() + 0.1
    y = torch.zeros(4, 8, 3)
    y[..., 0] = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    y += 0.01 * torch.randn(4, 8, 3)
    for solver in (edit_small, edit_large, knock):
        solver.add_batch(x, y)
        solver.finalize_scores()
    assert edit_small.editability_scores is not None
    assert edit_large.editability_scores is not None
    assert knock.knockout_scores is not None
    # Same (x, y) => same editability regardless of |W|.
    torch.testing.assert_close(
        edit_small.editability_scores, edit_large.editability_scores, atol=1e-5, rtol=1e-5
    )
    w2 = layer_small.weight.detach().cpu().float().square()
    torch.testing.assert_close(
        knock.knockout_scores,
        edit_small.editability_scores * w2,
        atol=1e-5,
        rtol=1e-5,
    )
    # On-tangent row still outranks off-tangent under editability (equal structure).
    assert float(edit_small.editability_scores[0].mean()) > float(
        edit_small.editability_scores[1].mean()
    )


def test_editability_registry_binds_and_prunes_tiny_linear():
    layer = nn.Linear(6, 4, bias=True)
    bound = get_pruner(
        "tangent_editability",
        charts=2,
        rank=1,
        calibration_samples=24,
        seed=0,
    ).bind(layer, block_index=0, module_name="x")
    x = torch.randn(2, 12, 6)
    bound.add_batch(x, layer(x))
    bound.finalize()
    before = layer.weight.data.clone()
    bound.prune(sparsity=0.25, blocksize=3)
    assert int((layer.weight.data == 0).sum()) > 0
    assert not torch.equal(layer.weight.data, before)
    bound.free()


def test_editability_bias_fold_flag_accepted():
    pruner = get_pruner(
        "tangent_editability",
        charts=2,
        rank=1,
        calibration_samples=16,
        seed=0,
        bias_fold=True,
    )
    layer = nn.Linear(4, 3, bias=True)
    bound = pruner.bind(layer, block_index=0, module_name="x")
    assert getattr(bound, "bias_fold", False) is True
