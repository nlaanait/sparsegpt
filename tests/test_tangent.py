"""Unit tests for Option-1 output-tangent coupling and TangentAwareSparseGPT.

Locks the note formulas for G_pq, inactive features, alpha blend, and the
mask-weight path that keeps ambient SparseGPT compensation.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sparsifier.prune.pruners import get_pruner
from sparsifier.prune.tangent import (
    TangentAwareSparseGPT,
    estimate_output_tangent_coupling,
)


def _two_chart_data(n: int = 40, din: int = 5, dout: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    assign = torch.arange(n) % 2
    centers = torch.tensor([[2.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0]])[:, :dout]
    y = centers[assign] + 0.02 * torch.randn(n, dout)
    # Distinct local variation per chart along orthogonal ambient axes.
    for chart in (0, 1):
        mask = assign == chart
        direction = torch.zeros(dout)
        direction[chart] = 1.0
        y[mask] = y[mask] + 0.4 * torch.randn(int(mask.sum()), 1) * direction
    x = torch.randn(n, din).abs() + 0.05
    return x, y, assign


def test_g_pq_matches_oracle_hard_chart_projectors():
    """G_pq = sum_n (P_n)_pp phi_q,n^2 / sum_n phi_q,n^2 for known rank-1 charts."""
    n, din, dout = 60, 4, 5
    torch.manual_seed(1)
    assign = torch.arange(n) % 2
    centers = torch.tensor([[3.0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0]])
    y = centers[assign].clone()
    y[:, 0] += (assign == 0).float() * 0.5 * torch.randn(n)
    y[:, 1] += (assign == 1).float() * 0.5 * torch.randn(n)
    y += 1e-3 * torch.randn(n, dout)
    x = torch.randn(n, din).abs() + 0.1

    pdiag = torch.zeros(n, dout)
    pdiag[assign == 0, 0] = 1.0
    pdiag[assign == 1, 1] = 1.0
    oracle = (pdiag.T @ x.square()) / x.square().sum(0).clamp_min(1e-12)

    score, diagnostics = estimate_output_tangent_coupling(
        x, y, charts=2, rank=1, alpha=1.0, seed=0
    )
    assert diagnostics["chart_count"] == 2
    assert diagnostics["chart_ranks"] == [1, 1]
    assert score.shape == (dout, din)
    torch.testing.assert_close(score[0], oracle[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(score[1], oracle[1], atol=1e-5, rtol=1e-5)
    assert float(score[2:].abs().max()) < 1e-4


def test_alpha_blend_is_convex_combination_with_ambient():
    x, y, _ = _two_chart_data()
    pure, _ = estimate_output_tangent_coupling(x, y, charts=2, rank=2, alpha=1.0, seed=0)
    blended, diagnostics = estimate_output_tangent_coupling(
        x, y, charts=2, rank=2, alpha=0.4, seed=0
    )
    expected = (1.0 - 0.4) + 0.4 * pure
    torch.testing.assert_close(blended, expected)
    assert 0.0 < diagnostics["mean_score_weight"] <= 1.0


def test_inactive_input_features_get_zero_coupling():
    x, y, _ = _two_chart_data()
    x = x.clone()
    x[:, 2] = 0.0
    score, diagnostics = estimate_output_tangent_coupling(
        x, y, charts=2, rank=2, alpha=1.0, seed=0
    )
    assert diagnostics["inactive_input_features"] == 1
    assert torch.count_nonzero(score[:, 2]) == 0


def test_coupling_rejects_bad_shapes_and_alpha():
    x, y, _ = _two_chart_data()
    with pytest.raises(ValueError, match="alpha"):
        estimate_output_tangent_coupling(x, y, alpha=1.5)
    with pytest.raises(ValueError, match="paired"):
        estimate_output_tangent_coupling(x[:10], y)
    with pytest.raises(ValueError, match="three"):
        estimate_output_tangent_coupling(x[:2], y[:2])


def test_tangent_aware_solver_requires_outputs_and_applies_mask_weight():
    layer = nn.Linear(6, 4, bias=False)
    solver = TangentAwareSparseGPT(
        layer, alpha=1.0, charts=2, rank=1, calibration_samples=32, seed=0
    )
    inputs = torch.randn(4, 8, 6)
    with pytest.raises(TypeError, match="outputs"):
        solver.add_batch(inputs, None)

    outputs = torch.randn(4, 8, 4)
    solver.add_batch(inputs, outputs)
    solver.finalize_tangents()
    assert solver.finalized
    assert solver.mask_weight is not None
    assert solver.mask_weight.shape == layer.weight.shape
    assert solver.tangent_diagnostics["sample_count"] == 32

    before = layer.weight.data.clone()
    solver.fasterprune(0.25, blocksize=3)
    assert not torch.equal(layer.weight.data, before)
    assert int((layer.weight.data == 0).sum()) > 0
    solver.free()


def test_tangent_pruner_registry_bind_and_prune():
    layer = nn.Linear(5, 3, bias=False)
    bound = get_pruner(
        "tangent", alpha=0.7, charts=2, rank=1, calibration_samples=24, seed=1
    ).bind(layer, block_index=0, module_name="mlp.down_proj")
    x = torch.randn(3, 7, 5)
    bound.add_batch(x, layer(x))
    bound.finalize()
    bound.prune(sparsity=0.4, blocksize=2)
    assert int((layer.weight.data == 0).sum()) > 0
    bound.free()


def test_tangent_and_geometry_are_mutually_exclusive():
    from sparsifier.prune.pruners import bind_layer

    layer = nn.Linear(4, 2, bias=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        bind_layer(
            layer,
            block_index=0,
            module_name="x",
            geometry_alpha=0.5,
            geometry_probes=None,
            tangent_alpha=0.5,
        )


def test_g_uses_sample_second_moments_not_ridge():
    """Denominator is sum phi^2; ridge must not inflate inactive/near-zero features."""
    x, y, _ = _two_chart_data(n=30, din=3, dout=4, seed=3)
    x = x.clone()
    x[:, 1] = 1e-8  # near-zero but not exactly inactive under 1e-12 clamp
    score, _ = estimate_output_tangent_coupling(x, y, charts=2, rank=1, alpha=1.0, seed=0)
    # Finite and in [0, 1]; no NaNs from a ridge-inflated covariance diagonal.
    assert torch.isfinite(score).all()
    assert float(score.min()) >= 0.0
    assert float(score.max()) <= 1.0 + 1e-6


def test_mask_weight_multiplies_sparsegpt_metric_not_hessian():
    """Option 1 reweights mask scores only; ambient H is unchanged vs SparseGPT."""
    from sparsifier.prune.sparsegpt import SparseGPT

    torch.manual_seed(0)
    layer_a = nn.Linear(8, 4, bias=False)
    layer_b = nn.Linear(8, 4, bias=False)
    layer_b.load_state_dict(layer_a.state_dict())
    x = torch.randn(2, 16, 8)
    y = layer_a(x)

    ambient = SparseGPT(layer_a)
    ambient.add_batch(x, y)
    h_ambient = ambient.H.clone()

    tangent = TangentAwareSparseGPT(
        layer_b, alpha=1.0, charts=2, rank=1, calibration_samples=32, seed=0
    )
    tangent.add_batch(x, y)
    assert torch.allclose(tangent.H, h_ambient)
    tangent.finalize_tangents()
    assert torch.allclose(tangent.H, h_ambient)
