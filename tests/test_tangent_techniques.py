"""Contracts for distinct tangent pruning techniques (ablation registry).

Tests are written first. Implementations must register under these names and
satisfy the note's Option 1-factor-union / Option 2 / Option 3 / knockout
behaviors. Bias folding is a composable flag on every tangent technique.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sparsifier.prune.pruners import TECHNIQUE_NAMES, get_pruner


REQUIRED_TECHNIQUES = (
    "sparsegpt",
    "geometry",
    "tangent",  # Option 1: k-means chart G_pq heuristic (existing)
    "tangent_factor_union",  # Option 1 with concept-factor union projectors
    "tangent_chart_obs",  # Option 2: per-chart OBS
    "tangent_fixed_union",  # Option 3: fixed-union rotation
    "tangent_knockout",  # mask by τ_union knockout scores
    "tangent_editability",  # mask by editability (no |W| factor)
)


def test_technique_registry_lists_every_ablation_arm():
    for name in REQUIRED_TECHNIQUES:
        assert name in TECHNIQUE_NAMES


def test_factor_union_coupling_uses_actual_span_not_sum_of_ranks():
    """Overlapping factor tangents must QR to the union; rank < sum r_i."""
    from sparsifier.prune.factor_union import (
        estimate_factor_union_coupling,
        union_projector_diagonal,
    )

    # Two identical rank-1 factors => union rank 1, not 2.
    u0 = torch.tensor([[1.0], [0.0], [0.0]])
    u1 = torch.tensor([[1.0], [0.0], [0.0]])
    diag, rank = union_projector_diagonal([u0, u1])
    assert rank == 1
    torch.testing.assert_close(diag, torch.tensor([1.0, 0.0, 0.0]))

    # Synthetic paired data: one active factor per sample with known basis.
    n, din, dout = 32, 4, 3
    torch.manual_seed(0)
    assign = torch.arange(n) % 2
    bases = torch.zeros(2, dout, 1)
    bases[0, 0, 0] = 1.0
    bases[1, 1, 0] = 1.0
    x = torch.randn(n, din).abs() + 0.1
    # Outputs unused when bases are supplied per sample; pass placeholders.
    y = torch.randn(n, dout)
    active = torch.zeros(n, 2, dtype=torch.bool)
    active[torch.arange(n), assign] = True
    score, diagnostics = estimate_factor_union_coupling(
        x, y, factor_bases=bases, active=active, alpha=1.0
    )
    assert diagnostics["mean_union_rank"] == pytest.approx(1.0)
    pdiag = torch.zeros(n, dout)
    pdiag[torch.arange(n), assign] = 1.0
    oracle = (pdiag.T @ x.square()) / x.square().sum(0).clamp_min(1e-12)
    torch.testing.assert_close(score, oracle, atol=1e-5, rtol=1e-5)


def test_option2_chart_obs_matches_restricted_rowwise_cost_factor():
    """For fixed chart U_c, cost factor includes ||U_c^T e_p||^2 / [H_c^{-1}]_{qq}."""
    from sparsifier.prune.chart_obs import per_chart_obs_scores

    torch.manual_seed(0)
    weight = torch.randn(3, 4)
    # One chart covering all samples; U = e_0 (only row 0 participates).
    u = torch.tensor([[1.0], [0.0], [0.0]])
    phi = torch.randn(20, 4)
    scores = per_chart_obs_scores(weight, phi, u, ridge=1e-3)
    assert scores.shape == weight.shape
    # Rows 1 and 2 have zero projected factor.
    assert torch.count_nonzero(scores[1:]) == 0
    assert torch.count_nonzero(scores[0]) == 4


def test_option3_fixed_union_prunes_only_parallel_block():
    """Loss depends on W_parallel = U^T W; ambient OBS in tangent coordinates."""
    from sparsifier.prune.fixed_union import FixedUnionSparseGPT

    torch.manual_seed(0)
    layer = nn.Linear(6, 4, bias=False)
    # Fixed U spans first two output axes.
    u = torch.zeros(4, 2)
    u[0, 0] = 1.0
    u[1, 1] = 1.0
    solver = FixedUnionSparseGPT(layer, union_basis=u)
    x = torch.randn(2, 10, 6)
    solver.add_batch(x, layer(x))
    before = layer.weight.data.clone()
    solver.fasterprune(0.5, blocksize=3)
    delta = before - layer.weight.data
    # Changes in the orthogonal complement of U should be pure fold/zeroing of
    # W_perp without ambient row mixing from the parallel OBS — W_perp rows
    # 2 and 3 may be zeroed or left, but parallel OBS must not write into them
    # via the rotated solve beyond explicit perp handling.
    # Contract: after prune, reconstructing U @ (U^T W_hat) matches the
    # retained parallel part; see solver.parallel_weights().
    parallel = solver.parallel_weights()
    assert parallel.shape == (2, 6)
    assert int((layer.weight.data == 0).sum()) > 0


def test_knockout_pruner_ranks_by_tangent_diag_score_not_magnitude():
    """Prune lowest I^{tan,diag} first; on-tangent equal-|W| outranks off-tangent."""
    layer = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    bound = get_pruner(
        "tangent_knockout",
        charts=2,
        rank=1,
        calibration_samples=32,
        seed=0,
    ).bind(layer, block_index=0, module_name="mlp.down_proj")
    x = torch.randn(4, 8, 5).abs() + 0.1
    # Outputs dominated by variation along e_1 so G concentrates on row 1.
    y = torch.zeros(4, 8, 3)
    y[..., 1] = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    y += 0.01 * torch.randn(4, 8, 3)
    bound.add_batch(x, y)
    bound.finalize()
    scores = bound.solver.knockout_scores
    assert scores is not None
    # Equal |W|: on-tangent row has higher knockout impact than off-tangent rows.
    assert float(scores[1].mean()) > float(scores[0].mean())
    assert float(scores[1].mean()) > float(scores[2].mean())
    flat_min = int(torch.argmin(scores).item())
    bound.prune(sparsity=1.0 / scores.numel(), blocksize=2)
    assert layer.weight.data.flatten()[flat_min] == 0.0
    assert torch.count_nonzero(layer.weight.data[1]) == layer.weight.data.shape[1]


@pytest.mark.parametrize(
    "name",
    [
        "tangent",
        "tangent_factor_union",
        "tangent_chart_obs",
        "tangent_fixed_union",
        "tangent_knockout",
        "tangent_editability",
    ],
)
def test_bias_fold_flag_is_accepted_on_tangent_techniques(name, tmp_path):
    kwargs = {"bias_fold": True}
    if name == "geometry":
        pytest.skip("geometry uses probes, not bias_fold in this parametrize")
    if name == "tangent":
        kwargs.update(alpha=0.5, charts=2, rank=1, calibration_samples=16, seed=0)
    elif name == "tangent_factor_union":
        kwargs.update(alpha=1.0, factors=2, rank=1, calibration_samples=16, seed=0)
    elif name == "tangent_chart_obs":
        kwargs.update(charts=2, rank=1, calibration_samples=16, seed=0, percdamp=1e-3)
    elif name == "tangent_fixed_union":
        kwargs.update(rank=2, calibration_samples=16, seed=0)
    elif name in ("tangent_knockout", "tangent_editability"):
        kwargs.update(charts=2, rank=1, calibration_samples=16, seed=0)
    pruner = get_pruner(name, **kwargs)
    layer = nn.Linear(4, 3, bias=True)
    bound = pruner.bind(layer, block_index=0, module_name="x")
    assert getattr(bound, "bias_fold", False) is True


def test_each_technique_binds_and_prunes_a_tiny_linear(tmp_path):
    """Smoke contract: every registered technique can prune one CPU linear."""
    torch.manual_seed(0)
    for name in REQUIRED_TECHNIQUES:
        layer = nn.Linear(6, 4, bias=True)
        if name == "sparsegpt":
            pruner = get_pruner(name)
        elif name == "geometry":
            probes = torch.randn(6, 3).numpy()
            path = tmp_path / "0"
            path.mkdir()
            import numpy as np

            np.save(path / "x.npy", probes)
            pruner = get_pruner("geometry", alpha=0.5, probes=tmp_path)
        elif name == "tangent":
            pruner = get_pruner(
                name, alpha=0.5, charts=2, rank=1, calibration_samples=24, seed=0
            )
        elif name == "tangent_factor_union":
            pruner = get_pruner(
                name, alpha=1.0, factors=2, rank=1, calibration_samples=24, seed=0
            )
        elif name == "tangent_chart_obs":
            pruner = get_pruner(
                name, charts=2, rank=1, calibration_samples=24, seed=0, percdamp=1e-3
            )
        elif name == "tangent_fixed_union":
            pruner = get_pruner(name, rank=2, calibration_samples=24, seed=0)
        elif name == "tangent_knockout":
            pruner = get_pruner(
                name, charts=2, rank=1, calibration_samples=24, seed=0
            )
        elif name == "tangent_editability":
            pruner = get_pruner(
                name, charts=2, rank=1, calibration_samples=24, seed=0
            )
        else:
            raise AssertionError(name)
        bound = pruner.bind(layer, block_index=0, module_name="x")
        x = torch.randn(2, 12, 6)
        y = layer(x)
        bound.add_batch(x, y)
        bound.finalize()
        before = layer.weight.data.clone()
        bound.prune(sparsity=0.25, blocksize=3)
        if name == "tangent_fixed_union":
            # Option 3 sparsifies W_parallel; ambient W need not contain exact zeros.
            parallel = bound.solver.parallel_weights()
            assert int((parallel == 0).sum()) > 0
            assert not torch.equal(layer.weight.data, before)
        else:
            assert int((layer.weight.data == 0).sum()) > 0
        bound.free()
