"""Exact ``L_tan`` OBS: diagonal-P row Hessians and small Kronecker path."""

from __future__ import annotations

import torch
from torch import nn

from sparsifier.prune.exact_ltan import (
    ExactLtanSparseGPT,
    diag_p_obs_prune,
    diag_p_row_hessians,
    greedy_full_obs,
    kronecker_projected_hessian,
)
from sparsifier.prune.pruners import get_pruner, list_pruners


def test_exact_obs_is_registered():
    assert "tangent_exact_obs" in list_pruners()


def test_diag_p_row_hessian_scales_with_metric():
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Only output-0 is active on all samples.
    diags = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    h = diag_p_row_hessians(inputs, diags, percdamp=0.0)
    assert h.shape == (2, 2, 2)
    assert float(h[1].abs().sum()) == 0.0
    assert float(h[0, 0, 0]) > 0.0


def test_diag_p_obs_prunes_and_compensates_within_row():
    torch.manual_seed(0)
    weight = torch.randn(3, 4)
    diags = torch.ones(16, 3)
    inputs = torch.randn(16, 4)
    h = diag_p_row_hessians(inputs, diags, percdamp=0.01)
    sparse = diag_p_obs_prune(weight, h, sparsity=0.25)
    assert int((sparse == 0).sum()) == int(0.25 * weight.numel())
    # Survivors are generally shifted by compensation (not pure magnitude mask).
    alive = sparse != 0
    assert not torch.allclose(sparse[alive], weight[alive])


def test_full_kronecker_matches_diag_metric_block_structure():
    inputs = torch.randn(8, 3)
    # Diagonal metrics → Kronecker couples only within each output row block
    # under Fortran vec, but still a valid PSD Hessian.
    metrics = torch.stack([torch.diag(torch.rand(2)) for _ in range(8)])
    h = kronecker_projected_hessian(inputs, metrics, ridge=1e-5)
    assert h.shape == (6, 6)
    evals = torch.linalg.eigvalsh(0.5 * (h + h.T))
    assert float(evals.min()) > -1e-6


def test_greedy_full_obs_reaches_requested_sparsity():
    torch.manual_seed(1)
    weight = torch.randn(2, 3)
    metrics = torch.eye(2).expand(10, 2, 2).clone()
    inputs = torch.randn(10, 3)
    h = kronecker_projected_hessian(inputs, metrics, ridge=1e-4)
    sparse = greedy_full_obs(h, weight, sparsity=0.5)
    assert int((sparse == 0).sum()) == 3


def test_exact_ltan_pruner_diag_mode_end_to_end():
    torch.manual_seed(0)
    layer = nn.Linear(5, 4, bias=False)
    bound = get_pruner(
        "tangent_exact_obs",
        charts=2,
        rank=1,
        calibration_samples=32,
        seed=0,
        mode="diag",
    ).bind(layer, block_index=0, module_name="x")
    x = torch.randn(4, 16, 5)
    # Separated outputs so chart PCA is stable.
    with torch.no_grad():
        y = layer(x)
        y[..., 0] += 3.0
    bound.add_batch(x, y)
    bound.finalize()
    bound.prune(sparsity=0.25, blocksize=2)
    zeros = int((layer.weight.data == 0).sum())
    assert zeros == int(0.25 * layer.weight.numel())


def test_exact_ltan_full_mode_on_tiny_layer():
    torch.manual_seed(0)
    layer = nn.Linear(3, 2, bias=False)
    solver = ExactLtanSparseGPT(
        layer, charts=2, rank=1, calibration_samples=16, seed=0, mode="full", max_full_params=64
    )
    x = torch.randn(2, 12, 3)
    y = layer(x)
    y = y + torch.tensor([2.0, -2.0])
    solver.add_batch(x, y)
    solver.finalize_charts()
    assert solver.resolved_mode == "full"
    solver.fasterprune(0.5)
    assert int((layer.weight.data == 0).sum()) == 3
