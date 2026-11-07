"""Revised-note presence, Fisher, CG refit, sparse-plus-low-rank, write alignment."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sparsifier.prune.revised_methods import (
    FisherScaleSparseGPT,
    SparseLowRankSparseGPT,
    presence_projector_from_offsets,
    projected_cg_refit,
    sparse_plus_low_rank,
    upstream_sample_weight,
    write_alignment,
)


def test_presence_offset_inside_the_tangent_has_no_projector():
    tangent = torch.eye(4, 1)
    offset = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    projector, rank = presence_projector_from_offsets(offset, tangent)
    assert rank == 0
    torch.testing.assert_close(projector, torch.zeros(4, 4))


def test_presence_offset_normal_to_the_tangent_is_kept():
    tangent = torch.eye(3, 1)
    offset = torch.tensor([[0.0, 4.0, 0.0]])
    projector, rank = presence_projector_from_offsets(offset, tangent)
    assert rank == 1
    torch.testing.assert_close(projector[1, 1], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(projector[0, :], torch.zeros(3), atol=1e-5, rtol=1e-5)


def test_write_alignment_is_one_when_the_write_lies_in_the_projector():
    projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    aligned = write_alignment(torch.tensor([3.0, 0.0, 0.0]), projector)
    missed = write_alignment(torch.tensor([0.0, 3.0, 0.0]), projector)
    assert aligned == pytest.approx(1.0)
    assert missed == pytest.approx(0.0)


def test_upstream_weight_scales_silu_and_geometric_energy():
    projector = torch.diag(torch.tensor([1.0, 0.0]))
    write = torch.tensor([3.0, 4.0])
    silu = torch.tensor([2.0, 0.5])
    gamma = 1.0
    got = upstream_sample_weight(silu, write, projector, gamma)
    # ||w||^2 = 25, ||P w||^2 = 9, factor = 34, times silu^2
    torch.testing.assert_close(got, torch.tensor([4.0, 0.25]) * 34.0)


def test_sparse_plus_low_rank_kills_concept_residual_and_does_not_grow_it():
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sparse = torch.tensor([[0.0, 2.0], [0.0, 0.0], [5.0, 0.0]])
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    hat = sparse_plus_low_rank(dense, sparse, basis)
    residual = dense - hat
    concept_component = basis @ (basis.T @ residual)
    torch.testing.assert_close(concept_component, torch.zeros_like(residual), atol=1e-5, rtol=1e-5)
    assert float(torch.linalg.norm(dense - hat)) <= float(torch.linalg.norm(dense - sparse)) + 1e-5


def test_projected_cg_refit_beats_the_uncompensated_mask():
    torch.manual_seed(0)
    weight = torch.randn(3, 4)
    hessian = torch.randn(4, 4)
    hessian = hessian @ hessian.T + 0.1 * torch.eye(4)
    keep = torch.zeros_like(weight, dtype=torch.bool)
    keep[:, :2] = True
    refit = projected_cg_refit(weight, keep, hessian)
    assert torch.equal(refit[:, 2:] == 0, torch.ones_like(refit[:, 2:], dtype=torch.bool))
    uncompensated = weight.clone()
    uncompensated[:, 2:] = 0
    def loss(hat):
        residual = weight - hat
        return float(torch.trace(residual @ hessian @ residual.T))
    assert loss(refit) < loss(uncompensated)


def test_fisher_pruner_changes_a_mask_when_gradients_concentrate():
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    phi = torch.randn(16, 4)
    grad = torch.zeros(16, 2)
    grad[:, 0] = 5.0
    solver = FisherScaleSparseGPT(layer, alpha=5.0)
    solver.add_batch(phi.unsqueeze(0), layer(phi))
    solver.add_fisher(phi, grad)
    solver.fasterprune(0.5, blocksize=4)
    assert int((layer.weight[0] == 0).sum()) < int((layer.weight[1] == 0).sum())


def test_sparse_low_rank_pruner_restores_the_concept_rows():
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=False)
    basis = torch.zeros(3, 1)
    basis[0, 0] = 1.0
    solver = SparseLowRankSparseGPT(layer, union_basis=basis, rank=1, calibration_samples=8)
    x = torch.randn(2, 12, 4)
    solver.add_batch(x, layer(x))
    dense = layer.weight.data.clone()
    solver.fasterprune(0.5, blocksize=4)
    hat = layer.weight.data.float()
    residual = dense - hat
    assert float(residual[0].abs().max()) < 1e-4
    assert int((hat[1:] == 0).sum()) > 0
