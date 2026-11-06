"""Bias fold must use union projectors when the solver retained them."""

from __future__ import annotations

import torch
from torch import nn

from sparsifier.prune.bias import projected_bias_correction
from sparsifier.prune.factor_union import FactorUnionSparseGPT
from sparsifier.prune.tangent import TangentAwareSparseGPT
from sparsifier.prune.tangent_common import apply_bias_fold_from_samples
from sparsifier.prune.fixed_union import FixedUnionSparseGPT


def test_apply_bias_fold_with_projector_diagonals_is_not_identity_mean():
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=True)
    dense = torch.ones_like(layer.weight.data)
    with torch.no_grad():
        layer.weight.zero_()  # residual spans all output coords
        layer.bias.zero_()
    inputs = torch.randn(20, 4)
    # Only first output coordinate is in the union.
    pdiag = torch.zeros(20, 3)
    pdiag[:, 0] = 1.0
    delta = apply_bias_fold_from_samples(
        layer,
        dense_weight=dense,
        sample_inputs=inputs,
        sample_outputs=None,
        projector_diagonals=pdiag,
    )
    residual = ((dense - layer.weight.data.cpu().float()) @ inputs.T).T
    expected = projected_bias_correction(residual, torch.diag_embed(pdiag))
    torch.testing.assert_close(delta.cpu(), expected)
    # Identity metric would move all coordinates; union metric only the first.
    assert delta[0].abs().item() > 0
    assert delta[1].item() == 0.0
    assert delta[2].item() == 0.0
    assert not torch.allclose(delta.cpu(), residual.mean(dim=0))


def test_apply_bias_fold_with_union_basis_uses_full_projector():
    layer = nn.Linear(3, 3, bias=True)
    dense = torch.eye(3)
    with torch.no_grad():
        layer.weight.copy_(dense)
        layer.weight.mul_(0.5)
        layer.bias.zero_()
    inputs = torch.ones(10, 3)
    u = torch.tensor([[1.0], [0.0], [0.0]])
    delta = apply_bias_fold_from_samples(
        layer,
        dense_weight=dense,
        sample_inputs=inputs,
        sample_outputs=None,
        union_basis=u,
    )
    # Residual is 0.5 * ones on every output; M = e0 e0^T projects to first coord.
    assert delta[0].item() != 0.0
    assert delta[1].item() == 0.0
    assert delta[2].item() == 0.0


def test_tangent_bias_fold_retains_and_applies_sample_projectors():
    torch.manual_seed(0)
    layer = nn.Linear(5, 3, bias=True)
    solver = TangentAwareSparseGPT(
        layer, alpha=1.0, charts=2, rank=1, calibration_samples=32, seed=0, bias_fold=True
    )
    x = torch.randn(4, 8, 5)
    y = torch.randn(4, 8, 3)
    y[..., 1] += torch.linspace(-1, 1, 32).reshape(4, 8)
    solver.add_batch(x, y)
    solver.finalize_tangents()
    assert solver.sample_projector_diagonals is not None
    assert solver.sample_projector_diagonals.shape[0] == 32
    assert solver.sample_projector_diagonals.shape[1] == 3
    bias_before = layer.bias.data.clone()
    solver.fasterprune(0.25, blocksize=2)
    assert not torch.equal(layer.bias.data, bias_before)


def test_factor_union_bias_fold_stores_union_diagonals():
    torch.manual_seed(0)
    layer = nn.Linear(5, 3, bias=True)
    solver = FactorUnionSparseGPT(
        layer,
        alpha=1.0,
        factors=2,
        rank=1,
        calibration_samples=32,
        seed=0,
        bias_fold=True,
        factor_source="kmeans",
        active_factors=2,
    )
    x = torch.randn(4, 8, 5)
    y = torch.randn(4, 8, 3)
    solver.add_batch(x, y)
    solver.finalize_tangents()
    assert solver.sample_projector_diagonals is not None
    solver.fasterprune(0.2, blocksize=2)


def test_fixed_union_bias_fold_uses_union_basis():
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=True)
    u = torch.zeros(3, 1)
    u[0, 0] = 1.0
    solver = FixedUnionSparseGPT(
        layer, rank=1, calibration_samples=24, seed=0, bias_fold=True, union_basis=u
    )
    x = torch.randn(3, 8, 4)
    solver.add_batch(x, layer(x))
    solver.finalize_union()
    bias_before = layer.bias.data.clone()
    solver.fasterprune(0.5, blocksize=2)
    assert not torch.equal(layer.bias.data, bias_before)
