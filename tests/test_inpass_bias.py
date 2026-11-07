"""Revised note §3.4: bias handling inside the pruning pass.

Centered covariance and a trailing ones column are equivalent for M_n = I.
The ones column is excluded from mask selection and processed last.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sparsifier.prune.bias import (
    centered_second_moment,
    ones_augmented_second_moment,
    schur_complement_last,
)
from sparsifier.prune.inpass_bias import InPassBiasSparseGPT
from sparsifier.prune.sparsegpt import SparseGPT


def test_ones_column_schur_complement_is_centered_covariance():
    torch.manual_seed(0)
    phi = torch.randn(20, 5)
    centered = centered_second_moment(phi)
    augmented = ones_augmented_second_moment(phi)
    torch.testing.assert_close(schur_complement_last(augmented), centered, atol=1e-5, rtol=1e-5)


def test_centered_hessian_matches_mean_subtracted_inputs():
    torch.manual_seed(1)
    phi = torch.randn(12, 4)
    mu = phi.mean(dim=0, keepdim=True)
    expected = (phi - mu).T @ (phi - mu) / phi.shape[0]
    torch.testing.assert_close(centered_second_moment(phi), expected)


def test_inpass_bias_keeps_bias_column_dense_and_last():
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=True)
    x = torch.randn(2, 16, 4)
    solver = InPassBiasSparseGPT(layer, mode="ones_column")
    solver.add_batch(x, layer(x))
    solver.fasterprune(0.5, blocksize=5)
    assert layer.bias is not None
    # Ones-column compensation must be able to update the bias; it is not masked.
    assert torch.isfinite(layer.bias).all()
    # Weight sparsity is taken over the real columns only.
    nnz_frac = float((layer.weight != 0).sum()) / layer.weight.numel()
    assert nnz_frac == pytest.approx(0.5, abs=0.15)


def test_centered_mode_changes_mask_versus_uncentered_sparsegpt():
    torch.manual_seed(3)
    # Column 0 is almost constant: uncentered H protects it, centered H does not.
    n, d_in = 32, 4
    x = 0.05 * torch.randn(1, n, d_in)
    x[..., 0] = 4.0
    dense = nn.Linear(d_in, 2, bias=False)
    with torch.no_grad():
        dense.weight.fill_(1.0)
    ambient = nn.Linear(d_in, 2, bias=False)
    centered = nn.Linear(d_in, 2, bias=False)
    ambient.load_state_dict(dense.state_dict())
    centered.load_state_dict(dense.state_dict())

    a = SparseGPT(ambient)
    a.add_batch(x, ambient(x))
    a.fasterprune(0.5, blocksize=4)

    c = InPassBiasSparseGPT(centered, mode="centered")
    c.add_batch(x, centered(x))
    c.fasterprune(0.5, blocksize=4)

    assert int((ambient.weight[:, 0] != 0).sum()) == 2
    assert int((centered.weight[:, 0] == 0).sum()) == 2
