"""Revised note §3.2: regularizer M_n = I + γ P_geo, presence, unbounded α.

Tests are the spec. Implementations must match these formulas before any
pruner wiring.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparsifier.prune.regularizer import (
    geometric_projector,
    geometric_weight,
    mean_geometric_rank,
    presence_direction,
    regularized_metric,
)


def test_geometric_weight_is_alpha_times_ambient_over_mean_geo_rank():
    gamma = geometric_weight(alpha=0.5, ambient_dim=4096, mean_geo_rank=64.0)
    assert gamma == pytest.approx(0.5 * 4096 / 64.0)


def test_geometric_weight_allows_alpha_above_one():
    gamma = geometric_weight(alpha=10.0, ambient_dim=8, mean_geo_rank=2.0)
    assert gamma == pytest.approx(40.0)


def test_geometric_weight_is_zero_when_alpha_is_zero():
    assert geometric_weight(alpha=0.0, ambient_dim=8, mean_geo_rank=0.0) == 0.0
    assert geometric_weight(alpha=0.0, ambient_dim=8, mean_geo_rank=2.0) == 0.0


def test_geometric_weight_rejects_nonfinite_or_negative_alpha():
    with pytest.raises(ValueError, match="alpha"):
        geometric_weight(alpha=-0.1, ambient_dim=8, mean_geo_rank=1.0)
    with pytest.raises(ValueError, match="alpha"):
        geometric_weight(alpha=float("nan"), ambient_dim=8, mean_geo_rank=1.0)


def test_mean_geometric_rank_includes_inactive_zeros():
    ranks = torch.tensor([2.0, 0.0, 1.0, 0.0])
    assert mean_geometric_rank(ranks) == pytest.approx(0.75)


def test_presence_direction_is_anchor_orthogonalized_against_tangents():
    tangent = torch.tensor([[1.0], [0.0], [0.0]])
    anchor = torch.tensor([3.0, 4.0, 0.0])
    presence = presence_direction(anchor, tangent)
    torch.testing.assert_close(presence, torch.tensor([0.0, 1.0, 0.0]))
    assert float(tangent.squeeze() @ presence) == pytest.approx(0.0)
    assert float(torch.linalg.norm(presence)) == pytest.approx(1.0)


def test_presence_vanishes_when_anchor_lies_in_the_tangent():
    tangent = torch.eye(3, 2)
    anchor = torch.tensor([2.0, -1.0, 0.0])
    presence = presence_direction(anchor, tangent)
    torch.testing.assert_close(presence, torch.zeros(3))


def test_geometric_projector_spans_tangents_and_presence():
    tangent = torch.tensor([[1.0], [0.0], [0.0]])
    presence = torch.tensor([0.0, 1.0, 0.0])
    projector, rank = geometric_projector(tangent, presence)
    assert rank == 2
    torch.testing.assert_close(projector @ projector, projector, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.trace(projector), torch.tensor(2.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(projector[:2, :2], torch.eye(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(projector[2, :], torch.zeros(3), atol=1e-5, rtol=1e-5)


def test_regularized_metric_is_identity_plus_gamma_geo_projector():
    projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    metric = regularized_metric(projector, gamma=3.0)
    expected = torch.eye(3) + 3.0 * projector
    torch.testing.assert_close(metric, expected)


def test_regularized_bias_lhs_is_positive_definite():
    """N I + γ Σ P_geo is PD, so the §3.2 bias solve needs no pseudoinverse."""
    n, d = 5, 4
    gamma = 2.0
    projector = torch.zeros(n, d, d)
    projector[:, 0, 0] = 1.0
    lhs = n * torch.eye(d) + gamma * projector.sum(dim=0)
    eigenvalues = torch.linalg.eigvalsh(lhs)
    assert float(eigenvalues.min()) > 0.0


def test_unweighted_mean_bias_is_not_optimal_when_metric_varies():
    """If M_n varies, Δb = A E[φ] is not the §3.2 least-squares bias."""
    from sparsifier.prune.regularizer import regularized_bias_correction

    residual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    projector = torch.zeros(2, 2, 2)
    projector[0, 0, 0] = 1.0
    projector[1, 1, 1] = 1.0
    gamma = 1.0
    delta = regularized_bias_correction(residual, projector, gamma=gamma)
    unweighted = residual.mean(dim=0)
    assert not torch.allclose(delta, unweighted)
    # Stationarity: (N I + γ Σ P) Δb = Σ (I + γ P) r
    lhs = residual.shape[0] * torch.eye(2) + gamma * projector.sum(dim=0)
    rhs = torch.einsum("nij,nj->i", torch.eye(2) + gamma * projector, residual)
    torch.testing.assert_close(lhs @ delta, rhs, atol=1e-5, rtol=1e-5)
