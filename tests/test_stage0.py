"""Revised note §6 Stage 0 diagnostics: D1, D3, D6 helpers.

Each diagnostic has a null. These tests lock the formulas; they do not
claim a pruning result.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparsifier.prune.stage0 import (
    beta_leverage_null,
    congruence_relative_error,
    gradient_enrichment,
    leverage_concentration,
    mask_flip_rate,
    mean_principal_angle,
    radial_share,
)


def test_beta_leverage_null_mean_is_rank_over_ambient():
    mean, variance = beta_leverage_null(rank=2, ambient_dim=10)
    assert mean == pytest.approx(0.2)
    # Beta(α,β) variance αβ / ((α+β)^2 (α+β+1)) with α=1, β=4.
    assert variance == pytest.approx((1.0 * 4.0) / ((5.0**2) * 6.0))


def test_leverage_concentration_reports_mean_and_ks_against_beta_null():
    torch.manual_seed(0)
    rank, d = 2, 16
    # Random orthonormal rows in general position: leverages ~ Beta(r/2,(d-r)/2).
    q, _ = torch.linalg.qr(torch.randn(d, rank), mode="reduced")
    leverages = q.square().sum(dim=1)
    summary = leverage_concentration(leverages, rank=rank)
    assert summary["mean"] == pytest.approx(rank / d, abs=1e-6)
    assert summary["null_mean"] == pytest.approx(rank / d)
    assert 0.0 <= summary["ks_statistic"] <= 1.0


def test_gradient_enrichment_is_one_for_isotropic_gradients():
    torch.manual_seed(0)
    d, r = 20, 3
    q, _ = torch.linalg.qr(torch.randn(d, r), mode="reduced")
    projector = q @ q.T
    gradients = torch.randn(50, d)
    kappa = gradient_enrichment(gradients, projector.expand(50, -1, -1), mean_geo_rank=r)
    assert kappa == pytest.approx(1.0, abs=0.35)


def test_identical_bases_have_zero_principal_angle():
    q, _ = torch.linalg.qr(torch.randn(8, 3), mode="reduced")
    assert mean_principal_angle(q, q) == pytest.approx(0.0, abs=1e-3)


def test_orthogonal_planes_have_right_angle():
    q = torch.eye(4, 2)
    p = torch.eye(4, 4)[:, 2:]
    assert mean_principal_angle(q, p) == pytest.approx(math.pi / 2, rel=1e-5)


def test_mask_flip_rate_is_the_disagreement_fraction():
    a = torch.tensor([[True, False], [False, True]])
    b = torch.tensor([[True, True], [False, True]])
    assert mask_flip_rate(a, b) == pytest.approx(0.25)


def test_congruence_error_is_zero_when_the_trick_is_exact():
    shared = torch.tensor([[2.0, 0.5], [0.5, 4.0]])
    scale = torch.tensor([1.0, 2.0])
    row = scale[:, None] * shared * scale[None, :]
    assert congruence_relative_error(row, shared, scale) == pytest.approx(0.0)


def test_radial_share_is_one_minus_tangent_energy_fraction():
    delta = torch.tensor([[0.0, 4.0, 3.0]])
    tangent = torch.tensor([[0.0], [1.0], [0.0]])
    share = radial_share(delta, tangent)
    assert share == pytest.approx(9.0 / 25.0)
