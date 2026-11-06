"""Unit tests for §1 editability, knockout impact, union ratio, and PR.

These lock the note formulas before any scoring implementation lands.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparsifier.prune.scores import (
    editability,
    knockout_impact,
    participation_ratio,
    union_knockout_impact,
)


def test_editability_is_rms_of_projected_jacobian_column():
    # h = W phi, so dh/dW_pq = phi_q * e_p. For scalar alpha=(p,q):
    # e_i = r_i^{-1/2} || U_i^T (phi_q e_p) ||_F = r_i^{-1/2} |phi_q| ||U_i[p,:]||
    phi_q = 2.0
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)  # d=3,r=2
    # Jacobian contribution for entry (p=0,q): phi_q * e_0
    jac_col = phi_q * torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    got = editability(jac_col, u)
    expected = (1.0 / math.sqrt(2.0)) * torch.linalg.norm(u.T @ jac_col).item()
    assert got == pytest.approx(expected)


def test_knockout_impact_scales_editability_by_weight_magnitude():
    weight = -3.0
    jac_col = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    u = torch.eye(3, 1, dtype=torch.float64)
    tau = knockout_impact(jac_col, weight, u)
    # tau = r^{-1/2} || U^T (jac * W) || = |W| * editability
    assert tau == pytest.approx(abs(weight) * editability(jac_col, u))


def test_union_knockout_is_rms_under_orthonormal_span_basis():
    # Two overlapping factor bases; union Q from QR of stacked bases.
    torch.manual_seed(0)
    u0 = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    u1 = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64)
    u1 = u1 / torch.linalg.norm(u1)
    delta_h = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
    tau2 = union_knockout_impact(delta_h, [u0, u1])
    stacked = torch.cat([u0, u1], dim=1)
    q, _ = torch.linalg.qr(stacked, mode="reduced")
    expected = torch.linalg.norm(q.T @ delta_h).item() ** 2
    assert tau2 == pytest.approx(expected)


def test_participation_ratio_is_one_for_single_factor_and_m_when_equal():
    single = torch.tensor([4.0, 0.0, 0.0])
    assert participation_ratio(single) == pytest.approx(1.0)
    equal = torch.ones(5)
    assert participation_ratio(equal) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="nonzero"):
        participation_ratio(torch.zeros(3))


def test_union_ratio_lies_in_unit_interval():
    from sparsifier.prune.scores import ambient_knockout_energy, union_ambient_ratio

    delta = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    q = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    rho = union_ambient_ratio(delta, q)
    assert 0.0 <= rho <= 1.0
    assert rho == pytest.approx(
        union_knockout_impact(delta, [q]) / ambient_knockout_energy(delta)
    )
