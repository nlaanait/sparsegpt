"""Unit tests for §3 projected least-squares bias correction."""

from __future__ import annotations

import pytest
import torch

from sparsifier.prune.bias import projected_bias_correction


def test_identity_metric_reduces_to_mean_residual():
    # A phi residuals; M_n = I => Delta b = mean(A phi)
    torch.manual_seed(0)
    residual = torch.randn(20, 5)  # rows = samples, cols = output dim
    metrics = torch.eye(5).expand(20, 5, 5).clone()
    delta = projected_bias_correction(residual, metrics)
    torch.testing.assert_close(delta, residual.mean(dim=0))


def test_varying_metric_solves_weighted_normal_equations():
    # (sum M_n) Delta b = sum M_n r_n
    residual = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64
    )
    m0 = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    m1 = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    m2 = torch.eye(2, dtype=torch.float64)
    metrics = torch.stack([m0, m1, m2])
    delta = projected_bias_correction(residual, metrics)
    lhs = m0 + m1 + m2
    rhs = m0 @ residual[0] + m1 @ residual[1] + m2 @ residual[2]
    torch.testing.assert_close(lhs @ delta, rhs)


def test_rank_deficient_metric_uses_minimum_norm_solution():
    # Only first coordinate is observed.
    residual = torch.tensor([[3.0, 9.0], [1.0, 7.0]], dtype=torch.float64)
    metrics = torch.zeros(2, 2, 2, dtype=torch.float64)
    metrics[:, 0, 0] = 1.0
    delta = projected_bias_correction(residual, metrics)
    assert delta[0] == pytest.approx(2.0)  # mean of first coords
    assert delta[1] == pytest.approx(0.0)  # null-space component zero (min-norm)
