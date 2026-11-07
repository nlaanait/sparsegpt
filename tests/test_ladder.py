"""Revised note §4.8 output-metric ladder helpers."""

from __future__ import annotations

import pytest
import torch

from sparsifier.prune.ladder import (
    fisher_sample_weights,
    row_scales_from_sample_weights,
    static_union_row_scales,
)
from sparsifier.prune.rescale import row_scales


def test_fisher_weights_are_one_plus_normalized_squared_gradients():
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    alpha = 2.0
    omega = fisher_sample_weights(gradients, alpha)
    mean = gradients.square().mean()
    torch.testing.assert_close(omega, 1.0 + alpha * gradients.square() / mean)


def test_row_scales_from_sample_weights_match_weighted_g():
    omega = torch.tensor([[1.0, 3.0], [1.0, 1.0]])
    phi = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    scale = row_scales_from_sample_weights(omega, phi)
    # den = [1, 4]; num[p,q] = Σ_n ω_np φ_nq^2
    expected = torch.sqrt(torch.tensor([[1.0, 1.0], [3.0, 1.0]]))
    torch.testing.assert_close(scale, expected)


def test_static_union_row_scales_match_one_plus_gamma_leverage():
    leverage = torch.tensor([1.0, 0.0, 0.25])
    gamma = 4.0
    scale = static_union_row_scales(leverage, gamma, input_dim=2)
    expected = row_scales(leverage[:, None].expand(3, 2), gamma)
    torch.testing.assert_close(scale, expected)


def test_fisher_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha"):
        fisher_sample_weights(torch.randn(3, 2), alpha=-0.1)
