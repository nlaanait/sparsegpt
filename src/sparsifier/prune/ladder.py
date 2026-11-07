"""Revised note §4.8: one rescaling implementation, a ladder of row weights."""

from __future__ import annotations

import math

import torch

from sparsifier.prune.rescale import row_scales


def fisher_sample_weights(gradients: torch.Tensor, alpha: float) -> torch.Tensor:
    """ω_{np} = 1 + α g_{np}^2 / mean(g^2)."""
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    g2 = gradients.float().square()
    if g2.ndim != 2:
        raise ValueError("gradients must have shape [samples, output_dim]")
    return 1.0 + float(alpha) * g2 / g2.mean().clamp_min(1e-12)


def row_scales_from_sample_weights(
    omega: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """λ_{pq}^2 = Σ_n ω_{np} φ_{nq}^2 / Σ_n φ_{nq}^2."""
    w = omega.float()
    x = phi.float()
    if w.ndim != 2 or x.ndim != 2 or w.shape[0] != x.shape[0]:
        raise ValueError("omega and phi must share a sample axis")
    x2 = x.square()
    numerator = w.T @ x2
    denom = x2.sum(dim=0).clamp_min(1e-12)
    return torch.sqrt((numerator / denom[None, :]).clamp_min(0.0))


def static_union_row_scales(leverage: torch.Tensor, gamma: float, input_dim: int) -> torch.Tensor:
    """Static concept union: ω_p = 1 + γ P_{pp}, a row-only factor."""
    g = leverage.float().reshape(-1, 1).expand(-1, input_dim)
    return row_scales(g, gamma)
