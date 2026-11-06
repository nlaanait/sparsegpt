"""§3 projected least-squares bias correction for a fixed residual Aφ."""

from __future__ import annotations

import torch


def projected_bias_correction(
    residual: torch.Tensor,
    metrics: torch.Tensor,
) -> torch.Tensor:
    """Solve (sum M_n) Δb = sum M_n r_n with a minimum-norm pseudoinverse.

    Parameters
    ----------
    residual:
        ``[N, d]`` activation residuals ``A φ_n`` (before bias correction).
    metrics:
        ``[N, d, d]`` PSD output-space metrics ``M_n`` (e.g. ``P_union,n``).
    """
    if residual.ndim != 2:
        raise ValueError("residual must have shape [samples, output_dim]")
    if metrics.ndim != 3 or metrics.shape[0] != residual.shape[0]:
        raise ValueError("metrics must have shape [samples, output_dim, output_dim]")
    if metrics.shape[1] != residual.shape[1] or metrics.shape[2] != residual.shape[1]:
        raise ValueError("metric ambient dimension must match residual")
    if not torch.isfinite(residual).all() or not torch.isfinite(metrics).all():
        raise ValueError("residual and metrics must be finite")

    out_dtype = residual.dtype
    r = residual.to(dtype=torch.float64)
    m = metrics.to(dtype=torch.float64)
    lhs = m.sum(dim=0)
    rhs = torch.einsum("nij,nj->i", m, r)
    # Minimum-norm solution via pseudoinverse; null-space components stay 0.
    return (torch.linalg.pinv(lhs, rcond=1e-6) @ rhs).to(dtype=out_dtype)
