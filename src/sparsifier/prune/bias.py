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


def centered_second_moment(phi: torch.Tensor) -> torch.Tensor:
    """(1/N) Σ (φ_n - μ)(φ_n - μ)^T."""
    x = phi.float()
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("phi must have shape [samples, input_dim]")
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / x.shape[0]


def ones_augmented_second_moment(phi: torch.Tensor) -> torch.Tensor:
    """Second moment of [φ, 1], the ones-column Hessian of §3.4."""
    x = phi.float()
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("phi must have shape [samples, input_dim]")
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    augmented = torch.cat((x, ones), dim=1)
    return augmented.T @ augmented / x.shape[0]


def schur_complement_last(matrix: torch.Tensor) -> torch.Tensor:
    """Schur complement of the last diagonal entry: A - b b^T / c."""
    m = matrix.float()
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        raise ValueError("matrix must be square with size at least 2")
    a = m[:-1, :-1]
    b = m[:-1, -1]
    c = m[-1, -1]
    if not torch.isfinite(c) or float(c) == 0.0:
        raise ValueError("ones-column variance must be nonzero")
    return a - torch.outer(b, b) / c
