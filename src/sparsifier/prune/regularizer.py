"""Revised note §3.2: ambient reconstruction plus a geometric regularizer.

The geometric term is never a replacement for the ambient loss. Use
``M_n = I + γ P^{geo}_n`` with ``γ = α d / r̄^{geo}``.
"""

from __future__ import annotations

import math

import torch


def geometric_weight(alpha: float, ambient_dim: int, mean_geo_rank: float) -> float:
    """γ = α d / r̄^{geo}. α = 0 is ambient-only even if the mean rank is 0."""
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if ambient_dim < 1:
        raise ValueError("ambient_dim must be positive")
    if alpha == 0.0:
        return 0.0
    if not math.isfinite(mean_geo_rank) or mean_geo_rank <= 0.0:
        raise ValueError("mean_geo_rank must be positive when alpha > 0")
    return float(alpha) * float(ambient_dim) / float(mean_geo_rank)


def mean_geometric_rank(ranks: torch.Tensor) -> float:
    """Mean of per-sample geometric ranks, inactive (zero) samples included."""
    values = torch.as_tensor(ranks, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError("mean_geometric_rank requires at least one sample")
    return float(values.mean())


def _orthonormal_basis(basis: torch.Tensor) -> torch.Tensor:
    u = basis.float()
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 1:
        raise ValueError("basis must have shape [d, r] with r >= 1")
    q, _ = torch.linalg.qr(u, mode="reduced")
    singular = torch.linalg.svdvals(u)
    rank = int((singular > max(float(singular[0]) * 1e-7, 1e-12)).sum().item())
    if rank == 0:
        raise ValueError("tangent basis has zero rank")
    return q[:, :rank].contiguous()


def presence_direction(anchor: torch.Tensor, tangent_basis: torch.Tensor) -> torch.Tensor:
    """Unit presence direction: chart anchor orthogonalized against the tangents.

    The anchor is ``m = h̄(s_0) - h_0`` (explicit concept-absent reference).
    If it already lies in the tangent, the presence direction is zero.
    """
    u = _orthonormal_basis(tangent_basis)
    v = anchor.float().reshape(-1)
    if v.numel() != u.shape[0]:
        raise ValueError("anchor length must match the tangent ambient dimension")
    residual = v - u @ (u.T @ v)
    norm = torch.linalg.norm(residual)
    if not torch.isfinite(norm) or float(norm) <= 1e-12:
        return torch.zeros_like(v)
    return residual / norm


def geometric_projector(
    tangent_basis: torch.Tensor,
    presence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Projector onto the span of active tangents plus presence directions."""
    u = _orthonormal_basis(tangent_basis)
    parts = [u]
    if presence is not None:
        p = presence.float().reshape(-1, 1)
        if p.shape[0] != u.shape[0]:
            raise ValueError("presence length must match the tangent ambient dimension")
        if float(torch.linalg.norm(p)) > 1e-12:
            parts.append(p)
    stacked = torch.cat(parts, dim=1)
    q, _ = torch.linalg.qr(stacked, mode="reduced")
    singular = torch.linalg.svdvals(stacked)
    rank = int((singular > max(float(singular[0]) * 1e-7, 1e-12)).sum().item())
    q = q[:, :rank]
    return q @ q.T, rank


def regularized_metric(projector: torch.Tensor, gamma: float) -> torch.Tensor:
    """M = I + γ P^{geo}."""
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")
    p = projector.float()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("projector must be a square matrix")
    eye = torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
    return eye + float(gamma) * p


def regularized_bias_correction(
    residual: torch.Tensor,
    projectors: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Solve (N I + γ Σ P_n) Δb = Σ (I + γ P_n) r_n.

    The left-hand side is positive definite for γ ≥ 0, so this is a
    linear solve rather than a pseudoinverse.
    """
    if residual.ndim != 2:
        raise ValueError("residual must have shape [samples, output_dim]")
    if projectors.ndim != 3 or projectors.shape[0] != residual.shape[0]:
        raise ValueError("projectors must have shape [samples, output_dim, output_dim]")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")
    r = residual.float()
    p = projectors.float()
    n, d = r.shape
    eye = torch.eye(d, dtype=r.dtype, device=r.device)
    metrics = eye + float(gamma) * p
    lhs = n * eye + float(gamma) * p.sum(dim=0)
    rhs = torch.einsum("nij,nj->i", metrics, r)
    return torch.linalg.solve(lhs, rhs)
