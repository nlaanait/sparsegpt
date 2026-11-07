"""§1 editability, knockout impact, union ratio, and participation ratio."""

from __future__ import annotations

import math

import torch


def _as_basis(u: torch.Tensor) -> torch.Tensor:
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 1:
        raise ValueError("basis must have shape [d, r] with r >= 1")
    return u


def editability(jac_col: torch.Tensor, basis: torch.Tensor) -> float:
    """Local editability e_i = r^{-1/2} || U^T (dh/dW_alpha) ||_F."""
    u = _as_basis(basis)
    jac = jac_col.reshape(-1)
    if jac.numel() != u.shape[0]:
        raise ValueError("jacobian column length must match basis ambient dim")
    rank = u.shape[1]
    return float(torch.linalg.norm(u.T @ jac) / math.sqrt(rank))


def knockout_impact(jac_col: torch.Tensor, weight: float, basis: torch.Tensor) -> float:
    """Per-factor first-order knockout τ_i = r^{-1/2} || U^T (J W) ||_F."""
    return editability(jac_col * float(weight), basis)


def ambient_knockout_energy(delta_h: torch.Tensor) -> float:
    """τ_ambient^2 for a single perturbation vector δh."""
    return float(torch.linalg.norm(delta_h.reshape(-1)) ** 2)


def union_basis(factor_bases: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Orthonormal basis for the span of stacked factor tangents."""
    if not factor_bases:
        raise ValueError("at least one factor basis is required")
    parts = [_as_basis(b) for b in factor_bases]
    ambient = parts[0].shape[0]
    if any(p.shape[0] != ambient for p in parts):
        raise ValueError("all factor bases must share the ambient dimension")
    stacked = torch.cat(parts, dim=1)
    q, _ = torch.linalg.qr(stacked, mode="reduced")
    # Drop numerically zero columns after QR.
    keep = min(q.shape[1], stacked.shape[1])
    # Rank from original stacked columns via singular values when needed.
    if stacked.numel() == 0:
        raise ValueError("empty factor bases")
    singular = torch.linalg.svdvals(stacked)
    rank = int((singular > max(float(singular[0]) * 1e-7, 1e-12)).sum().item())
    if rank == 0:
        raise ValueError("factor bases have zero union rank")
    return q[:, :rank].contiguous(), rank


def union_knockout_impact(
    delta_h: torch.Tensor, factor_bases: list[torch.Tensor]
) -> float:
    """τ_union^2 = || Q^T δh ||^2 for the actual span of active factor tangents."""
    q, _ = union_basis(factor_bases)
    delta = delta_h.reshape(-1).to(dtype=q.dtype)
    return float(torch.linalg.norm(q.T @ delta) ** 2)


def union_ambient_ratio(delta_h: torch.Tensor, union_q: torch.Tensor) -> float:
    """ρ_union = τ_union^2 / τ_ambient^2 ∈ [0, 1]."""
    ambient = ambient_knockout_energy(delta_h)
    if ambient <= 0.0:
        return 0.0
    q = _as_basis(union_q)
    tau_union = float(torch.linalg.norm(q.T @ delta_h.reshape(-1).to(dtype=q.dtype)) ** 2)
    return tau_union / ambient


def participation_ratio(tau_squares: torch.Tensor) -> float:
    """PR = (sum q_i)^2 / sum q_i^2 over the chosen concept set.

    The revised note computes this on per-dimension energies
    ``q_i = τ_i^{cond}^2 / r_i``. The original rank-normalized scores are
    already per-dimension, so both notes share this aggregator.
    """
    values = tau_squares.reshape(-1).to(dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("participation ratio requires finite scores")
    denom = float(torch.sum(values**2))
    if denom <= 0.0:
        raise ValueError("participation ratio requires a nonzero score vector")
    numer = float(torch.sum(values) ** 2)
    return numer / denom


def tangent_energy(delta_h: torch.Tensor, basis: torch.Tensor) -> float:
    """Raw per-factor tangent energy τ_i^2 = ||U^T δh||_2^2."""
    u = _as_basis(basis)
    delta = delta_h.reshape(-1).to(dtype=u.dtype)
    if delta.numel() != u.shape[0]:
        raise ValueError("delta length must match the basis ambient dimension")
    return float(torch.linalg.norm(u.T @ delta) ** 2)


def per_dimension_energy(delta_h: torch.Tensor, basis: torch.Tensor) -> float:
    """τ_i^2 / r_i, the original note's rank-normalized energy."""
    u = _as_basis(basis)
    return tangent_energy(delta_h, basis) / u.shape[1]


def enrichment(delta_h: torch.Tensor, basis: torch.Tensor) -> float:
    """(d / r) τ^2 / ||δh||^2. Equals 1 on average for a random r-plane."""
    u = _as_basis(basis)
    ambient = ambient_knockout_energy(delta_h)
    if ambient <= 0.0:
        return 0.0
    return (u.shape[0] / u.shape[1]) * tangent_energy(delta_h, basis) / ambient


def editability_enrichment(jac_col: torch.Tensor, basis: torch.Tensor) -> float:
    """ẽ^{edit} = (||U^T j|| / sqrt(r)) / (||j|| / sqrt(d))."""
    u = _as_basis(basis)
    jac = jac_col.reshape(-1)
    isotropic = float(torch.linalg.norm(jac) / math.sqrt(u.shape[0]))
    if isotropic <= 0.0:
        return 0.0
    return editability(jac_col, basis) / isotropic


def presence_energy(delta_h: torch.Tensor, projector: torch.Tensor) -> float:
    """τ_pres^2 = ||P^{pres} δh||_2^2."""
    p = projector.float()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("presence projector must be square")
    delta = delta_h.reshape(-1).to(dtype=p.dtype)
    if delta.numel() != p.shape[0]:
        raise ValueError("delta length must match the projector ambient dimension")
    return float(torch.linalg.norm(p @ delta) ** 2)


def unconditional_energy(conditional: float, presence_rate: float) -> float:
    """τ^{uncond}^2 = P(i ∈ S) τ^{cond}^2."""
    if not math.isfinite(conditional) or conditional < 0.0:
        raise ValueError("conditional energy must be finite and nonnegative")
    if not math.isfinite(presence_rate) or not 0.0 <= presence_rate <= 1.0:
        raise ValueError("presence_rate must be finite and in [0, 1]")
    return float(conditional) * float(presence_rate)
