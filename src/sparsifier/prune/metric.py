"""Output-space metrics ``M_n`` for projected reconstruction (note §3)."""

from __future__ import annotations

import torch


def projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
    """``P = U U^T`` for an orthonormal basis ``U`` with shape ``[d, r]``."""
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
        raise ValueError("basis must have shape [d, r] with r >= 1")
    u = basis.float()
    return u @ u.T


def anchor_projector(anchor: torch.Tensor) -> torch.Tensor:
    """Rank-1 projector onto the span of an absolute chart-anchor vector.

    Origin-dependent by construction (note §3); report separately from
    ``P_union``.
    """
    v = anchor.float().reshape(-1)
    norm = torch.linalg.norm(v)
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        return torch.zeros((v.numel(), v.numel()), dtype=torch.float32)
    u = v / norm
    return torch.outer(u, u)


def union_plus_anchor_metric(
    union_projector: torch.Tensor,
    anchor: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """``M = P_union + η P_anchor`` with ``η >= 0``."""
    if not torch.isfinite(torch.as_tensor(eta)) or float(eta) < 0.0:
        raise ValueError("anchor eta must be finite and nonnegative")
    p_union = union_projector.float()
    if float(eta) == 0.0:
        return p_union
    return p_union + float(eta) * anchor_projector(anchor)


def diagonal_metric_from_basis(basis: torch.Tensor) -> torch.Tensor:
    """Diagonal of ``U U^T`` (per-coordinate occupancy)."""
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    return basis.float().square().sum(dim=1).clamp_(0.0, 1.0)


def blend_anchor_into_diagonals(
    projector_diagonals: torch.Tensor,
    anchors: torch.Tensor,
    assignments: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Add ``η (P_anchor)_{pp}`` onto per-sample projector diagonals.

    ``projector_diagonals``: ``[N, d]``
    ``anchors``: ``[C, d]`` chart centers (absolute offsets)
    ``assignments``: ``[N]`` chart ids
    """
    if not torch.isfinite(torch.as_tensor(eta)) or float(eta) < 0.0:
        raise ValueError("anchor eta must be finite and nonnegative")
    diags = projector_diagonals.float()
    if float(eta) == 0.0:
        return diags
    if anchors.ndim != 2 or assignments.ndim != 1:
        raise ValueError("anchors must be [C, d] and assignments [N]")
    if diags.shape[0] != assignments.shape[0] or diags.shape[1] != anchors.shape[1]:
        raise ValueError("diagonal/anchor shapes must match samples and ambient dim")
    norms = torch.linalg.norm(anchors.float(), dim=1).clamp_min(1e-12)
    # (P_anchor)_{pp} = (c_p / ||c||)^2 for the assigned chart center.
    anchor_diag = (anchors.float() / norms[:, None]).square()
    return diags + float(eta) * anchor_diag[assignments]


def coupling_from_diagonals(
    projector_diagonals: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """``G_pq`` from per-sample diagonal metrics and input energies."""
    x2 = inputs.float().square()
    numerator = projector_diagonals.float().T @ x2
    denom = x2.sum(dim=0).clamp_min(1e-12)
    return (numerator / denom[None, :]).clamp_(0.0, 1.0)
