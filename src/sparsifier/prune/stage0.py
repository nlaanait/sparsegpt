"""Revised note §6 Stage 0 diagnostics (D1, D3, D6)."""

from __future__ import annotations

import math

import torch
from scipy.stats import beta as beta_dist


def beta_leverage_null(rank: int, ambient_dim: int) -> tuple[float, float]:
    """Mean and variance of Beta(r/2, (d-r)/2), the coordinate-leverage null."""
    if rank < 1 or ambient_dim <= rank:
        raise ValueError("require 1 <= rank < ambient_dim")
    alpha = rank / 2.0
    beta = (ambient_dim - rank) / 2.0
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total**2) * (total + 1.0))
    return mean, variance


def leverage_concentration(leverages: torch.Tensor, rank: int) -> dict[str, float]:
    """D1: compare coordinate leverages to the Beta null."""
    ell = leverages.float().reshape(-1)
    if ell.numel() < 2:
        raise ValueError("need at least two leverages")
    ambient_dim = int(ell.numel())
    null_mean, null_var = beta_leverage_null(rank, ambient_dim)
    mean = float(ell.mean())
    variance = float(ell.var(unbiased=False))
    alpha = rank / 2.0
    beta = (ambient_dim - rank) / 2.0
    x = ell.clamp(0.0, 1.0).sort().values
    cdf = torch.as_tensor(
        beta_dist.cdf(x.detach().cpu().numpy(), a=alpha, b=beta),
        dtype=x.dtype,
    )
    n = x.numel()
    ecdf = torch.arange(1, n + 1, dtype=x.dtype) / n
    ks = torch.maximum((ecdf - cdf).abs(), (ecdf - 1.0 / n - cdf).abs()).max()
    return {
        "mean": mean,
        "variance": variance,
        "null_mean": null_mean,
        "null_variance": null_var,
        "ks_statistic": float(ks),
        "rank": float(rank),
        "ambient_dim": float(ambient_dim),
    }


def gradient_enrichment(
    gradients: torch.Tensor,
    projectors: torch.Tensor,
) -> float:
    """D3: gradient energy enrichment over a rank-matched random-subspace null."""
    g = gradients.float()
    if g.ndim != 2:
        raise ValueError("gradients must have shape [samples, dim]")
    p = projectors.float()
    if p.ndim == 2:
        p = p.expand(g.shape[0], -1, -1)
    if p.ndim != 3 or p.shape[0] != g.shape[0] or p.shape[1] != g.shape[1]:
        raise ValueError("projectors must match the gradient batch and dimension")
    projected = torch.einsum("nij,nj->ni", p, g)
    grad_energy = g.square().sum(dim=1)
    den = float(grad_energy.sum())
    if den <= 0.0:
        return 0.0
    projected_energy = projected.square().sum(dim=1)
    # For varying projector ranks, the null is the ratio of summed energies,
    # weighted by each sample's gradient norm, rather than mean-rank / d.
    ranks = torch.diagonal(p, dim1=1, dim2=2).sum(dim=1).clamp_min(0.0)
    null_fraction = float((ranks * grad_energy).sum()) / (g.shape[1] * den)
    if null_fraction <= 0.0:
        return 0.0
    return float(projected_energy.sum() / den) / null_fraction


def mean_principal_angle(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """D4: mean principal angle (radians) between two orthonormal frames."""
    qa = basis_a.float()
    qb = basis_b.float()
    if qa.ndim == 1:
        qa = qa.reshape(-1, 1)
    if qb.ndim == 1:
        qb = qb.reshape(-1, 1)
    qa, _ = torch.linalg.qr(qa, mode="reduced")
    qb, _ = torch.linalg.qr(qb, mode="reduced")
    singular = torch.linalg.svdvals(qa.T @ qb).clamp(-1.0, 1.0)
    return float(torch.arccos(singular).mean())


def mask_flip_rate(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """D2: fraction of entries whose keep/drop bit differs."""
    if mask_a.shape != mask_b.shape:
        raise ValueError("masks must have the same shape")
    a = mask_a.to(dtype=torch.bool)
    b = mask_b.to(dtype=torch.bool)
    return float((a != b).to(dtype=torch.float32).mean())


def congruence_relative_error(
    row_hessian: torch.Tensor,
    shared_hessian: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    """D5: ||Λ H' Λ - H^{(p)}|| / ||H^{(p)}|| for one output row."""
    h_p = row_hessian.float()
    h_shared = shared_hessian.float()
    lam = scale.float().reshape(-1)
    if h_p.shape != h_shared.shape or lam.numel() != h_p.shape[0]:
        raise ValueError("row Hessian, shared Hessian, and scale must agree")
    approx = lam[:, None] * h_shared * lam[None, :]
    denom = float(torch.linalg.norm(h_p))
    if denom <= 0.0:
        return 0.0
    return float(torch.linalg.norm(approx - h_p)) / denom


def radial_share(deltas: torch.Tensor, tangent_basis: torch.Tensor) -> float:
    """D6: fraction of knob/presence variance normal to the local tangent."""
    d = deltas.float()
    if d.ndim == 1:
        d = d.reshape(1, -1)
    u = tangent_basis.float()
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    q, _ = torch.linalg.qr(u, mode="reduced")
    projected = d @ q @ q.T
    normal = d - projected
    total = float(d.square().sum())
    if total <= 0.0:
        return 0.0
    return float(normal.square().sum()) / total
