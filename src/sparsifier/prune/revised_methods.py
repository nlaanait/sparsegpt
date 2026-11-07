"""Revised-note pieces that sit on top of the Variant A/B solvers.

Presence directions from an explicit absent reference, the Fisher rung,
projected CG refit, sparse-plus-low-rank correction, and neuron write alignment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.ladder import fisher_sample_weights, row_scales_from_sample_weights
from sparsifier.prune.regularizer import geometric_projector, presence_direction
from sparsifier.prune.rescale import prune_with_rescaling
from sparsifier.prune.fixed_union import FixedUnionSparseGPT
from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix


def presence_projector_from_offsets(
    offsets: torch.Tensor,
    tangent_basis: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Projector onto presence directions orthogonal to ``tangent_basis``.

    Each row of ``offsets`` is a raw concept-minus-absent vector, ``h - h_0``.
    Rows that already lie in the tangent contribute nothing.
    """
    rows = offsets.float()
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("offsets must have shape [directions, dim]")
    directions = []
    for row in rows:
        direction = presence_direction(row, tangent_basis)
        if float(torch.linalg.norm(direction)) > 0.0:
            directions.append(direction)
    dim = rows.shape[1]
    if not directions:
        return torch.zeros(dim, dim), 0
    basis = torch.stack(directions, dim=1)
    return geometric_projector(basis)


def write_alignment(write_vector: torch.Tensor, projector: torch.Tensor) -> float:
    """χ = ||P w||^2 / ||w||^2 for one neuron write."""
    w = write_vector.float().reshape(-1)
    energy = float(w.square().sum())
    if energy <= 0.0:
        return 0.0
    projected = projector.float() @ w
    return float(projected.square().sum()) / energy


def upstream_sample_weight(
    silu_z: torch.Tensor,
    write_vector: torch.Tensor,
    projector: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """ω = silu(z)^2 (||w||^2 + γ ||P w||^2), one weight per sample."""
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")
    w = write_vector.float().reshape(-1)
    energy = float(w.square().sum())
    geometric = float((projector.float() @ w).square().sum())
    return silu_z.float().reshape(-1).square() * (energy + float(gamma) * geometric)


def sparse_plus_low_rank(
    dense: torch.Tensor,
    sparse: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Ŵ = W_sp + Π (W - W_sp), with Π the orthonormal span of ``basis``."""
    if dense.shape != sparse.shape:
        raise ValueError("dense and sparse weights must share a shape")
    q = basis.float()
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    q, _ = torch.linalg.qr(q, mode="reduced")
    if q.shape[0] != dense.shape[0]:
        raise ValueError("basis ambient dimension must match the output rows")
    delta = dense.float() - sparse.float()
    return sparse.float() + q @ (q.T @ delta)


def projected_cg_refit(
    weight: torch.Tensor,
    keep_mask: torch.Tensor,
    hessian: torch.Tensor,
    metric: torch.Tensor | None = None,
    iterations: int | None = None,
) -> torch.Tensor:
    """Exact refit of the surviving entries for Σ π M A H.

    ``keep_mask`` is true on entries that may change. Pruned entries stay zero.
    ``metric`` is the output-space matrix; identity is the ambient refit.
    """
    if weight.shape != keep_mask.shape:
        raise ValueError("keep_mask must match the weight shape")
    if hessian.ndim != 2 or hessian.shape[0] != weight.shape[1]:
        raise ValueError("hessian must be square on the input dimension")
    work = weight.double()
    keep = keep_mask.to(dtype=torch.bool)
    hessian_w = 0.5 * (hessian.double() + hessian.double().T)
    if metric is None:
        metric_w = torch.eye(work.shape[0], dtype=torch.float64)
    else:
        metric_w = metric.double()
        if metric_w.shape != (work.shape[0], work.shape[0]):
            raise ValueError("metric must be square on the output dimension")
    pruned = ~keep
    fixed = torch.zeros_like(work)
    fixed[pruned] = work[pruned]
    free = keep.reshape(-1)
    n_free = int(free.sum())
    if n_free == 0:
        return torch.zeros_like(weight)

    def apply(vector: torch.Tensor) -> torch.Tensor:
        perturbation = torch.zeros_like(work)
        perturbation.reshape(-1)[free] = vector
        gradient = metric_w @ perturbation @ hessian_w
        return gradient.reshape(-1)[free]

    rhs = -(metric_w @ fixed @ hessian_w).reshape(-1)[free]
    solution = _conjugate_gradient(apply, rhs, iterations or n_free)
    residual = fixed.clone()
    residual.reshape(-1)[free] = solution
    restored = work - residual
    restored[pruned] = 0
    return restored.to(dtype=weight.dtype)


def _conjugate_gradient(apply, rhs: torch.Tensor, iterations: int) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rr = torch.dot(residual, residual)
    if float(rr) == 0.0:
        return solution
    for _ in range(iterations):
        Ad = apply(direction)
        denom = torch.dot(direction, Ad)
        if float(denom.abs()) < 1e-18:
            break
        alpha = rr / denom
        solution = solution + alpha * direction
        residual = residual - alpha * Ad
        rr_next = torch.dot(residual, residual)
        if float(rr_next) < 1e-18:
            break
        direction = residual + (rr_next / rr) * direction
        rr = rr_next
    return solution


class SparseLowRankSparseGPT(FixedUnionSparseGPT):
    """Variant D: ambient SparseGPT, then Ŵ = W_sp + Π (W - W_sp)."""

    @torch.no_grad()
    def fasterprune(self, sparsity: float, *args, **kwargs) -> None:
        self.finalize_union()
        dense = _weight_matrix(self.layer).float()
        SparseGPT.fasterprune(self, sparsity, *args, **kwargs)
        sparse = _weight_matrix(self.layer).float()
        basis = self.union_basis
        if basis is None:
            raise RuntimeError("fixed union basis was not finalized")
        hat = sparse_plus_low_rank(dense, sparse, basis.to(device=sparse.device))
        if isinstance(self.layer, transformers.Conv1D):
            hat = hat.t()
        self.layer.weight.data = hat.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)


class FisherScaleSparseGPT(SparseGPT):
    """Variant B rescaling with Fisher row weights ω = 1 + α g^2 / mean(g^2)."""

    def __init__(self, layer: nn.Module, *, alpha: float = 1.0) -> None:
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("fisher alpha must be finite and nonnegative")
        super().__init__(layer)
        self.alpha = float(alpha)
        self._phi: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None

    def add_fisher(self, phi: torch.Tensor, grad: torch.Tensor) -> None:
        if phi.ndim != 2 or grad.ndim != 2 or phi.shape[0] != grad.shape[0]:
            raise ValueError("phi and grad must share a sample axis")
        if phi.shape[1] != self.columns or grad.shape[1] != _weight_matrix(self.layer).shape[0]:
            raise ValueError("fisher tensors must match the layer input and output sizes")
        self._phi = phi.detach().float().cpu()
        self._grad = grad.detach().float().cpu()

    def finalize_fisher(self) -> None:
        if self._phi is None or self._grad is None:
            raise RuntimeError("call add_fisher before pruning")

    def fasterprune(self, *args, **kwargs) -> None:
        self.finalize_fisher()
        omega = fisher_sample_weights(self._grad, self.alpha)
        scale = row_scales_from_sample_weights(omega, self._phi)
        prune_with_rescaling(self, scale, *args, **kwargs)
