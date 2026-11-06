"""Exact ``L_tan`` OBS under diagonal or full sample-projected metrics.

For diagonal ``M_n = diag(m_n)``, the projected loss separates by output row
with row Hessian ``H_p = (1/N) sum_n m_{n,p} φ_n φ_n^T``. That case is exact
for the stated quadratic. The full Kronecker form
``sum_n (φ_n φ_n^T) ⊗ M_n`` is exact for general PSD ``M_n`` but is only
practical for small layers (``d_out * d_in`` bounded by ``max_full_params``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.metric import blend_anchor_into_diagonals
from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import (
    SampleReservoir,
    apply_bias_fold_from_samples,
    ensure_linear_or_conv1d,
    fit_output_chart_atlas,
)


def diag_p_row_hessians(
    inputs: torch.Tensor,
    projector_diagonals: torch.Tensor,
    *,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """Build ``[d_out, d_in, d_in]`` row Hessians for diagonal metrics."""
    phi = inputs.reshape(-1, inputs.shape[-1]).float()
    diags = projector_diagonals.float()
    if phi.shape[0] != diags.shape[0]:
        raise ValueError("inputs and projector diagonals must share sample count")
    n_samples, d_in = phi.shape
    d_out = diags.shape[1]
    hessians = torch.zeros((d_out, d_in, d_in), dtype=torch.float64)
    # H_p = (1/N) sum_n m_{n,p} φ_n φ_n^T
    for p in range(d_out):
        weights = diags[:, p]
        if float(weights.sum()) <= 0.0:
            continue
        scaled = phi * weights.sqrt()[:, None]
        hessians[p] = (scaled.T @ scaled).to(dtype=torch.float64) / n_samples
        damp = percdamp * torch.mean(torch.diag(hessians[p]))
        hessians[p].diagonal().add_(damp)
    return hessians


def kronecker_projected_hessian(
    inputs: torch.Tensor,
    metrics: torch.Tensor,
    *,
    ridge: float = 1e-7,
) -> torch.Tensor:
    """``sum_n (φ φ^T) ⊗ M_n / N`` in Fortran (column-major) vec order."""
    phi = inputs.reshape(-1, inputs.shape[-1]).float()
    m = metrics.float()
    if phi.shape[0] != m.shape[0]:
        raise ValueError("inputs and metrics must share sample count")
    n_samples, d_in = phi.shape
    d_out = m.shape[1]
    total = d_out * d_in
    h = torch.zeros((total, total), dtype=torch.float64)
    for n in range(n_samples):
        xx = torch.outer(phi[n], phi[n]).to(dtype=torch.float64)
        h += torch.kron(xx, m[n].to(dtype=torch.float64))
    h /= n_samples
    scale = max(float(torch.trace(h) / total), 1.0)
    h.diagonal().add_(ridge * scale)
    return h


def _vec_f(weight: torch.Tensor) -> torch.Tensor:
    """Column-major vectorization of a ``[d_out, d_in]`` matrix."""
    return weight.transpose(0, 1).contiguous().reshape(-1)


def _unvec_f(vec: torch.Tensor, d_out: int, d_in: int) -> torch.Tensor:
    return vec.reshape(d_in, d_out).transpose(0, 1).contiguous()


@torch.no_grad()
def greedy_full_obs(
    hessian: torch.Tensor,
    weight: torch.Tensor,
    sparsity: float,
) -> torch.Tensor:
    """Greedy OBS on ``vec_F(W)``; returns the sparse weight matrix."""
    d_out, d_in = weight.shape
    total = d_out * d_in
    target = _vec_f(weight.float()).to(dtype=torch.float64)
    delta = torch.zeros(total, dtype=torch.float64)
    free = torch.arange(total)
    inv = torch.linalg.inv(hessian)
    prune_count = int(math.floor(sparsity * weight.numel()))
    pruned = torch.zeros(total, dtype=torch.bool)
    for _ in range(prune_count):
        denom = inv.diag()
        residual = target[free] - delta[free]
        costs = residual.square() / denom.clamp_min(1e-30)
        local = int(torch.argmin(costs).item())
        selected = int(free[local].item())
        pruned[selected] = True
        error = float(target[selected] - delta[selected])
        delta[free] = delta[free] + error * inv[:, local] / inv[local, local]
        keep = torch.arange(len(free)) != local
        col = inv[:, local]
        row = inv[local, :]
        inv = inv[keep][:, keep] - torch.outer(col[keep], row[keep]) / inv[local, local]
        free = free[keep]
    sparse = _unvec_f(target - delta, d_out, d_in).to(dtype=torch.float32)
    flat = _vec_f(sparse)
    flat[pruned] = 0.0
    return _unvec_f(flat, d_out, d_in)


@torch.no_grad()
def diag_p_obs_prune(
    weight: torch.Tensor,
    row_hessians: torch.Tensor,
    sparsity: float,
) -> torch.Tensor:
    """Exact greedy OBS for diagonal sample metrics (rows coupled only via budget)."""
    d_out, d_in = weight.shape
    sparse = weight.float().clone()
    prune_count = int(math.floor(sparsity * weight.numel()))
    alive = [torch.arange(d_in) for _ in range(d_out)]
    local_inv: list[torch.Tensor] = []
    for p in range(d_out):
        h = row_hessians[p]
        try:
            local_inv.append(torch.linalg.inv(h))
        except RuntimeError:
            local_inv.append(torch.linalg.pinv(h))
    for _ in range(prune_count):
        best_cost = float("inf")
        best_p = -1
        best_local = -1
        for p in range(d_out):
            free = alive[p]
            if free.numel() == 0:
                continue
            inv = local_inv[p]
            residual = sparse[p, free].to(dtype=inv.dtype)
            costs = residual.square() / inv.diag().clamp_min(1e-30)
            local = int(torch.argmin(costs).item())
            cost = float(costs[local])
            if cost < best_cost:
                best_cost = cost
                best_p = p
                best_local = local
        if best_p < 0:
            break
        p = best_p
        free = alive[p]
        inv = local_inv[p]
        q = int(free[best_local].item())
        error = float(sparse[p, q])
        sparse[p, free] = sparse[p, free] - (
            error * inv[:, best_local].to(sparse.dtype) / inv[best_local, best_local]
        )
        sparse[p, q] = 0.0
        keep = torch.arange(len(free)) != best_local
        col = inv[:, best_local]
        row = inv[best_local, :]
        local_inv[p] = inv[keep][:, keep] - torch.outer(col[keep], row[keep]) / inv[
            best_local, best_local
        ]
        alive[p] = free[keep]
    return sparse


class ExactLtanSparseGPT(SparseGPT):
    """Exact projected OBS under chart-diagonal (or small full) metrics."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        charts: int = 8,
        rank: int = 1,
        calibration_samples: int = 512,
        seed: int = 0,
        percdamp: float = 0.01,
        bias_fold: bool = False,
        anchor_eta: float = 0.0,
        mode: str = "diag",
        max_full_params: int = 4096,
    ) -> None:
        ensure_linear_or_conv1d(layer, "Exact L_tan")
        if charts < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("charts/rank must be positive and calibration_samples >= 3")
        if not math.isfinite(anchor_eta) or anchor_eta < 0.0:
            raise ValueError("anchor_eta must be finite and nonnegative")
        if mode not in {"diag", "full", "auto"}:
            raise ValueError("mode must be diag, full, or auto")
        super().__init__(layer)
        self.charts = charts
        self.rank = rank
        self.seed = seed
        self.percdamp = percdamp
        self.bias_fold = bias_fold
        self.anchor_eta = float(anchor_eta)
        self.mode = mode
        self.max_full_params = max_full_params
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.projector_diagonals: torch.Tensor | None = None
        self.full_metrics: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.resolved_mode: str | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after exact L_tan finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Exact L_tan requires tensor layer outputs")
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        self.nsamples += int(inputs.shape[0])
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        self.add_tangent_batch(inputs, out, blocksize)

    @torch.no_grad()
    def add_hessian_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        # Exact projected OBS does not use the ambient Hessian.
        _ = (inputs, out, blocksize)

    @torch.no_grad()
    def finalize_charts(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None or self.reservoir.outputs is None:
            raise RuntimeError("Collect calibration samples before exact L_tan")
        atlas = fit_output_chart_atlas(
            self.reservoir.outputs,
            charts=self.charts,
            rank=self.rank,
            seed=self.seed,
        )
        diags = torch.stack(
            [basis.square().sum(dim=1).clamp_(0.0, 1.0) for basis in atlas.bases],
            dim=0,
        )
        sample_diags = diags[atlas.assignments]
        sample_diags = blend_anchor_into_diagonals(
            sample_diags,
            atlas.centers,
            atlas.assignments,
            self.anchor_eta,
        )
        self.projector_diagonals = sample_diags.detach().cpu().float()
        weight = _weight_matrix(self.layer)
        params = int(weight.numel())
        want_full = self.mode == "full" or (
            self.mode == "auto" and params <= self.max_full_params
        )
        if want_full:
            # Full metrics: P_union from bases (+ optional anchor).
            metrics = []
            for n, chart_id in enumerate(atlas.assignments.tolist()):
                basis = atlas.bases[chart_id]
                p = basis @ basis.T
                if self.anchor_eta > 0.0:
                    from sparsifier.prune.metric import union_plus_anchor_metric

                    p = union_plus_anchor_metric(p, atlas.centers[chart_id], self.anchor_eta)
                metrics.append(p)
            self.full_metrics = torch.stack(metrics, dim=0).detach().cpu().float()
            self.resolved_mode = "full"
        else:
            self.resolved_mode = "diag"
        self._dense_weight = weight.detach().cpu().float().clone()
        self.finalized = True

    @torch.no_grad()
    def fasterprune(
        self,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        mask_weight: torch.Tensor | None = None,
        score_matrix: torch.Tensor | None = None,
    ) -> None:
        _ = (group_size, blocksize, mask_weight, score_matrix)
        if zeros_per_group != 0:
            raise ValueError("Exact L_tan currently supports unstructured sparsity only")
        self.finalize_charts()
        assert self.projector_diagonals is not None and self.reservoir.inputs is not None
        weight = _weight_matrix(self.layer).detach().cpu().float()
        damp = float(percdamp if percdamp is not None else self.percdamp)
        if self.resolved_mode == "full":
            assert self.full_metrics is not None
            if weight.numel() > self.max_full_params:
                raise ValueError(
                    f"full Kronecker OBS limited to {self.max_full_params} params; "
                    f"got {weight.numel()}"
                )
            h = kronecker_projected_hessian(
                self.reservoir.inputs, self.full_metrics, ridge=max(damp, 1e-7)
            )
            sparse = greedy_full_obs(h, weight, sparsity)
        else:
            row_h = diag_p_row_hessians(
                self.reservoir.inputs, self.projector_diagonals, percdamp=damp
            )
            sparse = diag_p_obs_prune(weight, row_h, sparsity)
        if isinstance(self.layer, transformers.Conv1D):
            sparse = sparse.t()
        self.layer.weight.data = sparse.reshape(self.layer.weight.shape).to(
            device=self.layer.weight.device, dtype=self.layer.weight.dtype
        )
        if self.bias_fold:
            apply_bias_fold_from_samples(
                self.layer,
                dense_weight=self._dense_weight,
                sample_inputs=self.reservoir.inputs,
                sample_outputs=self.reservoir.outputs,
                projector_diagonals=self.projector_diagonals,
            )

    def free(self) -> None:
        self.reservoir.clear()
        self.projector_diagonals = None
        self.full_metrics = None
        self._dense_weight = None
        super().free()


__all__ = [
    "ExactLtanSparseGPT",
    "diag_p_obs_prune",
    "diag_p_row_hessians",
    "greedy_full_obs",
    "kronecker_projected_hessian",
]
