"""Option 2 — per-chart OBS scores with chart-local H_c compensation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import (
    SampleReservoir,
    apply_bias_fold_from_samples,
    ensure_linear_or_conv1d,
    fit_output_chart_atlas,
)


def per_chart_obs_scores(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    chart_basis: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Projected chart OBS costs for a fixed chart basis U_c.

    For ambient entry (p, q), the note's restricted-compensation cost contains
    ``||U_c^T e_p||_2^2 * W_{pq}^2 / [H_c^{-1}]_{qq}``.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be [d_out, d_in]")
    if inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("inputs must be [N, d_in] matching weight columns")
    if chart_basis.ndim == 1:
        chart_basis = chart_basis.reshape(-1, 1)
    if chart_basis.shape[0] != weight.shape[0]:
        raise ValueError("chart basis ambient dim must match weight rows")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    phi = inputs.float()
    n = phi.shape[0]
    h = (phi.T @ phi) / max(n, 1)
    h = h + ridge * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    h = 0.5 * (h + h.T)
    h_inv = torch.linalg.inv(h)
    inv_diag = torch.diag(h_inv).clamp_min(1e-30)

    u = chart_basis.float()
    row_factor = u.square().sum(dim=1).clamp_min(0.0)
    scores = (weight.float().square() / inv_diag[None, :]) * row_factor[:, None]
    return scores.to(dtype=weight.dtype)


def _chart_input_hessian(inputs: torch.Tensor, ridge: float) -> torch.Tensor:
    phi = inputs.float()
    n = max(phi.shape[0], 1)
    h = (phi.T @ phi) / n
    h = h + ridge * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    return 0.5 * (h + h.T)


def _invert_pd(hessian: torch.Tensor, percdamp: float) -> torch.Tensor:
    """SparseGPT-style Cholesky factor of H^{-1} (upper)."""
    mean_diag = torch.mean(torch.diag(hessian)).clamp_min(1e-12)
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    damp_scale = float(percdamp)
    for _ in range(20):
        damped = hessian.clone()
        damped[diagonal, diagonal] += damp_scale * mean_diag
        damped = 0.5 * (damped + damped.T)
        try:
            cholesky = torch.linalg.cholesky(damped)
            inverse = torch.cholesky_inverse(cholesky)
            inverse = 0.5 * (inverse + inverse.T)
            return torch.linalg.cholesky(inverse, upper=True)
        except Exception:
            damp_scale *= 2.0
    raise RuntimeError("chart Hessian Cholesky failed after damp increases")


class ChartObsSparseGPT(SparseGPT):
    """Option 2: per-chart OBS mask scores; compensate with each row's H_c*.

    Mask selection aggregates chart costs (heuristic when charts > 1). Compensation
    uses the chart-conditional input curvature for the row's dominant chart
    ``c*(p) = argmax_c ||U_c^T e_p||^2``, which recovers the note's
    row-restricted single-chart OBS when ``charts == 1``.
    """

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
    ) -> None:
        ensure_linear_or_conv1d(layer, "Chart-OBS")
        if charts < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("charts/rank must be positive and samples >= 3")
        if not math.isfinite(percdamp) or percdamp < 0:
            raise ValueError("percdamp must be finite and nonnegative")
        super().__init__(layer)
        self.charts = charts
        self.rank = rank
        self.seed = seed
        self.chart_ridge = float(percdamp)
        self.bias_fold = bias_fold
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.chart_scores: torch.Tensor | None = None
        self.chart_hessians: list[torch.Tensor] | None = None
        self.chart_bases: list[torch.Tensor] | None = None
        self.row_chart_index: torch.Tensor | None = None
        self.sample_projector_diagonals: torch.Tensor | None = None
        self.compensation_source: str | None = None
        self.tangent_diagnostics: dict | None = None
        self._dense_weight: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after chart-OBS finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Chart-OBS pruning requires tensor layer outputs")
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized:
            raise RuntimeError("Cannot add batches after chart-OBS finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Chart-OBS pruning requires tensor layer outputs")
        # Still accumulate ambient H for API compatibility / diagnostics, but
        # compensation does not use it (see fasterprune).
        super().add_batch(inputs, out, blocksize)
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def finalize_charts(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None or self.reservoir.outputs is None:
            raise RuntimeError("Collect calibration samples before chart-OBS prune")
        atlas = fit_output_chart_atlas(
            self.reservoir.outputs,
            charts=self.charts,
            rank=self.rank,
            seed=self.seed,
        )
        weight = _weight_matrix(self.layer).detach().cpu().float()
        scores = torch.zeros_like(weight)
        hessians: list[torch.Tensor] = []
        row_factors = []
        chart_sizes = []
        pdiag = torch.zeros(
            (self.reservoir.outputs.shape[0], weight.shape[0]), dtype=torch.float32
        )
        for chart_id, basis in enumerate(atlas.bases):
            members = atlas.assignments == chart_id
            count = int(members.sum().item())
            chart_sizes.append(count)
            row_factor = basis.square().sum(dim=1).clamp_min(0.0)
            row_factors.append(row_factor)
            pdiag[members] = row_factor.to(dtype=torch.float32)
            if count < 2:
                # Degenerate chart: keep a ridge identity so indexing stays valid.
                dim = weight.shape[1]
                hessians.append(torch.eye(dim, dtype=torch.float32))
                continue
            chart_inputs = self.reservoir.inputs[members]
            hessians.append(_chart_input_hessian(chart_inputs, self.chart_ridge))
            scores = scores + per_chart_obs_scores(
                weight,
                chart_inputs,
                basis,
                ridge=self.chart_ridge,
            )
        stacked_factors = torch.stack(row_factors, dim=0)
        self.row_chart_index = stacked_factors.argmax(dim=0).to(dtype=torch.long)
        self.chart_scores = scores
        self.chart_hessians = hessians
        self.chart_bases = list(atlas.bases)
        self.sample_projector_diagonals = pdiag
        self.compensation_source = "chart"
        self.tangent_diagnostics = {
            "sample_count": int(self.reservoir.inputs.shape[0]),
            "chart_count": len(atlas.bases),
            "chart_ranks": list(atlas.ranks),
            "chart_counts": chart_sizes,
            "mean_chart_score": float(scores.mean()),
            "compensation_source": "chart",
        }
        self._dense_weight = weight.clone()
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
        _ = (mask_weight, score_matrix)
        if zeros_per_group != 0:
            raise ValueError("Chart-OBS currently supports unstructured sparsity only")
        self.finalize_charts()
        assert self.chart_scores is not None
        assert self.chart_hessians is not None
        assert self.row_chart_index is not None

        weights = _weight_matrix(self.layer).float()
        # Drop ambient H ownership; compensation uses chart Hessians only.
        self.H = None
        chart_factors = [
            _invert_pd(h.to(device=self.device), percdamp) for h in self.chart_hessians
        ]
        row_chart = self.row_chart_index.to(device=self.device)
        scores = self.chart_scores.to(device=self.device, dtype=weights.dtype)

        _ = blocksize  # retained for SparseGPT.fasterprune signature compatibility
        prune_count = int(scores.numel() * sparsity)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if prune_count:
            chosen = torch.topk(scores.flatten(), prune_count, largest=False).indices
            mask.flatten()[chosen] = True

        # Full-width OBS sweep: each ambient row uses its dominant chart's H_c^{-1}.
        working = weights.clone()
        for column in range(self.columns):
            for chart_id, factor in enumerate(chart_factors):
                rows = torch.nonzero(row_chart == chart_id, as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                diag = factor[column, column].clamp_min(1e-30)
                inv_tail = factor[column, column:]
                current = working[rows, column]
                pruned = current.clone()
                pruned[mask[rows, column]] = 0
                error = (current - pruned) / diag
                working[rows, column:] -= error.unsqueeze(1) * inv_tail.unsqueeze(0)
                working[rows, column] = pruned

        weights.copy_(working)
        if isinstance(self.layer, transformers.Conv1D):
            weights = weights.t()
        self.layer.weight.data = weights.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if self.bias_fold:
            apply_bias_fold_from_samples(
                self.layer,
                dense_weight=self._dense_weight,
                sample_inputs=self.reservoir.inputs,
                sample_outputs=self.reservoir.outputs,
                projector_diagonals=self.sample_projector_diagonals,
            )

    def free(self) -> None:
        self.reservoir.clear()
        self.chart_scores = None
        self.chart_hessians = None
        self.chart_bases = None
        self.row_chart_index = None
        self.sample_projector_diagonals = None
        self._dense_weight = None
        super().free()
