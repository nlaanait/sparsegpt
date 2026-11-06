"""Output-space tangent calibration for the SparseGPT mask heuristic.

This module estimates one local output tangent chart per calibration token,
then weights ambient SparseGPT mask scores by expected diagonal tangent
participation G[p, q]. Error compensation remains the ambient SparseGPT
update; this is a mask-ranking heuristic, not an exact projected OBS solver.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import apply_bias_fold_from_samples


@torch.no_grad()
def estimate_output_tangent_coupling(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    charts: int = 8,
    rank: int = 2,
    alpha: float = 1.0,
    seed: int = 0,
    iterations: int = 8,
    anchor_eta: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Estimate G[p,q] from paired layer inputs/outputs.

    Output samples are clustered into local charts. A rank-r PCA tangent basis
    is estimated in each chart. The basis projector diagonal is weighted by
    input-feature energy in that chart. The estimator represents one chart per
    sample and does not identify multiple overlapping concept factors.

    When ``anchor_eta > 0``, sample diagonals use ``M_n = P_union + η P_anchor``
    (note §3); report that arm separately from pure union metrics.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("tangent alpha must be finite and in [0, 1]")
    if not math.isfinite(anchor_eta) or anchor_eta < 0.0:
        raise ValueError("anchor_eta must be finite and nonnegative")
    if charts < 1 or rank < 1 or iterations < 1:
        raise ValueError("charts, rank, and iterations must be positive")
    if inputs.ndim != 2 or outputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs and outputs must be paired [samples, features] matrices")
    if inputs.shape[0] < 3 or inputs.shape[1] == 0 or outputs.shape[1] == 0:
        raise ValueError("at least three paired nonempty samples are required")

    device = inputs.device
    x = inputs.float()
    y = outputs.to(device=device, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("tangent calibration samples must be finite")
    n_samples, input_dim = x.shape
    output_dim = y.shape[1]
    chart_count = min(charts, max(1, n_samples // (rank + 2)))
    effective_rank = min(rank, output_dim, n_samples - chart_count)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial = torch.randperm(n_samples, generator=generator, device=device)[:chart_count]
    centers = y[initial].clone()

    for _ in range(iterations):
        distances = torch.cdist(y, centers)
        assignments = distances.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=chart_count)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, y)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        if not bool(nonempty.all()):
            nearest = distances.min(dim=1).values
            farthest = torch.argsort(nearest, descending=True)
            empty_ids = torch.nonzero(~nonempty, as_tuple=False).flatten()
            for offset, chart_id in enumerate(empty_ids.tolist()):
                centers[chart_id] = y[farthest[offset]]

    # Reassign after the final center update so each PCA chart uses the
    # nearest final center rather than the penultimate k-means partition.
    assignments = torch.cdist(y, centers).argmin(dim=1)
    counts = torch.bincount(assignments, minlength=chart_count)
    if bool((counts < 2).any()):
        if chart_count > 1:
            # Sparse routed experts can have few samples. Reduce atlas size
            # until every fitted chart has enough support for local PCA.
            return estimate_output_tangent_coupling(
                inputs,
                outputs,
                charts=chart_count - 1,
                rank=rank,
                alpha=alpha,
                seed=seed,
                iterations=iterations,
                anchor_eta=anchor_eta,
            )
        raise ValueError(
            "k-means produced a chart with fewer than two samples; "
            "increase calibration samples or reduce tangent charts"
        )

    tangent_diagonals = []
    chart_counts = []
    chart_ranks = []
    chart_variance = []
    for chart_id in range(chart_count):
        members = y[assignments == chart_id]
        chart_counts.append(int(members.shape[0]))
        if members.shape[0] < 2:
            raise ValueError(
                f"local tangent chart {chart_id} has fewer than two samples; "
                "increase tangent-calibration-samples or reduce tangent-charts"
            )
        centered = members - members.mean(dim=0, keepdim=True)
        gram = centered @ centered.T
        eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        positive = eigenvalues > max(float(eigenvalues[0]) * 1e-7, 1e-12)
        keep = min(effective_rank, int(positive.sum().item()))
        if keep == 0:
            raise ValueError(f"local tangent chart {chart_id} has zero variance")
        basis = centered.T @ eigenvectors[:, :keep]
        basis = basis / eigenvalues[:keep].clamp_min(1e-12).sqrt()[None, :]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        tangent_diagonals.append(basis.square().sum(dim=1).clamp_(0.0, 1.0))
        chart_ranks.append(keep)
        total = eigenvalues.clamp_min(0).sum()
        chart_variance.append(float(eigenvalues[:keep].sum() / total.clamp_min(1e-12)))

    chart_diag = torch.stack(tangent_diagonals, dim=0)
    # Per-sample projector diagonals (P_union,n)_pp for §3 union-metric bias fold.
    sample_projector_diagonals = chart_diag[assignments].detach().cpu()
    if float(anchor_eta) > 0.0:
        from sparsifier.prune.metric import blend_anchor_into_diagonals

        sample_projector_diagonals = blend_anchor_into_diagonals(
            sample_projector_diagonals,
            centers.detach().cpu(),
            assignments.detach().cpu(),
            float(anchor_eta),
        )
        # Rebuild chart_diag aggregates from blended sample diagonals for G_pq.
        chart_diag = torch.zeros_like(chart_diag)
        chart_diag.index_add_(0, assignments, sample_projector_diagonals.to(device=device))
        counts_f = counts.clamp_min(1).to(dtype=chart_diag.dtype)
        chart_diag = chart_diag / counts_f[:, None]
    chart_energy = torch.zeros((chart_count, input_dim), device=device, dtype=torch.float32)
    chart_energy.index_add_(0, assignments, x.square())
    denominator = x.square().sum(dim=0)
    # Keep the dense [output, input] score multiplier on host memory. Transfer
    # only one SparseGPT column block at a time during mask selection.
    score_weight = torch.empty((output_dim, input_dim), dtype=torch.float32, device="cpu")
    coupling_sum = 0.0
    output_chunk = 1024
    for start in range(0, output_dim, output_chunk):
        stop = min(start + output_chunk, output_dim)
        numerator = chart_diag[:, start:stop].T @ chart_energy
        coupling_chunk = (
            numerator / denominator.clamp_min(1e-12)[None, :]
        ).clamp_(0.0, 1.0)
        coupling_sum += float(coupling_chunk.sum())
        score_weight[start:stop] = ((1.0 - alpha) + alpha * coupling_chunk).cpu()
    diagnostics = {
        "sample_count": n_samples,
        "chart_count": chart_count,
        "requested_rank": rank,
        "chart_counts": chart_counts,
        "chart_ranks": chart_ranks,
        "mean_chart_variance_captured": sum(chart_variance) / len(chart_variance),
        "mean_tangent_coupling": coupling_sum / (output_dim * input_dim),
        "mean_score_weight": float(score_weight.mean()),
        "inactive_input_features": int((denominator <= 1e-12).sum().item()),
        "sample_projector_diagonals": sample_projector_diagonals,
        "anchor_eta": float(anchor_eta),
    }
    return score_weight, diagnostics


class TangentAwareSparseGPT(SparseGPT):
    """Ambient SparseGPT compensation with output-tangent-aware mask scores."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        alpha: float = 1.0,
        charts: int = 8,
        rank: int = 2,
        calibration_samples: int = 512,
        seed: int = 0,
        bias_fold: bool = False,
        anchor_eta: float = 0.0,
    ) -> None:
        if not isinstance(layer, (nn.Linear, transformers.Conv1D)):
            raise TypeError("Output tangent calibration supports Linear and Conv1D")
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("tangent alpha must be finite and in (0, 1]")
        if not math.isfinite(anchor_eta) or anchor_eta < 0.0:
            raise ValueError("anchor_eta must be finite and nonnegative")
        if charts < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("charts/rank must be positive and calibration_samples >= 3")
        super().__init__(layer)
        self.alpha = alpha
        self.charts = charts
        self.rank = rank
        self.calibration_samples = calibration_samples
        self.seed = seed
        self.bias_fold = bias_fold
        self.anchor_eta = float(anchor_eta)
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._sample_inputs: torch.Tensor | None = None
        self._sample_outputs: torch.Tensor | None = None
        self._sample_keys: torch.Tensor | None = None
        self.mask_weight: torch.Tensor | None = None
        self.tangent_diagnostics: dict | None = None
        self.sample_projector_diagonals: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def _update_reservoir(self, inputs: torch.Tensor, out: torch.Tensor) -> None:
        x = inputs.reshape(-1, inputs.shape[-1])
        y = out.reshape(-1, out.shape[-1])
        if x.shape[0] != y.shape[0] or x.shape[1] != self.columns:
            raise ValueError("Layer input/output samples do not align with this weight matrix")
        old_count = 0 if self._sample_keys is None else self._sample_keys.shape[0]
        new_keys = torch.rand(x.shape[0], generator=self._rng, device="cpu")
        if self._sample_keys is None:
            candidate_keys = new_keys
        else:
            candidate_keys = torch.cat((self._sample_keys, new_keys))
        keep_count = min(self.calibration_samples, candidate_keys.shape[0])
        keep = torch.topk(candidate_keys, keep_count, largest=True, sorted=False).indices
        old_mask = keep < old_count
        old_indices = keep[old_mask]
        new_slots = keep[keep >= old_count] - old_count
        kept_x = torch.empty((keep_count, x.shape[1]), dtype=torch.float32, device="cpu")
        kept_y = torch.empty((keep_count, y.shape[1]), dtype=torch.float32, device="cpu")
        if old_indices.numel():
            old_targets = torch.nonzero(old_mask, as_tuple=False).flatten()
            kept_x[old_targets] = self._sample_inputs[old_indices]
            kept_y[old_targets] = self._sample_outputs[old_indices]
        if new_slots.numel():
            new_indices = new_slots.to(device=x.device)
            new_x = x[new_indices].detach().to(device="cpu", dtype=torch.float32)
            new_y = y[new_indices].detach().to(device="cpu", dtype=torch.float32)
            target_slots = torch.nonzero(keep >= old_count, as_tuple=False).flatten()
            kept_x[target_slots] = new_x
            kept_y[target_slots] = new_y
        self._sample_keys = candidate_keys[keep].contiguous()
        self._sample_inputs = kept_x.contiguous()
        self._sample_outputs = kept_y.contiguous()

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add calibration batches after tangent finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Output tangent pruning requires tensor layer outputs")
        if inputs.ndim not in (2, 3) or out.ndim not in (2, 3):
            raise ValueError("Expected rank-2 or rank-3 layer input/output batches")
        self._update_reservoir(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized:
            raise RuntimeError("Cannot add calibration batches after tangent finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Output tangent pruning requires tensor layer outputs")
        if inputs.ndim not in (2, 3) or out.ndim not in (2, 3):
            raise ValueError("Expected rank-2 or rank-3 layer input/output batches")
        super().add_batch(inputs, out, blocksize)
        self._update_reservoir(inputs, out)

    @torch.no_grad()
    def finalize_tangents(self) -> None:
        if self.finalized:
            return
        if self._sample_inputs is None or self._sample_outputs is None:
            raise RuntimeError("Collect calibration inputs and outputs before pruning")
        self.mask_weight, self.tangent_diagnostics = estimate_output_tangent_coupling(
            self._sample_inputs.to(self.device),
            self._sample_outputs.to(self.device),
            charts=self.charts,
            rank=self.rank,
            alpha=self.alpha,
            seed=self.seed,
            anchor_eta=self.anchor_eta,
        )
        expected = _weight_matrix(self.layer).shape
        if self.mask_weight.shape != expected:
            if isinstance(self.layer, transformers.Conv1D):
                self.mask_weight = self.mask_weight.T.contiguous()
            if self.mask_weight.shape != expected:
                raise RuntimeError("Tangent score shape does not match SparseGPT weight layout")
        self._dense_weight = _weight_matrix(self.layer).detach().cpu().clone()
        diags = (self.tangent_diagnostics or {}).get("sample_projector_diagonals")
        self.sample_projector_diagonals = (
            None if diags is None else diags.detach().cpu().float()
        )
        self.finalized = True
        if not self.bias_fold:
            self._sample_inputs = None
            self._sample_outputs = None
            self._sample_keys = None

    def fasterprune(self, *args, **kwargs) -> None:
        self.finalize_tangents()
        super().fasterprune(*args, mask_weight=self.mask_weight, **kwargs)
        if self.bias_fold:
            apply_bias_fold_from_samples(
                self.layer,
                dense_weight=self._dense_weight,
                sample_inputs=self._sample_inputs,
                sample_outputs=self._sample_outputs,
                projector_diagonals=self.sample_projector_diagonals,
            )
            self._sample_inputs = None
            self._sample_outputs = None
            self._sample_keys = None

    def free(self) -> None:
        self._sample_inputs = None
        self._sample_outputs = None
        self._sample_keys = None
        self.mask_weight = None
        self.sample_projector_diagonals = None
        self._dense_weight = None
        super().free()
