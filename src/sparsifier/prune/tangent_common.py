"""Shared helpers for output-tangent pruning techniques."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.bias import projected_bias_correction
from sparsifier.prune.sparsegpt import _weight_matrix


@dataclass
class ChartAtlas:
    """K-means chart assignments plus per-chart PCA tangent bases."""

    assignments: torch.Tensor  # [N]
    bases: list[torch.Tensor]  # each [d_out, r_c]
    centers: torch.Tensor  # [C, d_out]
    ranks: list[int]


@torch.no_grad()
def fit_output_chart_atlas(
    outputs: torch.Tensor,
    *,
    charts: int,
    rank: int,
    seed: int = 0,
    iterations: int = 8,
) -> ChartAtlas:
    """Cluster outputs and estimate a local PCA tangent basis per chart."""
    if outputs.ndim != 2 or outputs.shape[0] < 3:
        raise ValueError("outputs must be [N, d] with N >= 3")
    if charts < 1 or rank < 1 or iterations < 1:
        raise ValueError("charts, rank, and iterations must be positive")
    y = outputs.float()
    n_samples, output_dim = y.shape
    chart_count = min(charts, max(1, n_samples // (rank + 2)))
    effective_rank = min(rank, output_dim, n_samples - chart_count)
    device = y.device
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
    assignments = torch.cdist(y, centers).argmin(dim=1)
    counts = torch.bincount(assignments, minlength=chart_count)
    if bool((counts < 2).any()):
        if chart_count > 1:
            return fit_output_chart_atlas(
                outputs,
                charts=chart_count - 1,
                rank=rank,
                seed=seed,
                iterations=iterations,
            )
        raise ValueError("k-means produced a chart with fewer than two samples")

    bases: list[torch.Tensor] = []
    ranks: list[int] = []
    for chart_id in range(chart_count):
        members = y[assignments == chart_id]
        centered = members - members.mean(dim=0, keepdim=True)
        gram = centered @ centered.T
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
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
        bases.append(basis.contiguous())
        ranks.append(keep)
    return ChartAtlas(assignments=assignments, bases=bases, centers=centers, ranks=ranks)


class SampleReservoir:
    """Reservoir of paired layer input/output tokens on CPU."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 3:
            raise ValueError("calibration reservoir capacity must be >= 3")
        self.capacity = capacity
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self.keys: torch.Tensor | None = None
        self.inputs: torch.Tensor | None = None
        self.outputs: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        x = inputs.reshape(-1, inputs.shape[-1]).detach()
        y = outputs.reshape(-1, outputs.shape[-1]).detach()
        if x.shape[0] != y.shape[0]:
            raise ValueError("input/output sample counts must match")
        old_count = 0 if self.keys is None else self.keys.shape[0]
        new_keys = torch.rand(x.shape[0], generator=self._rng, device="cpu")
        candidate = new_keys if self.keys is None else torch.cat((self.keys, new_keys))
        keep_count = min(self.capacity, candidate.shape[0])
        keep = torch.topk(candidate, keep_count, largest=True, sorted=False).indices
        old_mask = keep < old_count
        kept_x = torch.empty((keep_count, x.shape[1]), dtype=torch.float32)
        kept_y = torch.empty((keep_count, y.shape[1]), dtype=torch.float32)
        if old_mask.any():
            assert self.inputs is not None and self.outputs is not None
            old_indices = keep[old_mask]
            targets = torch.nonzero(old_mask, as_tuple=False).flatten()
            kept_x[targets] = self.inputs[old_indices]
            kept_y[targets] = self.outputs[old_indices]
        new_slots = keep[keep >= old_count] - old_count
        if new_slots.numel():
            new_x = x[new_slots.to(device=x.device)].to(device="cpu", dtype=torch.float32)
            new_y = y[new_slots.to(device=y.device)].to(device="cpu", dtype=torch.float32)
            targets = torch.nonzero(keep >= old_count, as_tuple=False).flatten()
            kept_x[targets] = new_x
            kept_y[targets] = new_y
        self.keys = candidate[keep].contiguous()
        self.inputs = kept_x.contiguous()
        self.outputs = kept_y.contiguous()

    def clear(self) -> None:
        self.keys = None
        self.inputs = None
        self.outputs = None


def apply_bias_fold_from_samples(
    layer: nn.Module,
    *,
    dense_weight: torch.Tensor | None,
    sample_inputs: torch.Tensor | None,
    sample_outputs: torch.Tensor | None,
    projector_diagonals: torch.Tensor | None = None,
    union_basis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fold projected residual mean into ``layer.bias``; return Δb.

    Metric priority (note §3):
    1. ``union_basis`` → ``M_n = U U^T`` for every sample (fixed-union case).
    2. ``projector_diagonals`` ``[N, d]`` → diagonal metrics ``diag(P_n)``.
    3. Otherwise ``M_n = I`` (ambient / global-mean bias).
    """
    _ = sample_outputs
    if dense_weight is None or sample_inputs is None:
        raise ValueError("bias folding requires retained dense weights and input samples")
    if projector_diagonals is not None and union_basis is not None:
        raise ValueError("pass at most one of projector_diagonals and union_basis")
    # Many HF linears (e.g. Qwen) have bias=False; skip rather than fail a run.
    if getattr(layer, "bias", None) is None or layer.bias is None:
        output_dim = _weight_matrix(layer).shape[0]
        return torch.zeros(output_dim, dtype=torch.float32)
    sparse = _weight_matrix(layer).detach().cpu().float()
    dense = dense_weight.detach().cpu().float()
    if dense.shape != sparse.shape:
        raise ValueError("dense/sparse weight shapes do not match")
    delta_w = dense - sparse
    residual = (delta_w @ sample_inputs.T).T  # [N, d]
    output_dim = residual.shape[1]
    n_samples = residual.shape[0]
    if union_basis is not None:
        basis = union_basis.detach().cpu().float()
        if basis.ndim == 1:
            basis = basis.reshape(-1, 1)
        if basis.shape[0] != output_dim:
            raise ValueError("union_basis ambient dim must match residual")
        projector = basis @ basis.T
        metrics = projector.expand(n_samples, output_dim, output_dim).clone()
    elif projector_diagonals is None:
        metrics = (
            torch.eye(output_dim, dtype=torch.float32)
            .expand(n_samples, output_dim, output_dim)
            .clone()
        )
    else:
        if projector_diagonals.shape != residual.shape:
            raise ValueError("projector_diagonals must match residual shape [N, d]")
        metrics = torch.diag_embed(projector_diagonals.float())
    delta_b = projected_bias_correction(residual, metrics)
    # Note §3: residual before correction is Aφ; we subtract Δb from the
    # activation, equivalent to adding Δb into the layer bias when the layer
    # computes Wφ + b.
    layer.bias.data = layer.bias.data + delta_b.to(
        device=layer.bias.device, dtype=layer.bias.dtype
    )
    return delta_b


def ensure_linear_or_conv1d(layer: nn.Module, label: str) -> None:
    if not isinstance(layer, (nn.Linear, transformers.Conv1D)):
        raise TypeError(f"{label} supports Linear and Conv1D only")
