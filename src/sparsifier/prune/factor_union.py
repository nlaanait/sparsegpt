"""Concept-factor union projectors and Option-1 G_pq with true span rank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.geometry_features import block_dictionary_factor_state
from sparsifier.prune.scores import union_basis
from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import (
    SampleReservoir,
    apply_bias_fold_from_samples,
    ensure_linear_or_conv1d,
    fit_output_chart_atlas,
)

FACTOR_SOURCES = ("kmeans", "block_dictionary")


def union_projector_diagonal(
    factor_bases: list[torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Diagonal of P_union = Q Q^T and the actual union rank (not sum r_i)."""
    q, rank = union_basis(factor_bases)
    diag = (q * q).sum(dim=1).clamp_(0.0, 1.0)
    return diag, rank


def estimate_factor_union_coupling(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    factor_bases: torch.Tensor,
    active: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """G_pq from per-sample union projectors of active concept-factor tangents.

    Parameters
    ----------
    factor_bases:
        ``[M, d_out, r]`` factor tangent bases.
    active:
        ``[N, M]`` boolean mask of factors active on each sample.
    alpha:
        Convex blend ``(1-alpha) + alpha G`` (same knob as chart Option 1).
    """
    _ = outputs
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if inputs.ndim != 2 or active.ndim != 2:
        raise ValueError("inputs must be [N, d_in] and active [N, M]")
    if inputs.shape[0] != active.shape[0]:
        raise ValueError("inputs and active must share the sample dimension")
    if factor_bases.ndim != 3:
        raise ValueError("factor_bases must have shape [M, d_out, r]")
    n_samples, _input_dim = inputs.shape
    n_factors, output_dim, _rank = factor_bases.shape
    if active.shape[1] != n_factors:
        raise ValueError("active factor count must match factor_bases")

    x = inputs.float()
    denominator = x.square().sum(dim=0)
    pdiag = torch.zeros((n_samples, output_dim), dtype=torch.float32)
    ranks = []
    for n in range(n_samples):
        ids = torch.nonzero(active[n], as_tuple=False).flatten().tolist()
        if not ids:
            ranks.append(0)
            continue
        bases = [factor_bases[i] for i in ids]
        # Drop numerically empty factor columns so QR rank stays well-defined.
        bases = [b for b in bases if float(b.norm()) > 1e-12]
        if not bases:
            ranks.append(0)
            continue
        diag, rank = union_projector_diagonal(bases)
        pdiag[n] = diag.to(dtype=torch.float32)
        ranks.append(rank)

    numerator = pdiag.T @ x.square()
    coupling = (numerator / denominator.clamp_min(1e-12)[None, :]).clamp_(0.0, 1.0)
    score = (1.0 - alpha) + alpha * coupling
    diagnostics = {
        "sample_count": n_samples,
        "factor_count": n_factors,
        "mean_union_rank": float(sum(ranks) / max(len(ranks), 1)),
        "inactive_input_features": int((denominator <= 1e-12).sum().item()),
        "mean_tangent_coupling": float(coupling.mean()),
        "mean_score_weight": float(score.mean()),
        "sample_projector_diagonals": pdiag.detach().cpu(),
    }
    return score, diagnostics


def _kmeans_factor_state(
    outputs: torch.Tensor,
    *,
    factors: int,
    rank: int,
    seed: int,
    active_factors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ablation path: k-means chart centers as concept factors."""
    atlas = fit_output_chart_atlas(
        outputs,
        charts=factors,
        rank=rank,
        seed=seed,
    )
    n_factors = len(atlas.bases)
    output_dim = outputs.shape[1]
    max_rank = max(b.shape[1] for b in atlas.bases)
    bases = torch.zeros((n_factors, output_dim, max_rank), dtype=torch.float32)
    for i, basis in enumerate(atlas.bases):
        bases[i, :, : basis.shape[1]] = basis
    distances = torch.cdist(outputs.float(), atlas.centers.float())
    k = min(active_factors, n_factors)
    nearest = torch.topk(distances, k=k, largest=False, dim=1).indices
    active = torch.zeros((distances.shape[0], n_factors), dtype=torch.bool)
    active.scatter_(1, nearest, True)
    return bases, active


class FactorUnionSparseGPT(SparseGPT):
    """Option 1 with concept-factor union projectors (multi-factor active sets)."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        alpha: float = 1.0,
        factors: int = 4,
        rank: int = 1,
        calibration_samples: int = 512,
        seed: int = 0,
        bias_fold: bool = False,
        active_factors: int = 2,
        factor_source: str = "kmeans",
        block_width: int = 3,
        fit_iterations: int = 20,
    ) -> None:
        ensure_linear_or_conv1d(layer, "Factor-union")
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be finite and in (0, 1]")
        if factors < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("factors/rank must be positive and samples >= 3")
        if factor_source not in FACTOR_SOURCES:
            raise ValueError(
                f"factor_source must be one of {FACTOR_SOURCES}, got {factor_source!r}"
            )
        if factor_source == "block_dictionary":
            if block_width < 1:
                raise ValueError("block_width must be positive")
            if rank > block_width:
                raise ValueError("rank must be <= block_width for block_dictionary")
            if not 1 <= active_factors <= factors:
                raise ValueError("active_factors must lie in [1, factors]")
        super().__init__(layer)
        self.alpha = alpha
        self.factors = factors
        self.rank = rank
        self.seed = seed
        self.bias_fold = bias_fold
        self.active_factors = max(1, min(active_factors, factors))
        self.factor_source = factor_source
        self.block_width = block_width
        self.fit_iterations = fit_iterations
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.mask_weight: torch.Tensor | None = None
        self.tangent_diagnostics: dict | None = None
        self.sample_projector_diagonals: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after factor-union finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Factor-union pruning requires tensor layer outputs")
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized:
            raise RuntimeError("Cannot add batches after factor-union finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Factor-union pruning requires tensor layer outputs")
        super().add_batch(inputs, out, blocksize)
        self.reservoir.update(inputs, out)

    def _factor_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.reservoir.outputs is not None
        if self.factor_source == "kmeans":
            return _kmeans_factor_state(
                self.reservoir.outputs,
                factors=self.factors,
                rank=self.rank,
                seed=self.seed,
                active_factors=self.active_factors,
            )
        return block_dictionary_factor_state(
            self.reservoir.outputs,
            groups=self.factors,
            width=self.block_width,
            active=self.active_factors,
            rank=self.rank,
            seed=self.seed,
            fit_iterations=self.fit_iterations,
        )

    @torch.no_grad()
    def finalize_tangents(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None or self.reservoir.outputs is None:
            raise RuntimeError("Collect calibration samples before factor-union prune")
        bases, active = self._factor_state()
        self.mask_weight, self.tangent_diagnostics = estimate_factor_union_coupling(
            self.reservoir.inputs,
            self.reservoir.outputs,
            factor_bases=bases,
            active=active,
            alpha=self.alpha,
        )
        if self.tangent_diagnostics is not None:
            self.tangent_diagnostics["factor_source"] = self.factor_source
            diags = self.tangent_diagnostics.get("sample_projector_diagonals")
            self.sample_projector_diagonals = (
                None if diags is None else diags.detach().cpu().float()
            )
        expected = _weight_matrix(self.layer).shape
        if self.mask_weight.shape != expected and isinstance(self.layer, transformers.Conv1D):
            self.mask_weight = self.mask_weight.T.contiguous()
        if self.mask_weight.shape != expected:
            raise RuntimeError("Factor-union score shape does not match weight layout")
        self._dense_weight = _weight_matrix(self.layer).detach().cpu().clone()
        self.finalized = True

    def fasterprune(self, *args, **kwargs) -> None:
        self.finalize_tangents()
        super().fasterprune(*args, mask_weight=self.mask_weight, **kwargs)
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
        self.mask_weight = None
        self.sample_projector_diagonals = None
        self._dense_weight = None
        super().free()
