"""§1 union-tangent knockout scores as an Option-1-style mask heuristic."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent import estimate_output_tangent_coupling
from sparsifier.prune.tangent_common import SampleReservoir, apply_bias_fold_from_samples, ensure_linear_or_conv1d


class KnockoutSparseGPT(SparseGPT):
    """Rank weights by I^{tan,diag} = W^2 G (mean φ_q^2); ambient compensation."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        charts: int = 8,
        rank: int = 1,
        calibration_samples: int = 512,
        seed: int = 0,
        bias_fold: bool = False,
    ) -> None:
        ensure_linear_or_conv1d(layer, "Knockout")
        if charts < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("charts/rank must be positive and calibration_samples >= 3")
        super().__init__(layer)
        self.charts = charts
        self.rank = rank
        self.seed = seed
        self.bias_fold = bias_fold
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.knockout_scores: torch.Tensor | None = None
        self.mask_weight: torch.Tensor | None = None
        self.sample_projector_diagonals: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after knockout finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Knockout pruning requires tensor layer outputs")
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized:
            raise RuntimeError("Cannot add batches after knockout finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Knockout pruning requires tensor layer outputs")
        super().add_batch(inputs, out, blocksize)
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def finalize_scores(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None or self.reservoir.outputs is None:
            raise RuntimeError("Collect calibration samples before knockout scoring")
        g, diagnostics = estimate_output_tangent_coupling(
            self.reservoir.inputs,
            self.reservoir.outputs,
            charts=self.charts,
            rank=self.rank,
            alpha=1.0,
            seed=self.seed,
        )
        weight = _weight_matrix(self.layer).detach().cpu().float()
        mean_phi2 = self.reservoir.inputs.square().mean(dim=0)
        # I^{tan,diag}_{pq} = W_pq^2 G_pq (1/N sum phi_q^2)
        self.knockout_scores = weight.square() * g * mean_phi2[None, :]
        self.mask_weight = (g * mean_phi2[None, :]).clamp_min(0.0)
        if isinstance(self.layer, transformers.Conv1D) and self.mask_weight.shape != weight.shape:
            self.mask_weight = self.mask_weight.T.contiguous()
            self.knockout_scores = self.knockout_scores.T.contiguous()
        diags = diagnostics.get("sample_projector_diagonals")
        self.sample_projector_diagonals = (
            None if diags is None else diags.detach().cpu().float()
        )
        self._dense_weight = weight.clone()
        self.finalized = True

    def fasterprune(self, *args, **kwargs) -> None:
        self.finalize_scores()
        # Rank by absolute I^{tan,diag}; keep ambient Hessian for compensation.
        kwargs.pop("mask_weight", None)
        kwargs.pop("score_matrix", None)
        super().fasterprune(*args, score_matrix=self.knockout_scores, **kwargs)
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
        self.knockout_scores = None
        self.mask_weight = None
        self.sample_projector_diagonals = None
        self._dense_weight = None
        super().free()
