"""§1 union-tangent editability scores as an Option-1-style mask heuristic."""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent import estimate_output_tangent_coupling
from sparsifier.prune.tangent_common import SampleReservoir, apply_bias_fold_from_samples, ensure_linear_or_conv1d


class EditabilitySparseGPT(SparseGPT):
    """Rank weights by G · E[φ²] (Jacobian energy without |W|); ambient compensation."""

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
        ensure_linear_or_conv1d(layer, "Editability")
        if charts < 1 or rank < 1 or calibration_samples < 3:
            raise ValueError("charts/rank must be positive and calibration_samples >= 3")
        super().__init__(layer)
        self.charts = charts
        self.rank = rank
        self.seed = seed
        self.bias_fold = bias_fold
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.editability_scores: torch.Tensor | None = None
        self.mask_weight: torch.Tensor | None = None
        self.sample_projector_diagonals: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after editability finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Editability pruning requires tensor layer outputs")
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized:
            raise RuntimeError("Cannot add batches after editability finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Editability pruning requires tensor layer outputs")
        super().add_batch(inputs, out, blocksize)
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def finalize_scores(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None or self.reservoir.outputs is None:
            raise RuntimeError("Collect calibration samples before editability scoring")
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
        # Editability-style energy: G · E[φ_q²] without the W² knockout factor.
        # For a linear layer, ∂h/∂W_pq = φ_q e_p, so scores track Jacobian energy.
        self.editability_scores = (g * mean_phi2[None, :]).clamp_min(0.0)
        self.mask_weight = self.editability_scores
        if isinstance(self.layer, transformers.Conv1D) and self.mask_weight.shape != weight.shape:
            self.mask_weight = self.mask_weight.T.contiguous()
            self.editability_scores = self.editability_scores.T.contiguous()
        diags = diagnostics.get("sample_projector_diagonals")
        self.sample_projector_diagonals = (
            None if diags is None else diags.detach().cpu().float()
        )
        self._dense_weight = weight.clone()
        self.finalized = True

    def fasterprune(self, *args, **kwargs) -> None:
        self.finalize_scores()
        kwargs.pop("mask_weight", None)
        kwargs.pop("score_matrix", None)
        super().fasterprune(*args, score_matrix=self.editability_scores, **kwargs)
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
        self.editability_scores = None
        self.mask_weight = None
        self.sample_projector_diagonals = None
        self._dense_weight = None
        super().free()
