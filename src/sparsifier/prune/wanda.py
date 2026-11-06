"""Wanda baseline: |W|·RMS(activations) mask, no OBS compensation."""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import SampleReservoir, ensure_linear_or_conv1d


def wanda_scores(weight: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Squared Wanda scores ``W_{pq}^2 * mean_n phi_{q,n}^2``.

    Ranking by these scores matches ranking by ``|W_{pq}| ||phi_q||_2`` per
    the note (Wanda's squared score is a diagonal input-covariance approximation
    to ambient reconstruction).
    """
    if weight.ndim != 2:
        raise ValueError("weight must be [d_out, d_in]")
    phi = inputs.reshape(-1, inputs.shape[-1]).float()
    if phi.shape[1] != weight.shape[1]:
        raise ValueError("inputs last dim must match weight columns")
    mean_phi2 = phi.square().mean(dim=0)
    return weight.float().square() * mean_phi2[None, :]


class WandaPruner(SparseGPT):
    """Activation-aware magnitude pruning without Hessian compensation."""

    def __init__(self, layer: nn.Module, *, calibration_samples: int = 512, seed: int = 0) -> None:
        ensure_linear_or_conv1d(layer, "Wanda")
        if calibration_samples < 3:
            raise ValueError("calibration_samples must be >= 3")
        super().__init__(layer)
        self.reservoir = SampleReservoir(calibration_samples, seed)
        self.scores: torch.Tensor | None = None
        self.finalized = False

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized:
            raise RuntimeError("Cannot add batches after Wanda finalization")
        if not isinstance(out, torch.Tensor):
            raise TypeError("Wanda pruning requires tensor layer outputs")
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        self.nsamples += int(inputs.shape[0])
        self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        # Wanda never uses the ambient Hessian.
        self.add_tangent_batch(inputs, out, blocksize)

    @torch.no_grad()
    def add_hessian_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = (inputs, out, blocksize)

    @torch.no_grad()
    def finalize_scores(self) -> None:
        if self.finalized:
            return
        if self.reservoir.inputs is None:
            raise RuntimeError("Collect activations before Wanda prune")
        weight = _weight_matrix(self.layer).detach().cpu().float()
        self.scores = wanda_scores(weight, self.reservoir.inputs)
        if isinstance(self.layer, transformers.Conv1D) and self.scores.shape != weight.shape:
            self.scores = self.scores.T.contiguous()
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
        _ = (group_size, blocksize, percdamp, mask_weight, score_matrix)
        if zeros_per_group != 0:
            raise ValueError("Wanda currently supports unstructured sparsity only")
        self.finalize_scores()
        assert self.scores is not None
        weights = _weight_matrix(self.layer).float().cpu()
        # Per-row comparison as in Wanda / the note.
        for row in range(weights.shape[0]):
            scores = self.scores[row]
            prune_count = int(scores.numel() * sparsity)
            if prune_count <= 0:
                continue
            chosen = torch.topk(scores, prune_count, largest=False).indices
            weights[row, chosen] = 0
        if isinstance(self.layer, transformers.Conv1D):
            weights = weights.t()
        self.layer.weight.data = weights.reshape(self.layer.weight.shape).to(
            device=self.layer.weight.device, dtype=self.layer.weight.dtype
        )

    def free(self) -> None:
        self.reservoir.clear()
        self.scores = None
        super().free()
