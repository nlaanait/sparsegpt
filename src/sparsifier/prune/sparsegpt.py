"""One-shot SparseGPT prune for a single linear (or Conv) layer.

Accumulates an activation Hessian with ``add_batch``, then runs
``fasterprune``. Unstructured / N:M sparsity only (no GPTQ path).

Copy/paste of the layer prune from upstream
https://github.com/IST-DASLab/sparsegpt (Frantar & Alistarh, 2023,
https://arxiv.org/abs/2301.00774), minus the GPTQ path. Geometry is not
in upstream — see ``sparsifier.prune.geometry`` for this fork's tangent
term on the same solver.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import transformers


def _weight_matrix(layer: nn.Module) -> torch.Tensor:
    """Layer weights as a 2-D matrix, rows = outputs, columns = inputs."""
    weights = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        weights = weights.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        weights = weights.t()
    return weights


class SparseGPT:
    """Hessian-aware one-shot prune bound to one layer.

    ``H`` is the accumulated activation Hessian. ``nsamples`` counts
    leading-dimension batches passed to ``add_batch``, matching upstream
    (not the flattened token count). ``columns`` is the Hessian width.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.device = layer.weight.device
        weights = _weight_matrix(layer)
        self.columns = weights.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        """Fold one activation batch into the Hessian.

        ``out`` and ``blocksize`` are unused. They stay in the signature so
        callers match upstream SparseGPT.
        """
        _ = (out, blocksize)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        batch_count = inputs.shape[0]
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if inputs.ndim == 3:
                inputs = inputs.reshape((-1, inputs.shape[-1]))
            inputs = inputs.t()
        self.H *= self.nsamples / (self.nsamples + batch_count)
        self.nsamples += batch_count
        inputs = math.sqrt(2 / self.nsamples) * inputs.float()
        self.H += inputs.matmul(inputs.t())

    def add_hessian_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        """Ambient Hessian only (skip tangent reservoirs on subclasses)."""
        SparseGPT.add_batch(self, inputs, out, blocksize)

    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        """Tangent / score reservoir only; default ambient solvers are a no-op."""
        _ = (inputs, out, blocksize)

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
        """Prune ``self.layer`` in place from the accumulated Hessian.

        Parameters
        ----------
        sparsity:
            Target fraction of weights to zero for unstructured pruning
            (``zeros_per_group == 0``). Ignored when ``zeros_per_group > 0``.
        zeros_per_group, group_size:
            N:M structured sparsity: in every block of ``group_size`` consecutive
            columns, zero the ``zeros_per_group`` smallest (by SparseGPT metric)
            weights. ``zeros_per_group == 0`` means unstructured. Example: 2:4 is
            ``zeros_per_group=2, group_size=4``.
        blocksize:
            Column block width for the OBS-style update sweep.
        percdamp:
            Relative Hessian diagonal dampening before the Cholesky inverse.
            Retried with a larger scale if the first factorization fails.
        mask_weight:
            Optional nonnegative multiplier on the ambient SparseGPT metric.
        score_matrix:
            Optional absolute ranking scores. When set, unstructured masks are
            taken from the lowest scores directly (Hessian used for compensation
            only). Mutually exclusive with ``mask_weight``.
        """
        weights = _weight_matrix(self.layer).float()
        if mask_weight is not None and score_matrix is not None:
            raise ValueError("mask_weight and score_matrix are mutually exclusive")
        if mask_weight is not None:
            if mask_weight.shape != weights.shape:
                raise ValueError("mask_weight must have shape " + str(tuple(weights.shape)))
            if not torch.isfinite(mask_weight).all() or (mask_weight < 0).any():
                raise ValueError("mask_weight must be finite and nonnegative")
        if score_matrix is not None:
            if score_matrix.shape != weights.shape:
                raise ValueError("score_matrix must have shape " + str(tuple(weights.shape)))
            if not torch.isfinite(score_matrix).all() or (score_matrix < 0).any():
                raise ValueError("score_matrix must be finite and nonnegative")

        # Same tensor as ``self.H``, not a copy. Clear the attribute so this
        # method owns the only reference while mutating it.
        hessian = self.H
        self.H = None
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0

        # Invert the Hessian via Cholesky. High sparsity / tall FFN layers can
        # make it only barely positive definite after the usual percdamp.
        hessian_inverse = self._invert_hessian(hessian, percdamp)

        global_score_mask = None
        if score_matrix is not None and zeros_per_group == 0:
            # Absolute score ranking is global across columns, not per block.
            prune_count = int(score_matrix.numel() * sparsity)
            global_score_mask = torch.zeros_like(score_matrix, dtype=torch.bool)
            if prune_count:
                chosen = torch.topk(
                    score_matrix.flatten(), prune_count, largest=False
                ).indices
                global_score_mask.flatten()[chosen] = True

        for col_start in range(0, self.columns, blocksize):
            col_end = min(col_start + blocksize, self.columns)
            self._prune_column_block(
                weights,
                hessian_inverse,
                col_start,
                col_end,
                sparsity=sparsity,
                zeros_per_group=zeros_per_group,
                group_size=group_size,
                mask_weight=mask_weight,
                score_matrix=score_matrix,
                preset_mask=None
                if global_score_mask is None
                else global_score_mask[:, col_start:col_end],
            )

        if isinstance(self.layer, transformers.Conv1D):
            weights = weights.t()
        self.layer.weight.data = weights.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def _invert_hessian(self, hessian: torch.Tensor, percdamp: float) -> torch.Tensor:
        mean_diag = torch.mean(torch.diag(hessian))
        diagonal = torch.arange(self.columns, device=self.device)
        damp_scale = float(percdamp)
        for attempt in range(20):
            damped = hessian.clone()
            damped[diagonal, diagonal] += damp_scale * mean_diag
            damped = (damped + damped.T) / 2
            try:
                cholesky = torch.linalg.cholesky(damped)
                inverse = torch.cholesky_inverse(cholesky)
                inverse = (inverse + inverse.T) / 2
                return torch.linalg.cholesky(inverse, upper=True)
            except Exception as exc:
                if "cholesky" not in str(exc).lower() and type(exc).__name__ != "LinAlgError":
                    raise
                damp_scale *= 2.0
                print(
                    f"Cholesky failed (attempt {attempt + 1}); "
                    f"retrying with percdamp={damp_scale:.4g}",
                    flush=True,
                )
        raise RuntimeError(
            f"Hessian Cholesky failed after 20 damp increases (last percdamp={damp_scale:.4g})"
        )

    def _prune_column_block(
        self,
        weights: torch.Tensor,
        hessian_inverse: torch.Tensor,
        col_start: int,
        col_end: int,
        *,
        sparsity: float,
        zeros_per_group: int,
        group_size: int,
        mask_weight: torch.Tensor | None = None,
        score_matrix: torch.Tensor | None = None,
        preset_mask: torch.Tensor | None = None,
    ) -> None:
        weight_block = weights[:, col_start:col_end].clone()
        pruned_block = torch.zeros_like(weight_block)
        error_block = torch.zeros_like(weight_block)
        inverse_block = hessian_inverse[col_start:col_end, col_start:col_end]

        if zeros_per_group == 0:
            if preset_mask is not None:
                mask = preset_mask.to(device=weight_block.device)
            elif score_matrix is not None:
                metric = score_matrix[:, col_start:col_end].to(
                    device=weight_block.device, dtype=weight_block.dtype
                )
                prune_count = int(metric.numel() * sparsity)
                mask = torch.zeros_like(metric, dtype=torch.bool)
                if prune_count:
                    chosen = torch.topk(metric.flatten(), prune_count, largest=False).indices
                    mask.flatten()[chosen] = True
            else:
                metric = weight_block**2 / (torch.diag(inverse_block).reshape((1, -1))) ** 2
                if mask_weight is None:
                    threshold = torch.sort(metric.flatten())[0][int(metric.numel() * sparsity)]
                    mask = metric <= threshold
                else:
                    metric *= mask_weight[:, col_start:col_end].to(
                        device=metric.device, dtype=metric.dtype
                    )
                    prune_count = int(metric.numel() * sparsity)
                    mask = torch.zeros_like(metric, dtype=torch.bool)
                    if prune_count:
                        chosen = torch.topk(metric.flatten(), prune_count, largest=False).indices
                        mask.flatten()[chosen] = True
        else:
            mask = torch.zeros_like(weight_block, dtype=torch.bool)

        for column in range(col_end - col_start):
            column_weights = weight_block[:, column]
            diag = inverse_block[column, column]

            if zeros_per_group != 0 and column % group_size == 0:
                if score_matrix is not None:
                    metric = score_matrix[
                        :, col_start + column : col_start + column + group_size
                    ].to(device=weight_block.device, dtype=weight_block.dtype)
                else:
                    metric = (
                        weight_block[:, column : (column + group_size)] ** 2
                        / (torch.diag(inverse_block)[column : (column + group_size)].reshape((1, -1))) ** 2
                    )
                    if mask_weight is not None:
                        metric *= mask_weight[
                            :, col_start + column : col_start + column + group_size
                        ].to(device=metric.device, dtype=metric.dtype)
                smallest = torch.topk(metric, zeros_per_group, dim=1, largest=False)[1]
                mask.scatter_(1, column + smallest, True)

            pruned = column_weights.clone()
            pruned[mask[:, column]] = 0
            pruned_block[:, column] = pruned
            error = (column_weights - pruned) / diag
            weight_block[:, column:] -= error.unsqueeze(1).matmul(
                inverse_block[column, column:].unsqueeze(0)
            )
            error_block[:, column] = error

        weights[:, col_start:col_end] = pruned_block
        weights[:, col_end:] -= error_block.matmul(hessian_inverse[col_start:col_end, col_end:])

    def free(self) -> None:
        """Drop the Hessian and flush the CUDA cache when the layer is on GPU."""
        self.H = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
