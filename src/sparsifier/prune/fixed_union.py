"""Option 3 — fixed-union subspace rotation of ambient SparseGPT."""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix
from sparsifier.prune.tangent_common import (
    SampleReservoir,
    apply_bias_fold_from_samples,
    ensure_linear_or_conv1d,
)


class FixedUnionSparseGPT(SparseGPT):
    """Run ambient OBS in fixed tangent coordinates W_parallel = U^T W.

    After pruning ``W_parallel``, reconstruct
    ``U W_parallel + (I - UU^T) W_dense`` so the orthogonal complement stays
    at its dense value (note Option 3).
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        rank: int = 2,
        calibration_samples: int = 512,
        seed: int = 0,
        bias_fold: bool = False,
        union_basis: torch.Tensor | None = None,
    ) -> None:
        ensure_linear_or_conv1d(layer, "Fixed-union")
        if rank < 1 and union_basis is None:
            raise ValueError("rank must be positive when union_basis is omitted")
        if calibration_samples < 3 and union_basis is None:
            raise ValueError("calibration_samples must be >= 3 when estimating U")
        super().__init__(layer)
        self.rank = rank
        self.seed = seed
        self.bias_fold = bias_fold
        self.reservoir = SampleReservoir(max(calibration_samples, 3), seed)
        self._provided_basis = None if union_basis is None else union_basis.float().contiguous()
        self.union_basis: torch.Tensor | None = None
        self._parallel: torch.Tensor | None = None
        self._dense_weight: torch.Tensor | None = None
        self.tangent_diagnostics: dict | None = None
        self.finalized = False

    def parallel_weights(self) -> torch.Tensor:
        if self._parallel is None:
            raise RuntimeError("Call fasterprune before reading parallel weights")
        return self._parallel

    @torch.no_grad()
    def add_tangent_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        _ = blocksize
        if self.finalized and self._provided_basis is None:
            raise RuntimeError("Cannot add batches after fixed-union finalization")
        if self._provided_basis is None and not isinstance(out, torch.Tensor):
            raise TypeError("Fixed-union estimation requires tensor layer outputs")
        if isinstance(out, torch.Tensor):
            self.reservoir.update(inputs, out)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.finalized and self._provided_basis is None:
            raise RuntimeError("Cannot add batches after fixed-union finalization")
        if self._provided_basis is None and not isinstance(out, torch.Tensor):
            raise TypeError("Fixed-union estimation requires tensor layer outputs")
        super().add_batch(inputs, out, blocksize)
        # Retain samples for output-PCA (when needed) and for §3 bias folding.
        if isinstance(out, torch.Tensor):
            self.reservoir.update(inputs, out)

    @torch.no_grad()
    def finalize_union(self) -> None:
        if self.finalized:
            return
        output_dim = _weight_matrix(self.layer).shape[0]
        if self._provided_basis is not None:
            basis = self._provided_basis
            if basis.ndim != 2 or basis.shape[0] != output_dim:
                raise ValueError("union_basis must be [d_out, r] matching layer rows")
            q, _ = torch.linalg.qr(basis, mode="reduced")
            self.union_basis = q.contiguous()
        else:
            if self.reservoir.outputs is None:
                raise RuntimeError("Collect calibration outputs before fixed-union prune")
            y = self.reservoir.outputs.float()
            centered = y - y.mean(dim=0, keepdim=True)
            # PCA on the full calibration cloud → one fixed union frame.
            gram = centered @ centered.T
            eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            keep = min(self.rank, int((eigenvalues > 1e-12).sum().item()), output_dim)
            if keep < 1:
                raise ValueError("fixed-union PCA found zero variance")
            basis = centered.T @ eigenvectors[:, :keep]
            basis = basis / eigenvalues[:keep].clamp_min(1e-12).sqrt()[None, :]
            basis, _ = torch.linalg.qr(basis, mode="reduced")
            self.union_basis = basis.contiguous()
        self.tangent_diagnostics = {
            "sample_count": 0
            if self.reservoir.inputs is None
            else int(self.reservoir.inputs.shape[0]),
            "union_rank": int(self.union_basis.shape[1]),
        }
        self._dense_weight = _weight_matrix(self.layer).detach().cpu().float().clone()
        self.finalized = True

    @torch.no_grad()
    def fasterprune(self, sparsity: float, *args, **kwargs) -> None:
        self.finalize_union()
        assert self.union_basis is not None
        dense = _weight_matrix(self.layer).float()
        u = self.union_basis.to(device=dense.device, dtype=dense.dtype)
        parallel = u.T @ dense
        temp = nn.Linear(parallel.shape[1], parallel.shape[0], bias=False)
        temp.weight.data.copy_(parallel)
        temp.to(device=self.device)
        proxy = SparseGPT(temp)
        proxy.H = self.H
        proxy.nsamples = self.nsamples
        self.H = None
        proxy.fasterprune(sparsity, *args, **kwargs)
        self._parallel = temp.weight.data.detach().float().cpu()
        projector = u @ u.T
        rebuilt = u @ temp.weight.data.float().to(device=dense.device) + (
            dense - projector @ dense
        )
        if isinstance(self.layer, transformers.Conv1D):
            rebuilt = rebuilt.t()
        self.layer.weight.data = rebuilt.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if self.bias_fold:
            apply_bias_fold_from_samples(
                self.layer,
                dense_weight=self._dense_weight,
                sample_inputs=self.reservoir.inputs,
                sample_outputs=self.reservoir.outputs,
                union_basis=self.union_basis,
            )

    def free(self) -> None:
        self.reservoir.clear()
        self._parallel = None
        self._dense_weight = None
        self.union_basis = None
        super().free()
