"""Tangent-aware calibration on top of the local SparseGPT solver.

Probe columns are ambient input directions. Call ``add_batch`` for
activations, ``add_geometry`` once, then ``fasterprune``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT


class GeometrySparseGPT(SparseGPT):
    """Preserve E||(S-W)h||² + alpha E||(S-W)t||².

    E is expectation over calibration samples (and over probes for the
    tangent term). S is the sparse weight matrix after prune, W the dense
    layer weights, h a layer input activation, and t a tangent probe.
    """

    def __init__(self, layer: nn.Module) -> None:
        if not isinstance(layer, (nn.Linear, transformers.Conv1D)):
            raise TypeError("Geometry calibration supports Linear and Conv1D only")
        super().__init__(layer)
        self._check_hessian_device()
        self.n_activation_vectors = 0
        self.geometry_finalized = False

    def _check_hessian_device(self) -> None:
        if self.H.device != self.device:
            raise RuntimeError(
                f"Hessian is on {self.H.device}, expected layer device {self.device}"
            )

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        """Accumulate activations, then count flattened input vectors."""
        if self.geometry_finalized:
            raise RuntimeError("Collect all activations before adding geometry")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Activations must be a torch.Tensor")
        if (
            inputs.ndim not in (2, 3)
            or inputs.shape[-1] != self.columns
            or inputs.numel() == 0
        ):
            raise ValueError(
                "Expected nonempty [vectors, inputs] or [batch, sequence, inputs]"
            )
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
        super().add_batch(inputs, out, blocksize)
        self.n_activation_vectors += inputs.numel() // self.columns

    def _probe_chunk(self, tangents, start: int, chunk_size: int) -> torch.Tensor:
        end = min(start + chunk_size, tangents.shape[1])
        if isinstance(tangents, torch.Tensor):
            chunk = tangents[:, start:end]
            if chunk.device != self.device or chunk.dtype != self.H.dtype:
                return chunk.to(
                    device=self.device,
                    dtype=self.H.dtype,
                    non_blocking=self.device.type == "cuda",
                )
            return chunk.contiguous()
        return torch.as_tensor(
            np.asarray(tangents[:, start:end]),
            device=self.device,
            dtype=self.H.dtype,
        )

    @torch.no_grad()
    def add_geometry(
        self,
        tangents=None,
        alpha: float = 1.0,
        chunk_size: int = 1024,
    ) -> None:
        """Add ``alpha`` times the probe covariance to the activation Hessian.

        ``alpha == 0`` leaves the Hessian unchanged. Call this once, after
        every ``add_batch``.
        """
        if self.geometry_finalized:
            raise RuntimeError("Geometry may be finalized only once")
        if not math.isfinite(alpha) or alpha < 0 or chunk_size < 1:
            raise ValueError(
                "alpha must be finite/nonnegative; chunk_size must be positive"
            )
        if not self.n_activation_vectors:
            raise RuntimeError("Collect activation calibration first")
        if alpha == 0:
            self.geometry_finalized = True
            return
        if tangents is None:
            raise ValueError("Positive alpha requires tangent probes")
        if not isinstance(tangents, (torch.Tensor, np.ndarray)):
            raise TypeError("Probes must be a tensor or NumPy array")
        if (
            tangents.ndim != 2
            or tangents.shape[0] != self.columns
            or tangents.shape[1] == 0
        ):
            raise ValueError("Probes must have shape [input_dim, nonzero_probe_count]")
        # Activation Hessian uses the upstream scale 2 * sum(h h^T) / n_sequences.
        # Multiply the mean probe covariance by 2 * N / n_sequences so both
        # terms share that scale. N is the flattened activation count.
        scale = (
            2.0
            * self.n_activation_vectors
            / self.nsamples
            * alpha
            / tangents.shape[1]
        )
        for start in range(0, tangents.shape[1], chunk_size):
            probes = self._probe_chunk(tangents, start, chunk_size)
            if not torch.isfinite(probes).all():
                raise ValueError("Probes must be finite in the Hessian dtype")
            self.H.addmm_(probes, probes.t(), beta=1, alpha=scale)
        self.geometry_finalized = True
