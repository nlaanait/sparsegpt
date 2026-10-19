"""Tangent-aware calibration using the unchanged SparseGPT pruning solver.

Works on CPU and CUDA: Hessian accumulation and pruning follow the layer
device (`self.dev`). Probe tensors are transferred in chunks to that device.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


class GeometrySparseGPT(SparseGPT):
    """Preserve E||(S-W)h||² + alpha E||(S-W)t||².

    Use add_batch as usual, add_geometry once after all calibration, then
    fasterprune. Probe columns are ambient input directions, not activations
    to propagate through the network. Their scale is part of the objective.
    """
    def __init__(self, layer):
        if not isinstance(layer, (nn.Linear, transformers.Conv1D)):
            raise TypeError('Geometry calibration supports Linear and Conv1D only')
        super().__init__(layer)
        self.nvectors = 0
        self.geometry_finalized = False

    @torch.no_grad()
    def add_batch(self, inp, out, blocksize=1024):
        if self.geometry_finalized:
            raise RuntimeError('Collect all activations before adding geometry')
        if not isinstance(inp, torch.Tensor):
            raise TypeError('Activations must be a torch.Tensor')
        if inp.ndim not in (2, 3) or inp.shape[-1] != self.columns or inp.numel() == 0:
            raise ValueError('Expected nonempty [vectors, inputs] or [batch, sequence, inputs]')
        # Step 1: retain upstream accumulation exactly, counting vectors separately.
        # Efficient: existing streaming covariance stores O(d²), not O(N*d).
        # Better calibration: sample rare factors and use upstream-pruned inputs.
        # Device: keep activations on the layer device before add_batch when possible;
        # SparseGPT matmuls run on whatever device `inp` / `self.H` share.
        if inp.device != self.dev:
            inp = inp.to(self.dev, non_blocking=self.dev.type == 'cuda')
        super().add_batch(inp, out, blocksize)
        self.nvectors += inp.numel() // self.columns

    def _probe_chunk(self, tangents, start, chunk_size):
        """Materialize one probe chunk on `self.dev` in the Hessian dtype."""
        end = min(start + chunk_size, tangents.shape[1])
        if isinstance(tangents, torch.Tensor):
            chunk = tangents[:, start:end]
            # Already-correct CUDA/CPU tensors skip a host round-trip.
            if chunk.device != self.dev or chunk.dtype != self.H.dtype:
                chunk = chunk.to(device=self.dev, dtype=self.H.dtype,
                                 non_blocking=self.dev.type == 'cuda')
            else:
                chunk = chunk.contiguous()
        else:
            # NumPy / memmap: one host->device copy per chunk.
            chunk = torch.as_tensor(
                np.asarray(tangents[:, start:end]),
                device=self.dev,
                dtype=self.H.dtype,
            )
        return chunk

    @torch.no_grad()
    def add_geometry(self, tangents=None, alpha=1.0, chunk_size=1024):
        if self.geometry_finalized:
            raise RuntimeError('Geometry may be finalized only once')
        if not math.isfinite(alpha) or alpha < 0 or chunk_size < 1:
            raise ValueError('alpha must be finite/nonnegative; chunk_size must be positive')
        if not self.nvectors:
            raise RuntimeError('Collect activation calibration first')
        if alpha == 0:
            # Exact baseline: no multiply/divide or rounding of the original H.
            self.geometry_finalized = True
            return
        if tangents is None:
            raise ValueError('Positive alpha requires tangent probes')
        if not isinstance(tangents, (torch.Tensor, np.ndarray)):
            raise TypeError('Probes must be a tensor or NumPy array')
        if tangents.ndim != 2 or tangents.shape[0] != self.columns or tangents.shape[1] == 0:
            raise ValueError('Probes must have shape [input_dim, nonzero_probe_count]')
        # Step 4: add covariance without concatenating calibration matrices.
        # Upstream H = 2*sum(h h.T)/nsequences. Multiply the desired mean probe
        # covariance by 2*N/nsequences so both terms share that overall scale:
        # H_new = (2*N/nsequences) * (H_data H_data.T/N + alpha*T T.T/P).
        # This also preserves relative diagonal damping. Feeding raw probes via
        # add_batch would incorrectly mix probe/sequence counts and lose alpha.
        factor = 2.0 * self.nvectors / self.nsamples * alpha / tangents.shape[1]
        # Single-pass chunk transfer: validate finiteness then accumulate.
        # Efficient: no second d*d covariance; CUDA uses non_blocking host copies
        # when the source is pinned. At very large d, use covariance blocks or
        # low-rank Hessian approximations instead of dense H.
        if self.H.device != self.dev:
            self.H = self.H.to(self.dev)
        for start in range(0, tangents.shape[1], chunk_size):
            t = self._probe_chunk(tangents, start, chunk_size)
            if not torch.isfinite(t).all():
                raise ValueError('Probes must be finite in the Hessian dtype')
            self.H.addmm_(t, t.t(), beta=1, alpha=factor)
        self.geometry_finalized = True
        # Step 5: call inherited fasterprune: same masks/compensation algorithm.
        # Better optimization: support swaps/regrowth and held-out alpha tuning;
        # these are deliberately excluded to isolate the geometry contribution.


def add_geometry_arguments(parser):
    parser.add_argument('--geometry-alpha', type=float, default=0.0,
                        help='Tangent objective weight; 0 preserves upstream calibration')
    parser.add_argument('--geometry-probes', type=Path,
                        help='Directory of <block_index>/<module_name>.npy input tangent matrices')


def apply_geometry(pruner, args, block_index, module_name):
    """Load precomputed layer-specific probes after activation hook removal."""
    alpha = args.geometry_alpha
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('--geometry-alpha must be finite and nonnegative')
    if alpha == 0:
        return
    if args.geometry_probes is None:
        raise ValueError('--geometry-probes is required with positive --geometry-alpha')
    path = args.geometry_probes / str(block_index) / (module_name + '.npy')
    if not path.is_file():
        raise FileNotFoundError(f'Missing geometry probes for layer {block_index}/{module_name}: {path}')
    # Step 3/4 interface: .npy avoids pickle and supports memory-mapped reading.
    # Improve throughput by prefetching next-layer probes while pruning this one.
    probes = np.load(path, allow_pickle=False, mmap_mode='c')
    # Copy-on-write mapping exposes writable views without changing probe files.
    pruner.add_geometry(probes, alpha=alpha)
