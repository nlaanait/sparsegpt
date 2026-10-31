"""Pruners bound to one linear layer.

SparseGPT observes calibration activations and prunes. Geometry does that
and then adds a tangent term to the Hessian before the same solve. The
solver classes still live in the ``sparsegpt`` package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch.nn as nn

from sparsegpt.geometry import GeometrySparseGPT
from sparsegpt.sparsegpt import SparseGPT as SparseGPTSolver


class BoundPruner(Protocol):
    def add_batch(self, inp, out) -> None: ...

    def finalize(self) -> None: ...

    def prune(
        self,
        *,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None: ...

    def free(self) -> None: ...


class Pruner(Protocol):
    name: str

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> BoundPruner: ...


@dataclass
class _SparseGPTBound:
    solver: SparseGPTSolver

    def add_batch(self, inp, out) -> None:
        self.solver.add_batch(inp, out)

    def finalize(self) -> None:
        return None

    def prune(
        self,
        *,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.solver.fasterprune(
            sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass
class _GeometryBound:
    solver: GeometrySparseGPT
    alpha: float
    probes: Path | None
    block_index: int
    module_name: str
    _finalized: bool = False

    def add_batch(self, inp, out) -> None:
        self.solver.add_batch(inp, out)

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.alpha == 0:
            self._finalized = True
            return
        if self.probes is None:
            raise ValueError("geometry alpha > 0 requires a probes directory")
        path = self.probes / str(self.block_index) / f"{self.module_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing geometry probes for layer {self.block_index}/{self.module_name}: {path}"
            )
        loaded = np.load(path, allow_pickle=False, mmap_mode="c")
        self.solver.add_geometry(loaded, alpha=self.alpha)
        self._finalized = True

    def prune(
        self,
        *,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.finalize()
        self.solver.fasterprune(
            sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass(frozen=True)
class SparseGPT:
    """Activation Hessian only."""

    name: str = "sparsegpt"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SparseGPTBound:
        del block_index, module_name
        return _SparseGPTBound(SparseGPTSolver(layer))


@dataclass(frozen=True)
class Geometry:
    """Activation Hessian plus a tangent penalty."""

    alpha: float = 0.0
    probes: Path | None = None
    name: str = "geometry"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("geometry alpha must be finite and nonnegative")
        if self.alpha > 0 and self.probes is None:
            raise ValueError("geometry alpha > 0 requires a probes directory")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _GeometryBound:
        return _GeometryBound(
            GeometrySparseGPT(layer),
            self.alpha,
            self.probes,
            block_index,
            module_name,
        )


def get_pruner(name: str, **options) -> Pruner:
    if name == SparseGPT.name:
        if options:
            raise TypeError(f"sparsegpt pruner takes no options, got {sorted(options)}")
        return SparseGPT()
    if name == Geometry.name:
        return Geometry(**options)
    raise KeyError(f"unknown pruner {name!r}; known: geometry, sparsegpt")
