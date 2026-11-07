"""Pruners bound to one linear layer.

SparseGPT observes calibration activations and prunes. Geometry adds an
input-probe term to the Hessian. Output-tangent techniques reweight or rotate
the SparseGPT solve; see ``list_pruners`` for the ablation registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

from sparsifier.prune.chart_obs import ChartObsSparseGPT
from sparsifier.prune.exact_ltan import ExactLtanSparseGPT
from sparsifier.prune.factor_union import FactorUnionSparseGPT
from sparsifier.prune.fixed_union import FixedUnionSparseGPT
from sparsifier.prune.geometry import GeometrySparseGPT
from sparsifier.prune.editability import EditabilitySparseGPT
from sparsifier.prune.input_side import InputSideSparseGPT
from sparsifier.prune.knockout import KnockoutSparseGPT
from sparsifier.prune.rescale import RescaleSparseGPT
from sparsifier.prune.revised_methods import FisherScaleSparseGPT, SparseLowRankSparseGPT
from sparsifier.prune.sparsegpt import SparseGPT as SparseGPTSolver
from sparsifier.prune.tangent import TangentAwareSparseGPT
from sparsifier.prune.wanda import WandaPruner


class BoundPruner(Protocol):
    """A solver already attached to one layer."""

    def add_batch(self, inputs, outputs) -> None: ...

    def finalize(self) -> None:
        """Finish calibration. Geometry loads probes here; SparseGPT is a no-op."""

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        """Write the pruned weights into the bound layer."""

    def free(self) -> None: ...


class Pruner(Protocol):
    """Factory that binds a solver to one linear."""

    name: str

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> BoundPruner: ...


@dataclass
class _SparseGPTBound:
    solver: SparseGPTSolver

    def add_batch(self, inputs, outputs) -> None:
        self.solver.add_batch(inputs, outputs)

    def finalize(self) -> None:
        return None

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.solver.fasterprune(
            sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
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

    def add_batch(self, inputs, outputs) -> None:
        self.solver.add_batch(inputs, outputs)

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
        probes = np.load(path, allow_pickle=False, mmap_mode="c")
        self.solver.add_geometry(probes, alpha=self.alpha)
        self._finalized = True

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.finalize()
        self.solver.fasterprune(
            sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass
class _SolverBound:
    """Generic bound wrapper for output-tangent technique solvers."""

    solver: Any
    block_index: int
    module_name: str
    label: str
    finalize_attr: str
    bias_fold: bool = False
    _finalized: bool = False

    def add_batch(self, inputs, outputs) -> None:
        self.solver.add_batch(inputs, outputs)

    def finalize(self) -> None:
        if self._finalized:
            return
        getattr(self.solver, self.finalize_attr)()
        diagnostics = getattr(self.solver, "tangent_diagnostics", None) or {}
        sample_count = diagnostics.get("sample_count")
        if sample_count is None:
            reservoir = getattr(self.solver, "reservoir", None)
            inputs = getattr(reservoir, "inputs", None)
            sample_count = 0 if inputs is None else int(inputs.shape[0])
        print(
            f"{self.label} calibration "
            f"{self.block_index}/{self.module_name}: samples={sample_count}",
            flush=True,
        )
        self._finalized = True

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.finalize()
        self.solver.fasterprune(
            sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass
class _TangentBound:
    solver: TangentAwareSparseGPT
    block_index: int
    module_name: str
    bias_fold: bool = False
    _finalized: bool = False

    def add_batch(self, inputs, outputs) -> None:
        self.solver.add_batch(inputs, outputs)

    def finalize(self) -> None:
        if self._finalized:
            return
        self.solver.finalize_tangents()
        diagnostics = self.solver.tangent_diagnostics or {}
        print(
            "Tangent calibration "
            f"{self.block_index}/{self.module_name}: "
            f"samples={diagnostics.get('sample_count')} "
            f"charts={diagnostics.get('chart_count')} "
            f"rank={diagnostics.get('chart_ranks')} "
            f"inactive_inputs={diagnostics.get('inactive_input_features')} "
            f"mean_coupling={diagnostics.get('mean_tangent_coupling', float('nan')):.4f} "
            f"captured_var={diagnostics.get('mean_chart_variance_captured', float('nan')):.4f}",
            flush=True,
        )
        self._finalized = True

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.finalize()
        self.solver.fasterprune(
            sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass(frozen=True)
class SparseGPT:
    """Activation Hessian only."""

    name: str = "sparsegpt"

    def bind(
        self,
        layer: nn.Module,
        *,
        block_index: int,
        module_name: str,
    ) -> _SparseGPTBound:
        _ = (block_index, module_name)
        return _SparseGPTBound(SparseGPTSolver(layer))


@dataclass(frozen=True)
class TangentAware:
    """Option 1: k-means chart G_pq mask weighting + ambient compensation."""

    alpha: float = 1.0
    charts: int = 8
    rank: int = 2
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    anchor_eta: float = 0.0
    name: str = "tangent"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("tangent alpha must be finite and in (0, 1]")
        if not math.isfinite(self.anchor_eta) or self.anchor_eta < 0.0:
            raise ValueError("anchor_eta must be finite and nonnegative")
        if self.charts < 1 or self.rank < 1 or self.calibration_samples < 3:
            raise ValueError("tangent charts/rank must be positive and samples >= 3")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _TangentBound:
        _ = (block_index, module_name)
        return _TangentBound(
            TangentAwareSparseGPT(
                layer,
                alpha=self.alpha,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
                anchor_eta=self.anchor_eta,
            ),
            block_index,
            module_name,
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentFactorUnion:
    """Option 1 with concept-factor union projectors."""

    alpha: float = 1.0
    factors: int = 4
    rank: int = 1
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    active_factors: int = 2
    factor_source: str = "kmeans"
    block_width: int = 3
    name: str = "tangent_factor_union"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            FactorUnionSparseGPT(
                layer,
                alpha=self.alpha,
                factors=self.factors,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
                active_factors=self.active_factors,
                factor_source=self.factor_source,
                block_width=self.block_width,
            ),
            block_index,
            module_name,
            label="Factor-union",
            finalize_attr="finalize_tangents",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentChartObs:
    """Option 2: per-chart OBS mask scores + ambient compensation."""

    charts: int = 8
    rank: int = 1
    calibration_samples: int = 512
    seed: int = 0
    percdamp: float = 0.01
    bias_fold: bool = False
    name: str = "tangent_chart_obs"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            ChartObsSparseGPT(
                layer,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                percdamp=self.percdamp,
                bias_fold=self.bias_fold,
            ),
            block_index,
            module_name,
            label="Chart-OBS",
            finalize_attr="finalize_charts",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentFixedUnion:
    """Option 3: fixed-union subspace rotation."""

    rank: int = 2
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    name: str = "tangent_fixed_union"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            FixedUnionSparseGPT(
                layer,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
            ),
            block_index,
            module_name,
            label="Fixed-union",
            finalize_attr="finalize_union",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentKnockout:
    """§1 union-tangent knockout impact mask ranking."""

    charts: int = 8
    rank: int = 1
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    name: str = "tangent_knockout"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            KnockoutSparseGPT(
                layer,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
            ),
            block_index,
            module_name,
            label="Knockout",
            finalize_attr="finalize_scores",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentEditability:
    """§1 union-tangent editability mask ranking (no |W| factor)."""

    charts: int = 8
    rank: int = 1
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    name: str = "tangent_editability"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            EditabilitySparseGPT(
                layer,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
            ),
            block_index,
            module_name,
            label="Editability",
            finalize_attr="finalize_scores",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class Geometry:
    """Option 4: activation Hessian plus input-probe covariance."""

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


@dataclass(frozen=True)
class Wanda:
    """Ambient Wanda baseline: |W|·RMS(φ) mask, no OBS compensation."""

    calibration_samples: int = 512
    seed: int = 0
    name: str = "wanda"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            WandaPruner(
                layer,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
            ),
            block_index,
            module_name,
            label="Wanda",
            finalize_attr="finalize_scores",
        )


@dataclass
class _InputSideBound:
    solver: InputSideSparseGPT
    block_index: int
    module_name: str
    alpha_in: float
    pairs: Path | None
    _finalized: bool = False

    def add_batch(self, inputs, outputs) -> None:
        self.solver.add_batch(inputs, outputs)

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.alpha_in > 0.0:
            if self.pairs is None:
                raise ValueError("tangent_input_side requires a pairs directory")
            path = self.pairs / str(self.block_index) / f"{self.module_name}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing input-side pairs for layer {self.block_index}/{self.module_name}: {path}"
                )
            import numpy as np

            from sparsifier.data.pair_deltas import load_knob_ids

            deltas = torch.as_tensor(np.load(path, allow_pickle=False))
            self.solver.add_pairs(deltas, knob_ids=load_knob_ids(path))
        self.solver.finalize_pairs(self.alpha_in)
        print(
            "Input-side calibration "
            f"{self.block_index}/{self.module_name}: alpha_in={self.alpha_in}",
            flush=True,
        )
        self._finalized = True

    def prune(
        self,
        *,
        sparsity: float,
        zeros_per_group: int = 0,
        group_size: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        self.finalize()
        self.solver.fasterprune(
            sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
            blocksize=blocksize,
            percdamp=percdamp,
        )

    def free(self) -> None:
        self.solver.free()


@dataclass(frozen=True)
class TangentInputSide:
    """Variant A: shared Hessian plus paired-input difference curvature."""

    alpha_in: float = 1.0
    pairs: Path | None = None
    name: str = "tangent_input_side"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_in) or self.alpha_in < 0.0:
            raise ValueError("alpha_in must be finite and nonnegative")
        if self.alpha_in > 0.0 and self.pairs is None:
            raise ValueError("alpha_in > 0 requires a pairs directory")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _InputSideBound:
        return _InputSideBound(
            InputSideSparseGPT(layer),
            block_index,
            module_name,
            alpha_in=self.alpha_in,
            pairs=self.pairs,
        )


@dataclass(frozen=True)
class TangentRescale:
    """Variant B: rescaled SparseGPT with M_n = I + γ P_geo (unbounded α)."""

    alpha: float = 1.0
    charts: int = 8
    rank: int = 2
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    anchor_eta: float = 0.0
    name: str = "tangent_rescale"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("rescale alpha must be finite and nonnegative")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            RescaleSparseGPT(
                layer,
                alpha=self.alpha,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
                heuristic=False,
                anchor_eta=self.anchor_eta,
            ),
            block_index,
            module_name,
            label="Tangent-rescale",
            finalize_attr="finalize_tangents",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentHeuristic:
    """Score-only 1+γG ablation: no matching rescaled compensation."""

    alpha: float = 1.0
    charts: int = 8
    rank: int = 2
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    anchor_eta: float = 0.0
    name: str = "tangent_heuristic"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("heuristic alpha must be finite and nonnegative")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            RescaleSparseGPT(
                layer,
                alpha=self.alpha,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
                heuristic=True,
                anchor_eta=self.anchor_eta,
            ),
            block_index,
            module_name,
            label="Tangent-heuristic",
            finalize_attr="finalize_tangents",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentFisher:
    """Fisher rung of the output-metric ladder, through Variant B rescaling."""

    alpha: float = 1.0
    name: str = "tangent_fisher"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("fisher alpha must be finite and nonnegative")

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            FisherScaleSparseGPT(layer, alpha=self.alpha),
            block_index,
            module_name,
            label="Fisher",
            finalize_attr="finalize_fisher",
            bias_fold=False,
        )


@dataclass(frozen=True)
class TangentSparseLowRank:
    """Variant D: ambient sparse weights plus a dense concept-block correction."""

    rank: int = 2
    calibration_samples: int = 512
    seed: int = 0
    bias_fold: bool = False
    name: str = "tangent_sparse_lowrank"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            SparseLowRankSparseGPT(
                layer,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                bias_fold=self.bias_fold,
            ),
            block_index,
            module_name,
            label="Sparse-low-rank",
            finalize_attr="finalize_union",
            bias_fold=self.bias_fold,
        )


@dataclass(frozen=True)
class TangentExactObs:
    """Exact ``L_tan`` OBS: diag-P row Hessians or small full Kronecker."""

    charts: int = 8
    rank: int = 1
    calibration_samples: int = 512
    seed: int = 0
    percdamp: float = 0.01
    bias_fold: bool = False
    anchor_eta: float = 0.0
    mode: str = "diag"
    max_full_params: int = 4096
    name: str = "tangent_exact_obs"

    def bind(self, layer: nn.Module, *, block_index: int, module_name: str) -> _SolverBound:
        return _SolverBound(
            ExactLtanSparseGPT(
                layer,
                charts=self.charts,
                rank=self.rank,
                calibration_samples=self.calibration_samples,
                seed=self.seed,
                percdamp=self.percdamp,
                bias_fold=self.bias_fold,
                anchor_eta=self.anchor_eta,
                mode=self.mode,
                max_full_params=self.max_full_params,
            ),
            block_index,
            module_name,
            label="Exact-Ltan",
            finalize_attr="finalize_charts",
            bias_fold=self.bias_fold,
        )


_PRUNER_TYPES: dict[str, type] = {
    SparseGPT.name: SparseGPT,
    Wanda.name: Wanda,
    Geometry.name: Geometry,
    TangentAware.name: TangentAware,
    TangentFactorUnion.name: TangentFactorUnion,
    TangentChartObs.name: TangentChartObs,
    TangentFixedUnion.name: TangentFixedUnion,
    TangentKnockout.name: TangentKnockout,
    TangentEditability.name: TangentEditability,
    TangentExactObs.name: TangentExactObs,
    TangentInputSide.name: TangentInputSide,
    TangentRescale.name: TangentRescale,
    TangentHeuristic.name: TangentHeuristic,
    TangentFisher.name: TangentFisher,
    TangentSparseLowRank.name: TangentSparseLowRank,
}


def list_pruners() -> list[str]:
    """Stable ablation registry names."""
    return list(_PRUNER_TYPES)


def _technique_names() -> list[str]:
    return list(_PRUNER_TYPES)


# Alias used by unit tests and external ablation drivers.
TECHNIQUE_NAMES = _technique_names()


def bind_layer(
    layer: nn.Module,
    *,
    block_index: int,
    module_name: str,
    geometry_alpha: float = 0.0,
    geometry_probes: Path | None = None,
    tangent_alpha: float = 0.0,
    tangent_charts: int = 8,
    tangent_rank: int = 2,
    tangent_samples: int = 512,
    tangent_seed: int = 0,
) -> BoundPruner:
    """Bind ambient, input-geometry, or output-tangent pruning."""
    if not math.isfinite(tangent_alpha) or not 0.0 <= tangent_alpha <= 1.0:
        raise ValueError("tangent alpha must be finite and in [0, 1]")
    if tangent_alpha > 0 and geometry_alpha > 0:
        raise ValueError("input geometry and output tangent modes are mutually exclusive")
    if tangent_alpha > 0:
        pruner = get_pruner(
            "tangent",
            alpha=tangent_alpha,
            charts=tangent_charts,
            rank=tangent_rank,
            calibration_samples=tangent_samples,
            seed=tangent_seed,
        )
    elif geometry_alpha > 0:
        pruner = get_pruner("geometry", alpha=geometry_alpha, probes=geometry_probes)
    else:
        pruner = get_pruner("sparsegpt")
    return pruner.bind(layer, block_index=block_index, module_name=module_name)


def get_pruner(name: str, **options) -> Pruner:
    """Look up a pruner by ablation registry name."""
    if name not in _PRUNER_TYPES:
        raise KeyError(
            f"unknown pruner {name!r}; known: {', '.join(list_pruners())}"
        )
    cls = _PRUNER_TYPES[name]
    if name == SparseGPT.name and options:
        raise TypeError(f"sparsegpt pruner takes no options, got {sorted(options)}")
    return cls(**options)
