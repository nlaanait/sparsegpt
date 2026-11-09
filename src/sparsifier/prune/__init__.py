"""Pruners and the SparseGPT / geometry solvers they bind.

Solver classes live here. The Qwen, OLMo, Llama, and Gemma loops bind them.

SparseGPT's Hessian / Cholesky path needs full FP32 on Ampere+ GPUs, so
TF32 is disabled for the process when this package is imported.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from sparsifier.prune.geometry import GeometrySparseGPT
from sparsifier.prune.downstream import (
    ChartTangentBasis,
    DownstreamTangentCalibrator,
    DownstreamTangentSparseGPT,
    TangentAnchor,
    tangent_pullback_diagonal,
)
from sparsifier.prune.pruners import (
    Geometry,
    Pruner,
    SparseGPT,
    TangentAware,
    TangentChartObs,
    TangentExactObs,
    TangentFactorUnion,
    TangentEditability,
    TangentFixedUnion,
    TangentHeuristic,
    TangentInputSide,
    TangentKnockout,
    TangentRescale,
    Wanda,
    bind_layer,
    get_pruner,
    list_pruners,
)
from sparsifier.prune.sparsegpt import SparseGPT as SparseGPTSolver
from sparsifier.prune.tangent import (
    TangentAwareSparseGPT,
    estimate_output_tangent_coupling,
)

__all__ = [
    "ChartTangentBasis",
    "DownstreamTangentCalibrator",
    "DownstreamTangentSparseGPT",
    "Geometry",
    "GeometrySparseGPT",
    "Pruner",
    "SparseGPT",
    "TangentAware",
    "TangentAnchor",
    "TangentAwareSparseGPT",
    "TangentChartObs",
    "TangentEditability",
    "TangentExactObs",
    "TangentFactorUnion",
    "TangentFixedUnion",
    "TangentHeuristic",
    "TangentInputSide",
    "TangentKnockout",
    "TangentRescale",
    "Wanda",
    "SparseGPTSolver",
    "bind_layer",
    "estimate_output_tangent_coupling",
    "get_pruner",
    "list_pruners",
    "tangent_pullback_diagonal",
]
