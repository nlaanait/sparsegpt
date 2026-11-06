"""Pruners and the SparseGPT / geometry solvers they bind.

Solver classes live here. The Qwen, OLMo, Llama, and Gemma loops bind them.

SparseGPT's Hessian / Cholesky path needs full FP32 on Ampere+ GPUs, so
TF32 is disabled for the process when this package is imported.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from sparsifier.prune.geometry import GeometrySparseGPT
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
    TangentKnockout,
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
    "Geometry",
    "GeometrySparseGPT",
    "Pruner",
    "SparseGPT",
    "TangentAware",
    "TangentAwareSparseGPT",
    "TangentChartObs",
    "TangentEditability",
    "TangentExactObs",
    "TangentFactorUnion",
    "TangentFixedUnion",
    "TangentKnockout",
    "Wanda",
    "SparseGPTSolver",
    "bind_layer",
    "estimate_output_tangent_coupling",
    "get_pruner",
    "list_pruners",
]
