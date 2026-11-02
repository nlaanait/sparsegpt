"""Pruners and the SparseGPT / geometry solvers they bind.

Solver classes live here. The Qwen, OLMo, Llama, and Gemma loops bind them.

SparseGPT's Hessian / Cholesky path needs full FP32 on Ampere+ GPUs, so
TF32 is disabled for the process when this package is imported.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from sparsifier.prune.geometry import GeometrySparseGPT
from sparsifier.prune.pruners import Geometry, Pruner, SparseGPT, bind_layer, get_pruner
from sparsifier.prune.sparsegpt import SparseGPT as SparseGPTSolver

__all__ = [
    "Geometry",
    "GeometrySparseGPT",
    "Pruner",
    "SparseGPT",
    "SparseGPTSolver",
    "bind_layer",
    "get_pruner",
]
