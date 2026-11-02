"""Run one prune and score it.

`sparsegpt` is a pruner. Geometry is another. Perplexity is a metric.
Qwen, OLMo, Llama, and Gemma run from the Click command. OPT and BLOOM stay in the legacy package.
"""

import os

# The first CUDA allocation otherwise tries to reserve a large pool. On this
# unified-memory machine that reservation fails once a large checkpoint is loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sparsifier.models import ModelRecord, resolve, stored_count
from sparsifier.metrics import BitsPerByte, Likelihood, Perplexity, format_result
from sparsifier.prune import Geometry, SparseGPT, get_pruner
from sparsifier.sweep import CheckedSweep, RunConfig, SweepSpec, repo_root, validate

__all__ = [
    "BitsPerByte",
    "CheckedSweep",
    "Geometry",
    "Likelihood",
    "ModelRecord",
    "Perplexity",
    "RunConfig",
    "SparseGPT",
    "SweepSpec",
    "format_result",
    "get_pruner",
    "models",
    "repo_root",
    "resolve",
    "stored_count",
    "validate",
]
