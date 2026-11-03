"""Unstructured SparseGPT sweeps for Qwen3.6-27B and Qwen3.8-27B.

The driver lives in analysis.sweep. This module stays importable so a restart
of the live command keeps the same CLI, result paths, and log names.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from drivers.qwen_ptb import _prune_args
from experiments.qwen import QWEN_36_38
from sparsifier.metrics import PPL_RE
from sparsifier.sweep import SPARSITIES, repo_root, sparsity_token

ROOT = repo_root()

MODELS = list(QWEN_36_38.models)
DATASETS = ["ptb", "wikitext2"]

__all__ = [
    "DATASETS",
    "MODELS",
    "PPL_RE",
    "ROOT",
    "SPARSITIES",
    "_prune_args",
    "sparsity_token",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--phase", choices=["ptb", "wikitext2", "all"], default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    run_sweep(QWEN_36_38, args.models, args.phase)


if __name__ == "__main__":
    main()
