"""Unstructured SparseGPT sweeps for Qwen3.6-27B and Qwen3.8-27B.

The driver lives in analysis.sweep. This module stays importable so a restart
of the live command keeps the same CLI, result paths, and log names.
"""

from __future__ import annotations

import argparse
import os

from qwen_figure1_family import PPL_RE, ROOT, sparsity_token
from qwen_ptb import _prune_args

from analysis.sweep import run_sweep
from experiments.qwen import QWEN_36_38

MODELS = list(QWEN_36_38.models)
SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
