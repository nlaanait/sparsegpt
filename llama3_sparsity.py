"""Llama 3.1 / 3.2 family sparsity sweep.

One family: 1B, 3B, 8B, 11B Vision, 70B, and 90B Vision. Each result file
records the full repo id. Dense WikiText2 smoke for the requested model must
clear before any SparseGPT prune. Calibration is C4 (128 × 2048, seed 0).
Eval is WikiText2 and PTB.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_smoke_dense, run_sweep
from experiments.llama import LLAMA3, MODELS, SMOKE_DATASET, SMOKE_PPL_RANGE
from qwen_figure1_family import ROOT

DATASETS = list(LLAMA3.eval_datasets)

__all__ = [
    "DATASETS",
    "LLAMA3",
    "MODELS",
    "ROOT",
    "SMOKE_DATASET",
    "SMOKE_PPL_RANGE",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument(
        "--phase",
        choices=[*DATASETS, "all", "smoke"],
        default="all",
        help="smoke: dense WikiText2 only. all: smoke then full sparsity sweep.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the dense WikiText2 gate (only when resuming a cleared smoke).",
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=None,
        help="Sparsity levels to prune. Default is 0.1 through 0.8.",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == "smoke":
        run_smoke_dense(LLAMA3, args.models, dataset=SMOKE_DATASET, ppl_range=SMOKE_PPL_RANGE)
        return
    if not args.skip_smoke:
        run_smoke_dense(LLAMA3, args.models, dataset=SMOKE_DATASET, ppl_range=SMOKE_PPL_RANGE)
    run_sweep(
        LLAMA3,
        args.models,
        "all" if args.phase == "all" else args.phase,
        sparsities=args.sparsities,
    )


if __name__ == "__main__":
    main()
