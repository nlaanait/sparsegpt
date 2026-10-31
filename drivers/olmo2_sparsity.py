"""OLMo 2 Instruct family sparsity sweep.

Dense WikiText2 smoke for every size must clear before any SparseGPT prune.
Calibration is C4 (128 × 2048, seed 0). Eval is WikiText2 and PTB.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_smoke_dense, run_sweep
from experiments.olmo import MODELS, OLMO2, SMOKE_DATASET, SMOKE_PPL_RANGE
from drivers.qwen_figure1_family import ROOT

DATASETS = list(OLMO2.eval_datasets)

__all__ = [
    "DATASETS",
    "MODELS",
    "OLMO2",
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
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == "smoke":
        run_smoke_dense(
            OLMO2,
            args.models,
            dataset=SMOKE_DATASET,
            ppl_range=SMOKE_PPL_RANGE,
        )
        return
    if not args.skip_smoke:
        run_smoke_dense(
            OLMO2,
            args.models,
            dataset=SMOKE_DATASET,
            ppl_range=SMOKE_PPL_RANGE,
        )
    run_sweep(OLMO2, args.models, "all" if args.phase == "all" else args.phase)


if __name__ == "__main__":
    main()
