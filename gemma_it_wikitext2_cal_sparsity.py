"""Gemma 4 instruction-tuned sweeps with WikiText2 calibration.

The three repos on the Gemma row of figures/universal_scaling_law.html:
E4B-it, 12B-it, and 31B-it. Prune on WikiText2, then evaluate each saved
checkpoint on C4 and PTB. Results stay under figures/gemma_it_wikitext2_cal/
so figures/gemma/ is untouched.

The sweep is resumable: completed JSON points and matching WikiText2-
calibrated checkpoints are reused.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from experiments.gemma import GEMMA_IT_WIKITEXT2_CAL
from qwen_figure1_family import ROOT

MODELS = list(GEMMA_IT_WIKITEXT2_CAL.models)
DATASETS = list(GEMMA_IT_WIKITEXT2_CAL.eval_datasets)

__all__ = [
    "DATASETS",
    "GEMMA_IT_WIKITEXT2_CAL",
    "MODELS",
    "ROOT",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--phase",
        choices=[*DATASETS, "all"],
        default="all",
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
    run_sweep(GEMMA_IT_WIKITEXT2_CAL, args.models, args.phase, sparsities=args.sparsities)


if __name__ == "__main__":
    main()
