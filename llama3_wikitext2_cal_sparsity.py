"""Llama 3 sweeps with WikiText2 calibration.

Same six Llama 3.1 / 3.2 repos as the C4 family. Prune on WikiText2, then
evaluate each saved checkpoint on C4 and PTB. Results stay under
figures/llama3_wikitext2_cal/ so figures/llama3/ is untouched.

The sweep is resumable: completed JSON points and matching WikiText2-
calibrated checkpoints are reused. Each point keeps the full repo id.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from experiments.llama import LLAMA3_WIKITEXT2_CAL
from qwen_figure1_family import ROOT

MODELS = list(LLAMA3_WIKITEXT2_CAL.models)
DATASETS = list(LLAMA3_WIKITEXT2_CAL.eval_datasets)

__all__ = [
    "DATASETS",
    "LLAMA3_WIKITEXT2_CAL",
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
    run_sweep(LLAMA3_WIKITEXT2_CAL, args.models, args.phase, sparsities=args.sparsities)


if __name__ == "__main__":
    main()
