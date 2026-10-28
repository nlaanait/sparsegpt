"""OLMo 2 Instruct sweeps with WikiText2 calibration.

Prune the four Instruct sizes at sparsities 0.1 through 0.8, then evaluate
each saved checkpoint on C4 and PTB. Results and logs stay under
figures/olmo2_wikitext2_cal/ so the C4-calibrated figures/olmo2/ tree is
untouched.

The sweep is resumable: completed JSON points and matching WikiText2-
calibrated checkpoints are reused.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from experiments.olmo import OLMO2_WIKITEXT2_CAL
from qwen_figure1_family import ROOT

MODELS = list(OLMO2_WIKITEXT2_CAL.models)
DATASETS = list(OLMO2_WIKITEXT2_CAL.eval_datasets)

__all__ = [
    "DATASETS",
    "MODELS",
    "OLMO2_WIKITEXT2_CAL",
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
    run_sweep(OLMO2_WIKITEXT2_CAL, args.models, args.phase, sparsities=args.sparsities)


if __name__ == "__main__":
    main()
