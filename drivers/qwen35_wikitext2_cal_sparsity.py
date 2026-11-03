"""Unstructured Qwen3.5 family sweeps with WikiText2 calibration.

Prune Qwen3.5 0.8B, 2B, 4B, 9B, and 27B at sparsities 0.1 through 0.8,
then evaluate each saved checkpoint on C4 and PTB. Results and logs are
isolated under results/qwen3.5_wikitext2_cal/.

The sweep is resumable: completed JSON points and matching WikiText2-
calibrated checkpoints are reused.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from experiments.qwen import QWEN_35_WIKITEXT2_CAL
from sparsifier.metrics import PPL_RE
from sparsifier.sweep import SPARSITIES, repo_root, sparsity_token
from drivers.qwen_ptb import _prune_args

ROOT = repo_root()

MODELS = list(QWEN_35_WIKITEXT2_CAL.models)
DATASETS = list(QWEN_35_WIKITEXT2_CAL.eval_datasets)

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
    parser.add_argument(
        "--phase",
        choices=[*DATASETS, "all"],
        default="all",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    run_sweep(QWEN_35_WIKITEXT2_CAL, args.models, args.phase)


if __name__ == "__main__":
    main()
