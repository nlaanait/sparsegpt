"""Unstructured SparseGPT sweeps for Qwen3.6/3.8 with WikiText2 calibration.

Same models and sparsities as qwen_36_38_sparsity.py, but SparseGPT is
calibrated on WikiText2 and evaluated on C4 and PTB. Results and logs go to
results/qwen_wikitext2_cal/ so the C4-calibrated results/qwen/ tree is untouched.
"""

from __future__ import annotations

import argparse
import os

from analysis.sweep import run_sweep
from drivers.qwen_ptb import _prune_args
from experiments.qwen import QWEN_WIKITEXT2_CAL
from sparsifier.metrics import PPL_RE
from sparsifier.sweep import SPARSITIES, repo_root, sparsity_token

ROOT = repo_root()

MODELS = list(QWEN_WIKITEXT2_CAL.models)
DATASETS = list(QWEN_WIKITEXT2_CAL.eval_datasets)

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
    run_sweep(QWEN_WIKITEXT2_CAL, args.models, args.phase)


if __name__ == "__main__":
    main()
