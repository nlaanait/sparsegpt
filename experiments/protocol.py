"""Repo root, the sweep spec, perplexity-log parse, and the prune namespace.

Sweep and gate code imports these from here. Figure scripts stay free to
re-export them, but analysis does not import a figure driver to run a sweep.
"""

from __future__ import annotations

import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepSpec:
    name: str
    models: tuple[str, ...]
    runner: str
    extra_args: tuple[str, ...]
    result_dir: Path
    log_dir: Path
    log_infix: str
    attention: str
    holdout_suffix: str
    plot: bool = True
    # SparseGPT calibration corpus (positional dataset arg to the runner).
    calibration: str = "c4"
    # Perplexity corpora after prune (or dense). Order is the sweep order.
    eval_datasets: tuple[str, ...] = ("ptb", "wikitext2")


PPL_RE = re.compile(
    r"Perplexity:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?|inf|nan)",
    re.I,
)


def prune_args(model: str, sparsity: float, dataset: str = "c4") -> Namespace:
    return Namespace(
        model=model,
        dataset=dataset,
        seed=0,
        nsamples=128,
        percdamp=0.01,
        sparsity=sparsity,
        prunen=0,
        prunem=0,
        blocksize=128,
        gmp=False,
        wbits=16,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        true_sequential=False,
        geometry_alpha=0.0,
        geometry_probes=None,
    )
