"""Gemma 4 base sweep.

The instruction-tuned -it repos are not part of this sweep. Stored N comes
from ``sparsifier.models.catalog`` and is replaced once a checkpoint is cached.
Effective N is Google's count and is never used as the fit's N. The 26B-A4B
entry is the MoE text tower (128 experts, top-8); N is still total stored
parameters, not active parameters.
"""

from __future__ import annotations

from sparsifier.sweep import SweepRecord, SweepSpec, repo_root

# Confirmed against the google/gemma-4 collection. The 12B checkpoint is the
# unified text tower. 26B-A4B is the mixture-of-experts text tower.
MODELS = (
    "google/gemma-4-E2B",
    "google/gemma-4-E4B",
    "google/gemma-4-12B",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-31B",
)

# Google's effective counts, in billions. Not used as N in the law.
# A4B = ~4B active parameters at inference.
EFFECTIVE_B = {
    "google/gemma-4-E2B": 2.3,
    "google/gemma-4-E4B": 4.5,
    "google/gemma-4-12B": 11.95,
    "google/gemma-4-26B-A4B": 4.0,
    "google/gemma-4-31B": 30.7,
}

GEMMA = SweepRecord.from_spec(
    SweepSpec(
        name="gemma",
        models=MODELS,
        calibration="c4",
        eval_datasets=("ptb", "wikitext2"),
        log_infix="sdpa",
        attention="sdpa",
        holdout_suffix="31B",
    ),
    repo_root(),
)


def effective_params() -> dict[str, float]:
    return dict(EFFECTIVE_B)


def stored_params() -> dict[str, float]:
    """Catalog stored N, replaced once that checkpoint's index is cached."""
    return GEMMA.stored_params()
