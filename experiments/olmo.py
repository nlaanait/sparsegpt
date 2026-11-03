"""OLMo 2 instruction-tuned sparsity sweep.

Four Instruct sizes, C4 calibration (128 × 2048, seed 0), eval on WikiText2
and PTB. Dense WikiText2 must land in SMOKE_PPL_RANGE before any prune runs.
Stored N comes from ``sparsifier.models.catalog`` and is replaced once a checkpoint
is cached.
"""

from __future__ import annotations

from sparsifier.sweep import SweepRecord, SweepSpec, repo_root

MODELS = (
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
)

# Dense WikiText2 must sit in this band before the sparsity sweep starts.
# Larger OLMo 2 Instruct can land near ~4.5; hundreds or thousands fail.
SMOKE_DATASET = "wikitext2"
SMOKE_PPL_RANGE = (3.0, 50.0)

OLMO2 = SweepRecord.from_spec(
    SweepSpec(
        name="olmo2",
        models=MODELS,
        calibration="c4",
        eval_datasets=("wikitext2", "ptb"),
        log_infix="sdpa",
        attention="sdpa",
        holdout_suffix="32B-Instruct",
    ),
    repo_root(),
)

# WikiText2 calibration, C4 and PTB eval. The 32B file is absent until that
# ladder is collected; the loader skips a model with no points.
OLMO2_WIKITEXT2_CAL = SweepRecord.from_spec(
    SweepSpec(
        name="olmo2_wikitext2_cal",
        models=MODELS,
        calibration="wikitext2",
        eval_datasets=("c4", "ptb"),
        log_infix="sdpa",
        attention="sdpa",
        holdout_suffix="32B-Instruct",
    ),
    repo_root(),
)


def stored_params() -> dict[str, float]:
    """Catalog stored N, replaced once that checkpoint's index is cached."""
    return OLMO2.stored_params()
