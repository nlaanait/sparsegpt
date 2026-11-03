"""Qwen family spec.

The instruction-tuned Qwen checkpoints used for the existing sweeps do not
put "it" in the repo name. The 3.6/3.8 driver is drivers.qwen_36_38_sparsity. Its results live under
results/qwen/ now that the 3.8 job has exited.
"""

from __future__ import annotations

from sparsifier.sweep import SweepRecord, SweepSpec, repo_root

QWEN_36_38 = SweepRecord.from_spec(
    SweepSpec(
        name="qwen",
        models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
        calibration="c4",
        eval_datasets=("ptb", "wikitext2"),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="27B",
        plot=False,
    ),
    repo_root(),
)

# Same 3.6/3.8 series with calibration and eval swapped: WikiText2 calibrate,
# C4 and PTB evaluate. Kept under its own tree so results/qwen/ stays the
# C4-calibrated baseline.
QWEN_WIKITEXT2_CAL = SweepRecord.from_spec(
    SweepSpec(
        name="qwen_wikitext2_cal",
        models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
        calibration="wikitext2",
        eval_datasets=("c4", "ptb"),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="27B",
        plot=True,
    ),
    repo_root(),
)

# C4-calibrated 3.5 ladders live in the combined family JSON, not per-model
# files. 3.6/3.8 are drawn on the same pages from results/qwen/. This spec is
# for figures only; the sweep still runs the per-model WikiText2-calibrated job.
QWEN_35 = SweepRecord.from_spec(
    SweepSpec(
        name="qwen3.5",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-27B",
        ),
        calibration="c4",
        eval_datasets=("wikitext2", "ptb"),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="27B",
        plot=False,
    ),
    repo_root(),
)

# Qwen3.5 family counterpart to QWEN_WIKITEXT2_CAL. Keep its artifacts
# separate from both the historical C4-calibrated family sweep and the live
# 3.6/3.8 sweep. Stored N is per model, so the shared law is identified.
QWEN_35_WIKITEXT2_CAL = SweepRecord.from_spec(
    SweepSpec(
        name="qwen3.5_wikitext2_cal",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-27B",
        ),
        calibration="wikitext2",
        eval_datasets=("c4", "ptb"),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="27B",
        plot=True,
    ),
    repo_root(),
)
