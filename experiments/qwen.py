"""Qwen family spec.

The instruction-tuned Qwen checkpoints used for the existing sweeps do not
put "it" in the repo name. The 3.6/3.8 driver is qwen_36_38_sparsity.py. Its results live under
figures/qwen/ now that the 3.8 job has exited.
"""

from __future__ import annotations

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


QWEN_36_38 = SweepSpec(
    name="qwen",
    models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
    runner="sparsegpt.qwen",
    extra_args=(),
    result_dir=ROOT / "figures" / "qwen",
    log_dir=ROOT / "figures" / "qwen" / "logs",
    log_infix="pytorch",
    attention="atlas-gdn",
    holdout_suffix="27B",
    plot=False,
    calibration="c4",
    eval_datasets=("ptb", "wikitext2"),
)

# Same 3.6/3.8 series with calibration and eval swapped: WikiText2 calibrate,
# C4 and PTB evaluate. Kept under its own tree so figures/qwen/ stays the
# C4-calibrated baseline.
QWEN_WIKITEXT2_CAL = SweepSpec(
    name="qwen_wikitext2_cal",
    models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
    runner="sparsegpt.qwen",
    extra_args=(),
    result_dir=ROOT / "figures" / "qwen_wikitext2_cal",
    log_dir=ROOT / "figures" / "qwen_wikitext2_cal" / "logs",
    log_infix="pytorch",
    attention="atlas-gdn",
    holdout_suffix="27B",
    plot=True,
    calibration="wikitext2",
    eval_datasets=("c4", "ptb"),
)
