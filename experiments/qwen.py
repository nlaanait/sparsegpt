"""Qwen family spec.

The instruction-tuned Qwen checkpoints used for the existing sweeps do not
put "it" in the repo name. The live 3.6/3.8 driver remains
qwen_36_38_sparsity.py and still writes figures/qwen3.8-27b_sparsity.json.
This module only names that protocol so the analysis package can find it.
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


QWEN_36_38 = SweepSpec(
    name="qwen",
    models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
    runner="sparsegpt.qwen",
    extra_args=("--pytorch-attention",),
    result_dir=ROOT / "figures",
    log_dir=ROOT / "figures" / "logs",
    log_infix="pytorch",
    attention="pytorch",
    holdout_suffix="27B",
    plot=False,
)
