"""Gemma 4 base sweep.

The instruction-tuned -it repos are not part of this sweep. Stored N is
safetensors bytes / 2 once a checkpoint is cached; the published totals are
the fallback until then. Effective N is Google's count and is never used as
the fit's N. The 26B-A4B entry is the MoE text tower (128 experts, top-8);
N is still total stored parameters, not active parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen import ROOT, SweepSpec

# Confirmed against the google/gemma-4 collection. The 12B checkpoint is the
# unified text tower. 26B-A4B is the mixture-of-experts text tower.
MODELS = (
    "google/gemma-4-E2B",
    "google/gemma-4-E4B",
    "google/gemma-4-12B",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-31B",
)

# Total stored parameters, in billions: safetensors tensor bytes / 2.
# Counted on the instruction-tuned files of these same sizes. stored_params()
# replaces a value once that base checkpoint's index is cached.
PUBLISHED_STORED_B = {
    "google/gemma-4-E2B": 5.123178979,
    "google/gemma-4-E4B": 7.996157418,
    "google/gemma-4-12B": 11.959730224,
    "google/gemma-4-26B-A4B": 25.805936206,
    "google/gemma-4-31B": 31.273088876,
}

# Google's effective counts, in billions. Not used as N in the law.
# A4B = ~4B active parameters at inference.
EFFECTIVE_B = {
    "google/gemma-4-E2B": 2.3,
    "google/gemma-4-E4B": 4.5,
    "google/gemma-4-12B": 11.95,
    "google/gemma-4-26B-A4B": 4.0,
    "google/gemma-4-31B": 30.7,
}

GEMMA = SweepSpec(
    name="gemma",
    models=MODELS,
    runner="sparsegpt.gemma",
    extra_args=(),
    result_dir=ROOT / "figures" / "gemma",
    log_dir=ROOT / "figures" / "gemma" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="31B",
)


def effective_params() -> dict[str, float]:
    return dict(EFFECTIVE_B)


def stored_params() -> dict[str, float]:
    sizes = dict(PUBLISHED_STORED_B)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for model in MODELS:
        org, name = model.split("/", 1)
        measured = _index_billions(hub / f"models--{org}--{name}")
        if measured is not None:
            sizes[model] = measured
    return sizes


def _index_billions(snapshot_root: Path) -> float | None:
    if not snapshot_root.exists():
        return None
    for path in snapshot_root.rglob("model.safetensors.index.json"):
        if ".no_exist" in path.parts or path.stat().st_size == 0:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        total = data.get("metadata", {}).get("total_size")
        if total:
            return float(total) / 2 / 1e9
    return None
