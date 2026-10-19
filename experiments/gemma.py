"""Gemma 4 instruction-tuned sweep.

Qwen's instruction-tuned checkpoints omit "it" from the repo name. Gemma's
matching weights are the -it repos. The 26B-A4B mixture is not part of this
sweep. Stored N is safetensors bytes / 2 once a checkpoint is cached; the
published totals are the fallback until then. Effective N is Google's count
and is never used as the fit's N.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen import ROOT, SweepSpec

# Confirmed against the google/gemma-4 collection. The 12B Unified checkpoint
# is the instruction-tuned text repo, not a pretrained or MoE id.
MODELS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-31B-it",
)

# Total stored parameters, in billions: safetensors tensor bytes / 2.
# Measured from the instruction-tuned checkpoints (single-file data offsets,
# or the 31B index metadata). This is the N used in the fit.
PUBLISHED_STORED_B = {
    "google/gemma-4-E2B-it": 5.123178979,
    "google/gemma-4-E4B-it": 7.996157418,
    "google/gemma-4-12B-it": 11.959730224,
    "google/gemma-4-31B-it": 31.273088876,
}

# Google's effective counts, in billions. Not used as N in the law.
EFFECTIVE_B = {
    "google/gemma-4-E2B-it": 2.3,
    "google/gemma-4-E4B-it": 4.5,
    "google/gemma-4-12B-it": 11.95,
    "google/gemma-4-31B-it": 30.7,
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
    holdout_suffix="31B-it",
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
        data = json.loads(path.read_text())
        total = data.get("metadata", {}).get("total_size")
        if total:
            return float(total) / 2 / 1e9
    return None
