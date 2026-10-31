"""Gemma 4 base sweep.

The instruction-tuned -it repos are not part of this sweep. Stored N comes
from ``sparsifier.catalog`` and is replaced once a checkpoint is cached.
Effective N is Google's count and is never used as the fit's N. The 26B-A4B
entry is the MoE text tower (128 experts, top-8); N is still total stored
parameters, not active parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.protocol import ROOT, SweepSpec
from sparsifier.catalog import stored_count

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

GEMMA = SweepSpec(
    name="gemma",
    models=MODELS,
    runner="sparsegpt.gemma",
    extra_args=(),
    result_dir=ROOT / "results" / "gemma",
    log_dir=ROOT / "results" / "gemma" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="31B",
)


def effective_params() -> dict[str, float]:
    return dict(EFFECTIVE_B)


def stored_params() -> dict[str, float]:
    """Catalog stored N, replaced once that checkpoint's index is cached."""
    sizes = {model: stored_count(model) for model in MODELS}
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
    for path in snapshot_root.rglob("model.safetensors"):
        if path.is_file() and ".no_exist" not in path.parts:
            return float(path.stat().st_size) / 2 / 1e9
    return None
