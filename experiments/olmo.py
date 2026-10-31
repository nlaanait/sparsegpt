"""OLMo 2 instruction-tuned sparsity sweep.

Four Instruct sizes, C4 calibration (128 × 2048, seed 0), eval on WikiText2
and PTB. Dense WikiText2 must land in SMOKE_PPL_RANGE before any prune runs.
Stored N comes from ``sparsifier.catalog`` and is replaced once a checkpoint
is cached.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.protocol import ROOT, SweepSpec
from sparsifier.catalog import stored_count

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

OLMO2 = SweepSpec(
    name="olmo2",
    models=MODELS,
    runner="sparsegpt.olmo",
    extra_args=(),
    result_dir=ROOT / "results" / "olmo2",
    log_dir=ROOT / "results" / "olmo2" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="32B-Instruct",
    plot=True,
    calibration="c4",
    eval_datasets=("wikitext2", "ptb"),
)

# WikiText2 calibration, C4 and PTB eval. The 32B file is absent until that
# ladder is collected; the loader skips a model with no points.
OLMO2_WIKITEXT2_CAL = SweepSpec(
    name="olmo2_wikitext2_cal",
    models=MODELS,
    runner="sparsegpt.olmo",
    extra_args=(),
    result_dir=ROOT / "results" / "olmo2_wikitext2_cal",
    log_dir=ROOT / "results" / "olmo2_wikitext2_cal" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="32B-Instruct",
    plot=True,
    calibration="wikitext2",
    eval_datasets=("c4", "ptb"),
)


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
        data = json.loads(path.read_text())
        total = data.get("metadata", {}).get("total_size")
        if total:
            return float(total) / 2 / 1e9
    for path in snapshot_root.rglob("model.safetensors"):
        if path.is_file():
            # bf16-dominated single-file packs: bytes / 2 ≈ parameter count.
            return float(path.stat().st_size) / 2 / 1e9
    return None
