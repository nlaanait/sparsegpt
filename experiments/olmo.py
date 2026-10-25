"""OLMo 2 instruction-tuned sparsity sweep.

Four Instruct sizes, C4 calibration (128 × 2048, seed 0), eval on WikiText2
and PTB. Dense WikiText2 must land in SMOKE_PPL_RANGE before any prune runs.
Stored N is safetensors bytes / 2 once a checkpoint is cached; the published
totals are the fallback until then.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen import ROOT, SweepSpec

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

# Total stored parameters, in billions: HF safetensors parameter counts
# (bf16 bytes / 2). Fallback until a local snapshot is measured.
PUBLISHED_STORED_B = {
    "allenai/OLMo-2-0425-1B-Instruct": 1.484916736,
    "allenai/OLMo-2-1124-7B-Instruct": 7.298617344,
    "allenai/OLMo-2-1124-13B-Instruct": 13.716198400,
    "allenai/OLMo-2-0325-32B-Instruct": 32.234279936,
}

OLMO2 = SweepSpec(
    name="olmo2",
    models=MODELS,
    runner="sparsegpt.olmo",
    extra_args=(),
    result_dir=ROOT / "figures" / "olmo2",
    log_dir=ROOT / "figures" / "olmo2" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="32B-Instruct",
    plot=True,
    calibration="c4",
    eval_datasets=("wikitext2", "ptb"),
)


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
    for path in snapshot_root.rglob("model.safetensors"):
        if path.is_file():
            # bf16-dominated single-file packs: bytes / 2 ≈ parameter count.
            return float(path.stat().st_size) / 2 / 1e9
    return None
