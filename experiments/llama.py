"""Llama 3.1 and Llama 3.2 as one sparsity family.

Base text repos and the two Vision repos share figures/llama3/, one fit, and
one set of plots. Every JSON point keeps the full Hugging Face id. Stored N
is safetensors bytes / 2 of that full checkpoint, including the vision tower
on the 11B and 90B repos. The published totals are the architecture counts
until a local snapshot is measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen import ROOT, SweepSpec

MODELS = (
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-11B-Vision",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.2-90B-Vision",
)

# Dense WikiText2 must sit in this band before the sparsity sweep starts.
# 70B and 90B can land near ~3; hundreds or thousands fail.
SMOKE_DATASET = "wikitext2"
SMOKE_PPL_RANGE = (2.0, 40.0)

# Parameter counts, in billions. Text models are LlamaForCausalLM. 11B and
# 90B include the vision tower and the cross-attention layers.
PUBLISHED_STORED_B = {
    "meta-llama/Llama-3.2-1B": 1.2358144,
    "meta-llama/Llama-3.2-3B": 3.212749824,
    "meta-llama/Llama-3.1-8B": 8.030261248,
    "meta-llama/Llama-3.2-11B-Vision": 10.642941475,
    "meta-llama/Llama-3.1-70B": 70.553706496,
    "meta-llama/Llama-3.2-90B-Vision": 88.593355323,
}

LLAMA3 = SweepSpec(
    name="llama3",
    models=MODELS,
    runner="sparsegpt.llama",
    extra_args=(),
    result_dir=ROOT / "figures" / "llama3",
    log_dir=ROOT / "figures" / "llama3" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="90B-Vision",
    plot=True,
    calibration="c4",
    eval_datasets=("wikitext2", "ptb"),
)

# Same six repos with calibration and eval swapped: WikiText2 calibrate,
# C4 and PTB evaluate. Artifacts stay out of figures/llama3/.
LLAMA3_WIKITEXT2_CAL = SweepSpec(
    name="llama3_wikitext2_cal",
    models=MODELS,
    runner="sparsegpt.llama",
    extra_args=(),
    result_dir=ROOT / "figures" / "llama3_wikitext2_cal",
    log_dir=ROOT / "figures" / "llama3_wikitext2_cal" / "logs",
    log_infix="sdpa",
    attention="sdpa",
    holdout_suffix="90B-Vision",
    plot=True,
    calibration="wikitext2",
    eval_datasets=("c4", "ptb"),
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
