"""Registered models.

One record per checkpoint the sweeps measure: Hugging Face id, family,
runner, who owns the layer loop, and stored parameter count in billions.
An id that is not in the table still resolves when it matches one prefix,
with no stored count. Experiments import this table. This module does not
import them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    """One checkpoint the sweeps can measure.

    ``stored_b`` is safetensors bytes / 2, in billions. ``None`` means the
    id matched a family prefix but is not a registered checkpoint.
    ``loop`` is ``decoder`` when the shared layer loop can drive it, or
    ``own`` when the family runs its own loop (Gemma).
    """

    model_id: str
    family: str
    runner: str
    loop: str
    stored_b: float | None = None


@dataclass(frozen=True)
class _Prefix:
    prefix: str
    family: str
    runner: str
    loop: str


_PREFIXES = (
    _Prefix("Qwen/", "qwen", "sparsifier", "decoder"),
    _Prefix("google/gemma", "gemma", "sparsifier", "own"),
    _Prefix("allenai/OLMo", "olmo", "sparsifier", "decoder"),
    _Prefix("meta-llama/", "llama", "sparsifier", "decoder"),
    _Prefix("facebook/opt", "opt", "sparsegpt.opt", "decoder"),
    _Prefix("bigscience/bloom", "bloom", "sparsegpt.bloom", "decoder"),
)

# Stored N is safetensors bytes / 2, in billions. Qwen 3.6 and 3.8 use the
# Qwen3.5-27B count. Gemma's 26B-A4B count is the whole mixture-of-experts
# checkpoint, not the active parameters.
_MODELS = (
    ModelRecord("Qwen/Qwen3.5-0.8B", "qwen", "sparsifier", "decoder", 0.873438784),
    ModelRecord("Qwen/Qwen3.5-2B", "qwen", "sparsifier", "decoder", 2.274069824),
    ModelRecord("Qwen/Qwen3.5-4B", "qwen", "sparsifier", "decoder", 4.659865088),
    ModelRecord("Qwen/Qwen3.5-9B", "qwen", "sparsifier", "decoder", 9.653104368),
    ModelRecord("Qwen/Qwen3.5-27B", "qwen", "sparsifier", "decoder", 27.781427952),
    ModelRecord("Qwen/Qwen3.6-27B", "qwen", "sparsifier", "decoder", 27.781427952),
    ModelRecord("Qwen/Qwen3.8-27B", "qwen", "sparsifier", "decoder", 27.781427952),
    ModelRecord("google/gemma-4-E2B", "gemma", "sparsifier", "own", 5.123178979),
    ModelRecord("google/gemma-4-E4B", "gemma", "sparsifier", "own", 7.996157418),
    ModelRecord("google/gemma-4-12B", "gemma", "sparsifier", "own", 11.959730224),
    ModelRecord("google/gemma-4-26B-A4B", "gemma", "sparsifier", "own", 25.805936206),
    ModelRecord("google/gemma-4-31B", "gemma", "sparsifier", "own", 31.273088876),
    ModelRecord("allenai/OLMo-2-0425-1B-Instruct", "olmo", "sparsifier", "decoder", 1.484916736),
    ModelRecord("allenai/OLMo-2-1124-7B-Instruct", "olmo", "sparsifier", "decoder", 7.298617344),
    ModelRecord("allenai/OLMo-2-1124-13B-Instruct", "olmo", "sparsifier", "decoder", 13.716198400),
    ModelRecord("allenai/OLMo-2-0325-32B-Instruct", "olmo", "sparsifier", "decoder", 32.234279936),
    ModelRecord("meta-llama/Llama-3.2-1B", "llama", "sparsifier", "decoder", 1.2358144),
    ModelRecord("meta-llama/Llama-3.2-3B", "llama", "sparsifier", "decoder", 3.212749824),
    ModelRecord("meta-llama/Llama-3.1-8B", "llama", "sparsifier", "decoder", 8.030261248),
    ModelRecord("meta-llama/Llama-3.2-11B-Vision", "llama", "sparsifier", "decoder", 10.642941475),
    ModelRecord("meta-llama/Llama-3.1-70B", "llama", "sparsifier", "decoder", 70.553706496),
    ModelRecord("meta-llama/Llama-3.2-90B-Vision", "llama", "sparsifier", "decoder", 88.593355323),
)

_EXACT = {record.model_id: record for record in _MODELS}


def models() -> tuple[ModelRecord, ...]:
    """Registered checkpoints, sorted by Hugging Face id."""
    return tuple(_EXACT[model_id] for model_id in sorted(_EXACT))


def stored_count(model_id: str) -> float:
    """Stored parameter count in billions. Rejects prefix-only ids."""
    count = resolve(model_id).stored_b
    if count is None:
        raise LookupError(f"{model_id} has no stored parameter count")
    return count


def snapshot_billions(model_id: str, hub: Path | None = None) -> float | None:
    """Safetensors bytes / 2, in billions, from a local Hugging Face snapshot."""
    if "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    root = (Path.home() / ".cache" / "huggingface" / "hub" if hub is None else hub)
    return _snapshot_billions(root / f"models--{org}--{name}")


def measured_stored_counts(model_ids) -> dict[str, float]:
    """Catalog N, replaced when that checkpoint's snapshot is on disk."""
    sizes = {}
    for model_id in model_ids:
        measured = snapshot_billions(model_id)
        sizes[model_id] = stored_count(model_id) if measured is None else measured
    return sizes


def _snapshot_billions(snapshot_root: Path) -> float | None:
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


def resolve(model_id: str) -> ModelRecord:
    """Exact catalog row, or a prefix match with no stored count."""
    exact = _EXACT.get(model_id)
    if exact is not None:
        return exact
    matches = [prefix for prefix in _PREFIXES if model_id.startswith(prefix.prefix)]
    if len(matches) != 1:
        found = ", ".join(prefix.family for prefix in matches) or "none"
        raise LookupError(f"expected one family for {model_id!r}, found {found}")
    prefix = matches[0]
    return ModelRecord(model_id, prefix.family, prefix.runner, prefix.loop, None)
