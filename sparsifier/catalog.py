"""Registered models.

One record per checkpoint the sweeps measure: Hugging Face id, family,
runner, who owns the layer loop, and stored parameter count in billions.
An id that is not in the table still resolves when it matches one prefix,
with no stored count. Experiments import this table. This module does not
import them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecord:
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
    _Prefix("Qwen/", "qwen", "sparsegpt.qwen", "decoder"),
    _Prefix("google/gemma", "gemma", "sparsegpt.gemma", "own"),
    _Prefix("allenai/OLMo", "olmo", "sparsegpt.olmo", "decoder"),
    _Prefix("meta-llama/", "llama", "sparsegpt.llama", "decoder"),
    _Prefix("facebook/opt", "opt", "sparsegpt.opt", "decoder"),
    _Prefix("bigscience/bloom", "bloom", "sparsegpt.bloom", "decoder"),
)

# Stored N is safetensors bytes / 2, in billions. Qwen 3.6 and 3.8 use the
# Qwen3.5-27B count. Gemma's 26B-A4B count is the whole mixture-of-experts
# checkpoint, not the active parameters.
_MODELS = (
    ModelRecord("Qwen/Qwen3.5-0.8B", "qwen", "sparsegpt.qwen", "decoder", 0.873438784),
    ModelRecord("Qwen/Qwen3.5-2B", "qwen", "sparsegpt.qwen", "decoder", 2.274069824),
    ModelRecord("Qwen/Qwen3.5-4B", "qwen", "sparsegpt.qwen", "decoder", 4.659865088),
    ModelRecord("Qwen/Qwen3.5-9B", "qwen", "sparsegpt.qwen", "decoder", 9.653104368),
    ModelRecord("Qwen/Qwen3.5-27B", "qwen", "sparsegpt.qwen", "decoder", 27.781427952),
    ModelRecord("Qwen/Qwen3.6-27B", "qwen", "sparsegpt.qwen", "decoder", 27.781427952),
    ModelRecord("Qwen/Qwen3.8-27B", "qwen", "sparsegpt.qwen", "decoder", 27.781427952),
    ModelRecord("google/gemma-4-E2B", "gemma", "sparsegpt.gemma", "own", 5.123178979),
    ModelRecord("google/gemma-4-E4B", "gemma", "sparsegpt.gemma", "own", 7.996157418),
    ModelRecord("google/gemma-4-12B", "gemma", "sparsegpt.gemma", "own", 11.959730224),
    ModelRecord("google/gemma-4-26B-A4B", "gemma", "sparsegpt.gemma", "own", 25.805936206),
    ModelRecord("google/gemma-4-31B", "gemma", "sparsegpt.gemma", "own", 31.273088876),
    ModelRecord("allenai/OLMo-2-0425-1B-Instruct", "olmo", "sparsegpt.olmo", "decoder", 1.484916736),
    ModelRecord("allenai/OLMo-2-1124-7B-Instruct", "olmo", "sparsegpt.olmo", "decoder", 7.298617344),
    ModelRecord("allenai/OLMo-2-1124-13B-Instruct", "olmo", "sparsegpt.olmo", "decoder", 13.716198400),
    ModelRecord("allenai/OLMo-2-0325-32B-Instruct", "olmo", "sparsegpt.olmo", "decoder", 32.234279936),
    ModelRecord("meta-llama/Llama-3.2-1B", "llama", "sparsegpt.llama", "decoder", 1.2358144),
    ModelRecord("meta-llama/Llama-3.2-3B", "llama", "sparsegpt.llama", "decoder", 3.212749824),
    ModelRecord("meta-llama/Llama-3.1-8B", "llama", "sparsegpt.llama", "decoder", 8.030261248),
    ModelRecord("meta-llama/Llama-3.2-11B-Vision", "llama", "sparsegpt.llama", "decoder", 10.642941475),
    ModelRecord("meta-llama/Llama-3.1-70B", "llama", "sparsegpt.llama", "decoder", 70.553706496),
    ModelRecord("meta-llama/Llama-3.2-90B-Vision", "llama", "sparsegpt.llama", "decoder", 88.593355323),
)

_EXACT = {record.model_id: record for record in _MODELS}


def models() -> tuple[ModelRecord, ...]:
    return tuple(_EXACT[model_id] for model_id in sorted(_EXACT))


def stored_count(model_id: str) -> float:
    count = resolve(model_id).stored_b
    if count is None:
        raise LookupError(f"{model_id} has no stored parameter count")
    return count


def resolve(model_id: str) -> ModelRecord:
    exact = _EXACT.get(model_id)
    if exact is not None:
        return exact
    matches = [prefix for prefix in _PREFIXES if model_id.startswith(prefix.prefix)]
    if len(matches) != 1:
        found = ", ".join(prefix.family for prefix in matches) or "none"
        raise LookupError(f"expected one family for {model_id!r}, found {found}")
    prefix = matches[0]
    return ModelRecord(model_id, prefix.family, prefix.runner, prefix.loop, None)
