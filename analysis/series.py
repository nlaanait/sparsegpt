"""Perplexity rows for family figures.

``*_sparsity.json`` is the primary record. A likelihood file contributes only
its perplexity, and only at a sparsity the sparsity file does not already
have. Points above 0.8 stay out of these pages. A dense perplexity in the
hundreds is not a language-model score and is left out of the fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.protocol import ROOT, SweepSpec

TENTHS = [round(index / 10, 1) for index in range(1, 9)]
MAX_SPARSITY = 0.8
# Hundreds or thousands are not a language-model perplexity.
PLAUSIBLE_DENSE_MAX = 100.0

# 3.6 and 3.8 share the 3.5-27B stored count and sit on the 3.5 pages.
EXTRA_MODELS = ("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B")
FIGURE_MODELS = {
    "qwen3.5": EXTRA_MODELS,
    "qwen3.5_wikitext2_cal": EXTRA_MODELS,
}


def models_for(spec: SweepSpec) -> tuple[str, ...]:
    extra = FIGURE_MODELS.get(spec.name, ())
    return tuple(dict.fromkeys((*spec.models, *extra)))


def plausible_dense(perplexity: float | None) -> bool:
    return (
        perplexity is not None
        and math.isfinite(perplexity)
        and 0.0 < perplexity < PLAUSIBLE_DENSE_MAX
    )


def family_sizes(spec: SweepSpec) -> dict[str, float]:
    if spec.name in ("gemma", "gemma_it_wikitext2_cal"):
        from experiments.gemma import stored_params

        return stored_params()
    if spec.name in ("olmo2", "olmo2_wikitext2_cal"):
        from experiments.olmo import stored_params

        return stored_params()
    if spec.name in ("llama3", "llama3_wikitext2_cal"):
        from experiments.llama import stored_params

        return stored_params()
    if spec.name in ("qwen3.5", "qwen3.5_wikitext2_cal", "qwen", "qwen_wikitext2_cal"):
        return _qwen_sizes()
    return {}


def load_rows(spec: SweepSpec) -> list[dict]:
    """SparseGPT perplexity points with ``sparsity <= 0.8``."""
    found: dict[tuple[str, str, float], dict] = {}
    for model in models_for(spec):
        path = _sparsity_path(spec, model)
        if path is not None:
            _add_file(found, path, model=model, dataset=None)
    for path, dataset in _bundles(spec):
        _add_file(found, path, model=None, dataset=dataset)
    for model in models_for(spec):
        for path, dataset in _likelihood_files(spec, model):
            _add_likelihood(found, path, model=model, dataset=dataset)
    return list(found.values())


def points_for(spec: SweepSpec, model: str, rows: list[dict] | None = None) -> list[dict]:
    source = load_rows(spec) if rows is None else rows
    return [point for point in source if point["model"] == model]


def lookup(rows: list[dict], model: str, dataset: str, sparsity: float) -> float | None:
    for point in rows:
        if (
            point["model"] == model
            and point["dataset"] == dataset
            and abs(point["sparsity"] - sparsity) < 1e-9
        ):
            return float(point["perplexity"])
    return None


def short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


def _qwen_sizes() -> dict[str, float]:
    from sparsifier.catalog import models

    return {
        record.model_id: float(record.stored_b)
        for record in models()
        if record.family == "qwen" and record.stored_b is not None
    }


def _sparsity_path(spec: SweepSpec, model: str) -> Path | None:
    name = f"{short_name(model)}_sparsity.json"
    direct = spec.result_dir / name
    if direct.is_file():
        return direct
    extras = {
        "qwen3.5": ROOT / "results" / "qwen" / name,
        "qwen3.5_wikitext2_cal": ROOT / "results" / "qwen_wikitext2_cal" / name,
    }
    other = extras.get(spec.name)
    if other is not None and other.is_file():
        return other
    return None


def _bundles(spec: SweepSpec) -> list[tuple[Path, str]]:
    if spec.name != "qwen3.5":
        return []
    root = ROOT / "results" / "qwen3.5"
    return [
        (root / "qwen3.5_family_sparsity_wikitext2.json", "wikitext2"),
        (root / "qwen3.5_family_sparsity_ptb.json", "ptb"),
    ]


def _likelihood_dirs(spec: SweepSpec, model: str) -> list[Path]:
    dirs = [spec.result_dir / "likelihood"]
    extra = _sparsity_path(spec, model)
    if extra is not None:
        dirs.append(extra.parent / "likelihood")
    unique = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def _likelihood_files(spec: SweepSpec, model: str) -> list[tuple[Path, str]]:
    prefix = f"{short_name(model)}_"
    found = []
    for directory in _likelihood_dirs(spec, model):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{prefix}*.json")):
            dataset = path.stem[len(prefix) :]
            if dataset:
                found.append((path, dataset))
    return found


def _add_file(found: dict, path: Path, *, model: str | None, dataset: str | None) -> None:
    data = json.loads(path.read_text())
    for point in data.get("points", []):
        method = str(point.get("method", "")).lower()
        if method not in ("dense", "sparsegpt"):
            continue
        sparsity = float(point["sparsity"])
        if sparsity > MAX_SPARSITY + 1e-9:
            continue
        row_model = model or point["model"]
        row_dataset = dataset or point["dataset"]
        _put(
            found,
            row_model,
            row_dataset,
            sparsity,
            float(point["perplexity"]),
            "dense" if sparsity == 0.0 or method == "dense" else "sparsegpt",
        )


def _add_likelihood(found: dict, path: Path, *, model: str, dataset: str) -> None:
    data = json.loads(path.read_text())
    row_model = data.get("model") or model
    for key, value in data.items():
        if not isinstance(value, dict) or "perplexity" not in value:
            continue
        try:
            sparsity = float(key)
        except ValueError:
            continue
        if sparsity > MAX_SPARSITY + 1e-9:
            continue
        _put(
            found,
            row_model,
            dataset,
            sparsity,
            float(value["perplexity"]),
            "dense" if sparsity == 0.0 else "sparsegpt",
        )


def _put(found, model, dataset, sparsity, perplexity, method) -> None:
    if not math.isfinite(perplexity):
        return
    key = (model, dataset, round(float(sparsity), 5))
    if key in found:
        return
    found[key] = {
        "model": model,
        "dataset": dataset,
        "method": method,
        "sparsity": float(sparsity),
        "perplexity": float(perplexity),
    }
