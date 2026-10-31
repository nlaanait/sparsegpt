"""Principle page for one family, fit on that family's stored parameter counts.

The comparison page at results/sparsity_principle.html is left untouched.
A family section is written once two models on that eval have a plausible
dense perplexity and every tenth from 0.1 to 0.8.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from analysis.fit import format_n_star, solve_reference_law
from analysis.series import TENTHS, family_sizes, load_rows, lookup, models_for, plausible_dense
from experiments.protocol import SweepSpec


def _have(points: list[dict], dataset: str, sparsity: float) -> bool:
    return any(
        point["dataset"] == dataset and abs(point["sparsity"] - sparsity) < 1e-9 for point in points
    )


_DATASET_LABELS = {
    "wikitext2": "WikiText2",
    "ptb": "PTB",
    "c4": "C4",
}


def _ready(spec: SweepSpec, rows: list[dict], dataset: str) -> list[str]:
    ready = []
    for model in models_for(spec):
        dense = lookup(rows, model, dataset, 0.0)
        if not plausible_dense(dense):
            continue
        if all(_have(points_for_model(rows, model), dataset, sparsity) for sparsity in TENTHS):
            ready.append(model)
    return ready


def points_for_model(rows: list[dict], model: str) -> list[dict]:
    return [point for point in rows if point["model"] == model]


def write_family_page(spec: SweepSpec, sizes: dict[str, float] | None = None) -> Path | None:
    rows = load_rows(spec)
    if sizes is None:
        sizes = family_sizes(spec)
    effective = {}
    if spec.name == "gemma":
        from experiments.gemma import effective_params

        effective = effective_params()
    sections = []
    ready_union: list[str] = []
    for dataset in spec.eval_datasets:
        ready = [model for model in _ready(spec, rows, dataset) if model in sizes]
        if len(ready) < 2:
            print(f"{spec.name}: {dataset} principle waits for 2 complete models ({len(ready)} ready)", flush=True)
            continue
        sparsities, log_ratios, billions = [], [], []
        for model in ready:
            dense = lookup(rows, model, dataset, 0.0)
            for point in points_for_model(rows, model):
                if point["dataset"] != dataset or point["sparsity"] <= 0:
                    continue
                perplexity = float(point["perplexity"])
                if perplexity <= dense:
                    continue
                sparsities.append(float(point["sparsity"]))
                log_ratios.append(math.log(perplexity / dense))
                billions.append(sizes[model])
        if len(sparsities) < 4:
            print(f"{spec.name}: not enough {dataset} points above dense", flush=True)
            continue
        amplitude, n_star, slope, rmse = solve_reference_law(
            np.asarray(sparsities),
            np.asarray(log_ratios),
            np.asarray(billions),
        )
        print(
            f"{spec.name} {dataset}: {amplitude:.6f} * (N / {format_n_star(n_star)})^({slope:.6f} log10 s) "
            f"RMSE {rmse:.4f} over {len(sparsities)}",
            flush=True,
        )
        sections.append((dataset, amplitude, n_star, slope, rmse, len(sparsities)))
        for model in ready:
            if model not in ready_union:
                ready_union.append(model)
    if not sections:
        return None
    path = spec.result_dir / "sparsity_principle.html"
    path.write_text(_html(spec, ready_union, sizes, effective, sections))
    print(f"wrote {path}", flush=True)
    return path


def _html(spec, ready, sizes, effective, sections) -> str:
    laws = []
    for dataset, amplitude, n_star, slope, rmse, count in sections:
        name = _DATASET_LABELS.get(dataset, dataset)
        laws.append(
            f"<p><b>{name}</b> ln(perplexity / dense) = "
            f"{amplitude:.2f} · (N / {format_n_star(n_star)})^({slope:.2f} log10 s). "
            f"RMSE {rmse:.3f} over {count} points. "
            f"N and N* are total stored parameters in billions.</p>"
        )
    rows = []
    for model in ready:
        label = model.split("/")[-1]
        stored = sizes.get(model)
        eff = effective.get(model)
        stored_cell = f"{stored:.3f}" if stored else ""
        eff_cell = f"{eff:.3f}" if eff else ""
        rows.append(f"<tr><td>{label}</td><td>{stored_cell}</td><td>{eff_cell}</td></tr>")
    cal = _DATASET_LABELS.get(spec.calibration, spec.calibration)
    evals = ", ".join(_DATASET_LABELS.get(name, name) for name in spec.eval_datasets)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{spec.name} sparsity principle</title></head>
<body>
<h1>{spec.name} sparsity principle</h1>
<p>Unstructured SparseGPT, {cal} calibration (128 × 2048, seed 0), eval on {evals}. Points at or below each model's own dense perplexity are excluded. The fit uses total stored parameters so N matches the Qwen page. Effective counts are recorded and are not the fit's N.</p>
{''.join(laws)}
<table>
<tr><th>model</th><th>stored N (billions)</th><th>effective N (billions)</th></tr>
{''.join(rows)}
</table>
</body>
</html>
"""
