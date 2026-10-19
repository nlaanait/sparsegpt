"""Principle page for one family, fit on that family's stored parameter counts.

The Qwen page at figures/sparsity_principle.html is left untouched. A family
page is written only after at least two models have dense perplexity and every
tenth from 0.1 to 0.8 on both WikiText2 and PTB.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis.fit import solve_reference_law
from experiments.qwen import SweepSpec

TENTHS = [round(i / 10, 1) for i in range(1, 9)]


def _points(spec: SweepSpec, model: str) -> list[dict]:
    path = spec.result_dir / f"{model.rstrip('/').split('/')[-1].lower()}_sparsity.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("points", [])


def _have(points: list[dict], dataset: str, sparsity: float) -> bool:
    return any(
        point["dataset"] == dataset and abs(point["sparsity"] - sparsity) < 1e-9 for point in points
    )


def complete_models(spec: SweepSpec) -> list[str]:
    ready = []
    for model in spec.models:
        points = _points(spec, model)
        if all(_have(points, dataset, sparsity) for dataset in ("wikitext2", "ptb") for sparsity in [0.0, *TENTHS]):
            ready.append(model)
    return ready


def write_family_page(spec: SweepSpec) -> Path | None:
    ready = complete_models(spec)
    if len(ready) < 2:
        print(f"{spec.name}: principle page waits for 2 complete models ({len(ready)} ready)", flush=True)
        return None
    if spec.name == "gemma":
        from experiments.gemma import stored_params

        sizes = stored_params()
        effective = __import__("experiments.gemma", fromlist=["effective_params"]).effective_params()
    else:
        sizes = {}
        effective = {}
    sections = []
    for dataset in ("wikitext2", "ptb"):
        sparsities, log_ratios, billions = [], [], []
        for model in ready:
            if model not in sizes:
                continue
            points = _points(spec, model)
            dense = next(point["perplexity"] for point in points if point["dataset"] == dataset and point["sparsity"] == 0)
            for sparsity in TENTHS:
                perplexity = next(
                    point["perplexity"]
                    for point in points
                    if point["dataset"] == dataset and abs(point["sparsity"] - sparsity) < 1e-9
                )
                if perplexity <= dense:
                    continue
                sparsities.append(sparsity)
                log_ratios.append(math.log(perplexity / dense))
                billions.append(sizes[model])
        if len(sparsities) < 4:
            print(f"{spec.name}: not enough {dataset} points above dense", flush=True)
            return None
        amplitude, n_star, slope, rmse = solve_reference_law(
            np.asarray(sparsities),
            np.asarray(log_ratios),
            np.asarray(billions),
        )
        print(
            f"{spec.name} {dataset}: {amplitude:.6f} * (N / {n_star:.6f})^({slope:.6f} log10 s) "
            f"RMSE {rmse:.4f} over {len(sparsities)}",
            flush=True,
        )
        sections.append((dataset, amplitude, n_star, slope, rmse, len(sparsities)))
    path = spec.result_dir / "sparsity_principle.html"
    path.write_text(_html(spec, ready, sizes, effective, sections))
    print(f"wrote {path}", flush=True)
    return path


def _html(spec, ready, sizes, effective, sections) -> str:
    laws = []
    for dataset, amplitude, n_star, slope, rmse, count in sections:
        name = "WikiText2" if dataset == "wikitext2" else "PTB"
        laws.append(
            f"<p><b>{name}</b> ln(perplexity / dense) = "
            f"{amplitude:.2f} · (N / {n_star:.3f})^({slope:.2f} log10 s). "
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
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{spec.name} sparsity principle</title></head>
<body>
<h1>{spec.name} sparsity principle</h1>
<p>Unstructured SparseGPT, C4 calibration (128 × 2048, seed 0). Points at or below each model's own dense perplexity are excluded. The fit uses total stored parameters so N matches the Qwen page. Effective counts are recorded and are not the fit's N.</p>
{''.join(laws)}
<table>
<tr><th>model</th><th>stored N (billions)</th><th>effective N (billions)</th></tr>
{''.join(rows)}
</table>
</body>
</html>
"""
