"""Principle page for one family, fit on that family's stored parameter counts.

This writes figures/<family>/sparsity_principle.html only. The cross-family
pages are figures/sparsity_principle.html and figures/universal_scaling_law.html.
Regenerate those with `python -m sparsity_principle` and
`python -m universal_scaling` when a family gains C4 points. A family page
here is written only after at least two models have dense perplexity and every
tenth from 0.1 to 0.8 on every eval dataset in the SweepSpec.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis.fit import solve_scale_free_law
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


_DATASET_LABELS = {
    "wikitext2": "WikiText2",
    "ptb": "PTB",
    "c4": "C4",
}


def complete_models(spec: SweepSpec) -> list[str]:
    ready = []
    for model in spec.models:
        points = _points(spec, model)
        if all(
            _have(points, dataset, sparsity)
            for dataset in spec.eval_datasets
            for sparsity in [0.0, *TENTHS]
        ):
            ready.append(model)
    return ready


def write_family_page(spec: SweepSpec) -> Path | None:
    ready = complete_models(spec)
    if len(ready) < 2:
        print(f"{spec.name}: principle page waits for 2 complete models ({len(ready)} ready)", flush=True)
        return None
    if spec.name in ("gemma", "gemma_it_wikitext2_cal"):
        from experiments.gemma import stored_params

        sizes = stored_params()
        effective = __import__("experiments.gemma", fromlist=["effective_params"]).effective_params()
    elif spec.name in ("olmo2", "olmo2_wikitext2_cal"):
        from experiments.olmo import stored_params

        sizes = stored_params()
        effective = {}
    elif spec.name in ("llama3", "llama3_wikitext2_cal"):
        from experiments.llama import stored_params

        sizes = stored_params()
        effective = {}
    else:
        from sparsity_principle import EXTRA_PARAMS_B

        sizes = {model: float(EXTRA_PARAMS_B) for model in ready}
        effective = {}
    sections = []
    for dataset in spec.eval_datasets:
        sparsities, log_ratios, billions = [], [], []
        for model in ready:
            if model not in sizes:
                continue
            points = _points(spec, model)
            dense = next(point["perplexity"] for point in points if point["dataset"] == dataset and point["sparsity"] == 0)
            measured = sorted(
                {
                    round(float(point["sparsity"]), 5)
                    for point in points
                    if point["dataset"] == dataset and float(point["sparsity"]) > 0
                }
            )
            for sparsity in measured:
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
        alpha, beta, exponent, rmse = solve_scale_free_law(
            np.asarray(sparsities),
            np.asarray(log_ratios),
            np.asarray(billions),
        )
        print(
            f"{spec.name} {dataset}: {alpha:.6f} * N^({beta:.6f}) * s^{exponent:.6f} "
            f"RMSE {rmse:.4f} over {len(sparsities)}",
            flush=True,
        )
        sections.append((dataset, alpha, beta, exponent, rmse, len(sparsities)))
    path = spec.result_dir / "sparsity_principle.html"
    path.write_text(_html(spec, ready, sizes, effective, sections))
    print(f"wrote {path}", flush=True)
    return path


def _html(spec, ready, sizes, effective, sections) -> str:
    laws = []
    for dataset, alpha, beta, exponent, rmse, count in sections:
        name = _DATASET_LABELS.get(dataset, dataset)
        laws.append(
            f"<p><b>{name}</b> ln(perplexity / dense) = "
            f"{alpha:.2f} · N<sup>{beta:.3f}</sup> · s<sup>{exponent:.2f}</sup>. "
            f"RMSE {rmse:.3f} over {count} points. "
            f"N is total stored parameters in billions.</p>"
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
<p>Unstructured SparseGPT, {cal} calibration (128 × 2048, seed 0), eval on {evals}. The law is scale-free: ln(perplexity / dense) = α · N<sup>β</sup> · s<sup>b</sup>. Size and sparsity factorize, so changing the unit of N only rescales α. Points at or below each model's own dense perplexity are excluded. N is total stored parameters. Effective counts are recorded and are not the fit's N.</p>
{''.join(laws)}
<table>
<tr><th>model</th><th>stored N (billions)</th><th>effective N (billions)</th></tr>
{''.join(rows)}
</table>
</body>
</html>
"""
