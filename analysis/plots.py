"""Perplexity, log-log, and normalized Plotly pages for one family.

Magnitude panels are Qwen-only and are not written here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from analysis.fit import exponent_at, solve_reference_law
from experiments.qwen import SweepSpec

SPARSITIES = [round(i / 10, 1) for i in range(1, 9)]
COLORS = ("#1d4ed8", "#0f766e", "#b45309", "#6d28d9", "#be123c", "#0369a1")


def _load(spec: SweepSpec) -> list[dict]:
    rows = []
    for model in spec.models:
        path = spec.result_dir / f"{model.rstrip('/').split('/')[-1].lower()}_sparsity.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for point in data["points"]:
            rows.append({**point, "model": model})
    return rows


def _lookup(rows: list[dict], model: str, dataset: str, sparsity: float) -> float | None:
    for point in rows:
        if (
            point["model"] == model
            and point["dataset"] == dataset
            and abs(point["sparsity"] - sparsity) < 1e-9
        ):
            return float(point["perplexity"])
    return None


def _law(rows: list[dict], dataset: str, sizes: dict[str, float]):
    import numpy as np

    sparsities, log_ratios, billions = [], [], []
    usable = []
    for model, n_billions in sizes.items():
        dense = _lookup(rows, model, dataset, 0.0)
        if dense is None or dense <= 0:
            continue
        kept = []
        for sparsity in SPARSITIES:
            perplexity = _lookup(rows, model, dataset, sparsity)
            if perplexity is None or perplexity <= dense:
                continue
            kept.append((sparsity, perplexity))
            sparsities.append(sparsity)
            log_ratios.append(math.log(perplexity / dense))
            billions.append(n_billions)
        if kept:
            usable.append((model, dense, n_billions, kept))
    if len(usable) < 2 or len(sparsities) < 4:
        return None
    amplitude, n_star, slope, rmse = solve_reference_law(
        np.asarray(sparsities),
        np.asarray(log_ratios),
        np.asarray(billions),
    )
    return {
        "a": amplitude,
        "n_star": n_star,
        "b1": slope,
        "rmse": rmse,
        "n": len(sparsities),
        "usable": usable,
    }


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, path, include_plotlyjs=True, full_html=True)
    print(f"wrote {path}", flush=True)


def refresh_family(spec: SweepSpec, sizes: dict[str, float] | None = None) -> None:
    from experiments.gemma import stored_params

    rows = _load(spec)
    if not rows:
        return
    if sizes is None:
        sizes = stored_params() if spec.name == "gemma" else {}
    datasets = sorted({point["dataset"] for point in rows})
    for dataset in datasets:
        suffix = "" if dataset == "wikitext2" else f"_{dataset}"
        _perplexity(spec, rows, dataset, sizes, suffix)
        _loglog(spec, rows, dataset, sizes, suffix)
        _normalized(spec, rows, dataset, sizes, suffix)


def _series(spec: SweepSpec, rows: list[dict], dataset: str):
    present = []
    for index, model in enumerate(spec.models):
        dense = _lookup(rows, model, dataset, 0.0)
        xs, ys = [], []
        if dense is not None:
            xs.append(0.0)
            ys.append(dense)
        for sparsity in SPARSITIES:
            perplexity = _lookup(rows, model, dataset, sparsity)
            if perplexity is not None:
                xs.append(sparsity)
                ys.append(perplexity)
        if len(xs) >= 1:
            present.append((model, COLORS[index % len(COLORS)], xs, ys, dense))
    return present


def _perplexity(spec, rows, dataset, sizes, suffix) -> None:
    fig = go.Figure()
    for model, color, xs, ys, _dense in _series(spec, rows, dataset):
        label = model.split("/")[-1]
        n_billions = sizes.get(model)
        if n_billions:
            label = f"{label} ({n_billions:.2f}B stored)"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label, line={"color": color}))
    fig.update_layout(
        title=f"{spec.name} SparseGPT perplexity ({dataset})",
        xaxis_title="unstructured sparsity",
        yaxis_title="perplexity",
        template="plotly_white",
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_perplexity{suffix}.html")


def _loglog(spec, rows, dataset, sizes, suffix) -> None:
    fig = go.Figure()
    fit = _law(rows, dataset, sizes)
    for model, color, xs, ys, dense in _series(spec, rows, dataset):
        if dense is None or dense <= 0:
            continue
        sx, sy = [], []
        for sparsity, perplexity in zip(xs, ys):
            if sparsity <= 0 or perplexity <= dense:
                continue
            sx.append(math.log10(sparsity))
            sy.append(math.log(math.log(perplexity / dense)))
        if sx:
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="markers",
                    name=model.split("/")[-1],
                    marker={"color": color},
                )
            )
    if fit:
        fig.update_layout(
            title=(
                f"ln ln(ppl/dense) vs log10 s · "
                f"{fit['a']:.2f} · (N / {fit['n_star']:.3f})^({fit['b1']:.2f} log10 s)"
            )
        )
    fig.update_layout(
        xaxis_title="log10 sparsity",
        yaxis_title="ln ln(perplexity / dense)",
        template="plotly_white",
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_loglog{suffix}.html")


def _normalized(spec, rows, dataset, sizes, suffix) -> None:
    fig = go.Figure()
    fit = _law(rows, dataset, sizes)
    for model, color, xs, ys, dense in _series(spec, rows, dataset):
        n_billions = sizes.get(model)
        if dense is None or dense <= 0 or not n_billions or not fit:
            continue
        exponent = exponent_at(n_billions, fit["n_star"], fit["b1"])
        sx, sy = [], []
        for sparsity, perplexity in zip(xs, ys):
            if sparsity <= 0 or perplexity <= dense or exponent == 0:
                continue
            sx.append(sparsity ** exponent)
            sy.append(math.log(perplexity / dense))
        if sx:
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="markers",
                    name=model.split("/")[-1],
                    marker={"color": color},
                )
            )
    if fit:
        xs = [index / 40 for index in range(1, 41)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[fit["a"] * x for x in xs],
                mode="lines",
                name="shared law",
                line={"color": "#111827", "dash": "dash"},
            )
        )
    fig.update_layout(
        title="normalized SparseGPT law",
        xaxis_title="s ^ (b1 log10(N/N*))",
        yaxis_title="ln(perplexity / dense)",
        template="plotly_white",
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_normalized{suffix}.html")
