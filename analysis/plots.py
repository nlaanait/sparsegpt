"""Perplexity, log-log, and normalized Plotly pages for one family.

Magnitude panels are Qwen-only and are not written here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from analysis.fit import solve_scale_free_law
from experiments.qwen import SweepSpec

SPARSITIES = [round(i / 10, 1) for i in range(1, 9)]
COLORS = ("#1d4ed8", "#0f766e", "#b45309", "#6d28d9", "#be123c", "#0369a1")
# Follow-up samples reach 0.99. Curves and log axes cover that end.
SPARSITY_AXIS_MAX = 0.99


def _measured_sparsities(rows: list[dict], dataset: str | None = None) -> list[float]:
    found = {float(sparsity) for sparsity in SPARSITIES}
    for point in rows:
        if dataset is not None and point["dataset"] != dataset:
            continue
        sparsity = float(point["sparsity"])
        if sparsity > 0:
            found.add(round(sparsity, 5))
    return sorted(found)


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
        for sparsity in _measured_sparsities(rows, dataset):
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
    alpha, beta, exponent, rmse = solve_scale_free_law(
        np.asarray(sparsities),
        np.asarray(log_ratios),
        np.asarray(billions),
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "b": exponent,
        "rmse": rmse,
        "n": len(sparsities),
        "usable": usable,
    }


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, path, include_plotlyjs=True, full_html=True)
    print(f"wrote {path}", flush=True)


def _family_sizes(spec: SweepSpec) -> dict[str, float]:
    if spec.name in ("gemma", "gemma_it_wikitext2_cal"):
        from experiments.gemma import stored_params

        return stored_params()
    if spec.name in ("olmo2", "olmo2_wikitext2_cal"):
        from experiments.olmo import stored_params

        return stored_params()
    if spec.name in ("llama3", "llama3_wikitext2_cal"):
        from experiments.llama import stored_params

        return stored_params()
    # Qwen 3.6/3.8 share the 3.5-27B stored count used on the principle page.
    from sparsity_principle import EXTRA_PARAMS_B

    return {model: float(EXTRA_PARAMS_B) for model in spec.models}


def refresh_family(spec: SweepSpec, sizes: dict[str, float] | None = None) -> None:
    rows = _load(spec)
    if not rows:
        return
    if sizes is None:
        sizes = _family_sizes(spec)
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
        for sparsity in _measured_sparsities(rows, dataset):
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


def _law_log_ratio(fit: dict, n_billions: float, sparsity: float) -> float | None:
    """ln(perplexity / dense) = α · N^β · s^b."""
    if sparsity <= 0 or n_billions <= 0:
        return None
    value = fit["alpha"] * (n_billions ** fit["beta"]) * (sparsity ** fit["b"])
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _collapsed(fit: dict, n_billions: float, log_ratio: float) -> float | None:
    """N^(-β) · ln(perplexity / dense). One curve, α · s^b, for every size."""
    if n_billions <= 0 or log_ratio <= 0:
        return None
    scale = n_billions ** (-fit["beta"])
    value = log_ratio * scale
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _law_label(fit: dict | None) -> str:
    if not fit:
        return ""
    return (
        f"{fit['alpha']:.2f} · N^({fit['beta']:.3f}) · s^{fit['b']:.2f}"
        f" · RMSE {fit['rmse']:.3f} over {fit['n']}"
    )


def _loglog(spec, rows, dataset, sizes, suffix) -> None:
    """Scale-free collapse: N^(-β) · ln(ppl / dense) versus sparsity, both axes log."""
    fig = go.Figure()
    fit = _law(rows, dataset, sizes)
    if fit:
        grid = [0.08 * (SPARSITY_AXIS_MAX / 0.08) ** (index / 79) for index in range(80)]
        curve = [(sparsity, fit["alpha"] * sparsity ** fit["b"]) for sparsity in grid]
        curve = [(sparsity, value) for sparsity, value in curve if math.isfinite(value) and value > 0]
        if curve:
            fig.add_trace(
                go.Scatter(
                    x=[sparsity for sparsity, _value in curve],
                    y=[value for _sparsity, value in curve],
                    mode="lines",
                    name="α · s^b",
                    line={"color": "#111827", "width": 2},
                )
            )
    for model, color, xs, ys, dense in _series(spec, rows, dataset):
        if dense is None or dense <= 0 or not fit:
            continue
        n_billions = sizes.get(model)
        if not n_billions:
            continue
        sx, sy = [], []
        for sparsity, perplexity in zip(xs, ys):
            if sparsity <= 0 or perplexity <= dense:
                continue
            collapsed = _collapsed(fit, n_billions, math.log(perplexity / dense))
            if collapsed is None:
                continue
            sx.append(sparsity)
            sy.append(collapsed)
        if sx:
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="markers",
                    name=model.split("/")[-1],
                    marker={"color": color, "size": 8},
                )
            )
    fig.update_layout(
        title=f"{spec.name} collapse ({dataset})",
        xaxis_title="unstructured sparsity",
        yaxis_title="N^(−β) · ln(perplexity / dense)",
        template="plotly_white",
        xaxis={"type": "log", "range": [math.log10(0.08), math.log10(SPARSITY_AXIS_MAX + 0.005)]},
        yaxis={"type": "log"},
    )
    if fit:
        fig.add_annotation(
            text=_law_label(fit),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
        )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_loglog{suffix}.html")


def _normalized(spec, rows, dataset, sizes, suffix) -> None:
    """Perplexity / own dense perplexity versus sparsity. Log y, dashed line at 1."""
    fig = go.Figure()
    fit = _law(rows, dataset, sizes)
    fig.add_trace(
        go.Scatter(
            x=[0.0, SPARSITY_AXIS_MAX],
            y=[1.0, 1.0],
            mode="lines",
            name="dense",
            line={"color": "#666666", "width": 1, "dash": "dash"},
        )
    )
    shown = [1.0]
    for model, color, xs, ys, dense in _series(spec, rows, dataset):
        if dense is None or dense <= 0:
            continue
        label = model.split("/")[-1]
        n_billions = sizes.get(model)
        if fit and n_billions:
            grid = [0.05 + (SPARSITY_AXIS_MAX - 0.05) * index / 79 for index in range(80)]
            curve = []
            for sparsity in grid:
                log_ratio = _law_log_ratio(fit, n_billions, sparsity)
                if log_ratio is None:
                    continue
                predicted = math.exp(log_ratio)
                if math.isfinite(predicted) and predicted > 0:
                    curve.append((sparsity, predicted))
            if curve:
                shown.extend(value for _sparsity, value in curve)
                fig.add_trace(
                    go.Scatter(
                        x=[sparsity for sparsity, _value in curve],
                        y=[value for _sparsity, value in curve],
                        mode="lines",
                        name=f"{label} law",
                        legendgroup=label,
                        showlegend=False,
                        line={"color": color, "width": 2},
                    )
                )
        ratios = [
            (sparsity, perplexity / dense)
            for sparsity, perplexity in zip(xs, ys)
            if perplexity > 0 and math.isfinite(perplexity)
        ]
        if not ratios:
            continue
        shown.extend(ratio for _sparsity, ratio in ratios)
        fig.add_trace(
            go.Scatter(
                x=[sparsity for sparsity, _ratio in ratios],
                y=[ratio for _sparsity, ratio in ratios],
                mode="markers",
                name=label,
                legendgroup=label,
                marker={"color": color, "size": 8},
            )
        )
    finite = [value for value in shown if value > 0 and math.isfinite(value)]
    y_axis: dict = {"type": "log", "title": "perplexity / dense"}
    if finite:
        y_axis["range"] = [math.log10(min(finite) / 1.15), math.log10(max(finite) * 1.25)]
    fig.update_layout(
        title=f"{spec.name} SparseGPT normalized ({dataset})",
        xaxis_title="unstructured sparsity",
        yaxis=y_axis,
        template="plotly_white",
    )
    if fit:
        fig.add_annotation(
            text=_law_label(fit),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
        )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_normalized{suffix}.html")
