"""Perplexity, log-log, and normalized Plotly pages for one family.

Each page is unstructured SparseGPT perplexity. The normalized page is
perplexity divided by that model's own dense perplexity, with a dashed line
at 1 and the shared-law curve. Magnitude panels are not written here.

The fit is ln(perplexity / dense) = a · (N / N*)^(b1 log10 s). Only models
with a plausible dense score and every tenth from 0.1 to 0.8 enter it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from analysis.fit import exponent_at, format_n_star, solve_reference_law
from analysis.series import (
    MAX_SPARSITY,
    TENTHS,
    family_sizes,
    load_rows,
    lookup,
    models_for,
    plausible_dense,
)
from experiments.protocol import SweepSpec

COLORS = (
    "#1d4ed8",
    "#0284c7",
    "#0d9488",
    "#7c3aed",
    "#15803d",
    "#65a30d",
    "#ca8a04",
    "#be123c",
    "#0369a1",
)
QWEN_COLORS = {
    "Qwen/Qwen3.5-0.8B": "#1d4ed8",
    "Qwen/Qwen3.5-2B": "#0284c7",
    "Qwen/Qwen3.5-4B": "#0d9488",
    "Qwen/Qwen3.5-9B": "#7c3aed",
    "Qwen/Qwen3.5-27B": "#15803d",
    "Qwen/Qwen3.6-27B": "#65a30d",
    "Qwen/Qwen3.8-27B": "#ca8a04",
}


def refresh_family(spec: SweepSpec, sizes: dict[str, float] | None = None) -> dict | None:
    rows = load_rows(spec)
    if not rows:
        print(f"{spec.name}: no perplexity points", flush=True)
        return None
    if sizes is None:
        sizes = family_sizes(spec)
    datasets = [dataset for dataset in spec.eval_datasets if any(point["dataset"] == dataset for point in rows)]
    extra = sorted({point["dataset"] for point in rows} - set(datasets))
    written = {}
    for dataset in [*datasets, *extra]:
        page = _dataset_pages(spec, rows, dataset, sizes)
        if page is not None:
            written[dataset] = page
    if written:
        path = spec.result_dir / f"{spec.name}_law.json"
        path.write_text(json.dumps(written, indent=2) + "\n")
        print(f"wrote {path}", flush=True)
    return written or None


def _dataset_pages(spec, rows, dataset, sizes) -> dict | None:
    present = []
    for index, model in enumerate(models_for(spec)):
        dense = lookup(rows, model, dataset, 0.0)
        series = _series(rows, model, dataset)
        if dense is None and not series:
            continue
        if not plausible_dense(dense):
            if dense is not None:
                print(
                    f"{spec.name} {dataset}: omit {model.split('/')[-1]} "
                    f"dense perplexity {dense:.1f}",
                    flush=True,
                )
            continue
        present.append((model, _color(model, index), dense, series))
    if not present:
        return None
    fit = _law(present, sizes)
    if fit:
        print(
            f"{spec.name} {dataset}: ln(ppl/dense) = {fit['a']:.4f} * "
            f"(N / {format_n_star(fit['n_star'])})^({fit['b1']:.4f} log10 s)  "
            f"RMSE {fit['rmse']:.4f} over {fit['n']} points",
            flush=True,
        )
    suffix = "" if dataset == "wikitext2" else f"_{dataset}"
    _perplexity(spec, dataset, suffix, present, sizes, fit)
    _loglog(spec, dataset, suffix, present, fit)
    _normalized(spec, dataset, suffix, present, fit)
    return _law_record(fit, present, sizes)


def _series(rows, model, dataset) -> list[tuple[float, float]]:
    points = []
    for point in rows:
        if point["model"] != model or point["dataset"] != dataset:
            continue
        sparsity = float(point["sparsity"])
        if sparsity <= 0 or sparsity > MAX_SPARSITY + 1e-9:
            continue
        points.append((sparsity, float(point["perplexity"])))
    points.sort()
    return points


def _complete(series: list[tuple[float, float]]) -> bool:
    have = {round(sparsity, 5) for sparsity, _ in series}
    return all(any(abs(sparsity - tenth) < 1e-9 for sparsity in have) for tenth in TENTHS)


def _law(present, sizes) -> dict | None:
    sparsities, log_ratios, billions = [], [], []
    usable = []
    for model, _color, dense, series in present:
        n_billions = sizes.get(model)
        if not n_billions or not _complete(series):
            if series and n_billions:
                missing = [
                    tenth
                    for tenth in TENTHS
                    if not any(abs(sparsity - tenth) < 1e-9 for sparsity, _ in series)
                ]
                print(
                    f"  held out of fit: {model.split('/')[-1]} missing {missing}",
                    flush=True,
                )
            continue
        kept = []
        for sparsity, perplexity in series:
            if perplexity <= dense:
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
    if not math.isfinite(n_star) or n_star <= 0:
        return None
    curves = {}
    for model, dense, n_billions, kept in usable:
        curves[model] = {
            "dense": dense,
            "exponent": exponent_at(n_billions, n_star, slope),
            "s_max": max(sparsity for sparsity, _ in kept),
            "n": len(kept),
        }
    return {
        "a": amplitude,
        "n_star": n_star,
        "b1": slope,
        "rmse": rmse,
        "n": len(sparsities),
        "curves": curves,
    }


def _law_record(fit, present, sizes) -> dict:
    if not fit:
        return {"fit": None, "models": [model for model, *_rest in present]}
    models = []
    for model, _color, dense, _series in present:
        curve = fit["curves"].get(model)
        models.append(
            {
                "model": model,
                "stored_b": sizes.get(model),
                "dense": dense,
                "in_fit": curve is not None,
                "exponent": None if curve is None else curve["exponent"],
                "n_points": None if curve is None else curve["n"],
            }
        )
    return {
        "law": "ln(perplexity / dense) = a * (N / N*)^(b1 log10 s)",
        "a": fit["a"],
        "n_star": fit["n_star"],
        "b1": fit["b1"],
        "rmse": fit["rmse"],
        "n_points": fit["n"],
        "models": models,
    }


def _color(model: str, index: int) -> str:
    return QWEN_COLORS.get(model, COLORS[index % len(COLORS)])


def _label(model: str, n_billions: float | None) -> str:
    short = model.split("/")[-1]
    if n_billions:
        return f"{short} ({n_billions:.2f}B)"
    return short


def _law_text(fit: dict) -> str:
    return (
        "ln(ppl / dense) = "
        f"{fit['a']:.2f} · (N / {format_n_star(fit['n_star'])})"
        f"<sup>{fit['b1']:.2f} log10 s</sup>"
        f"<br>RMSE {fit['rmse']:.3f} over {fit['n']} points"
        "<br>N, N* = stored params in billions"
    )


def _curve(fit: dict, model: str):
    curve = fit["curves"].get(model)
    if curve is None:
        return None
    exponent = curve["exponent"]
    if not math.isfinite(exponent):
        return None
    grid = np.linspace(0.02, curve["s_max"], 80)
    predicted = curve["dense"] * np.exp(fit["a"] * np.power(grid, exponent))
    if not np.all(np.isfinite(predicted)):
        return None
    return grid, predicted, exponent


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, path, include_plotlyjs=True, full_html=True)
    print(f"wrote {path}", flush=True)


def _annotate(fig, text: str) -> None:
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=12),
    )


def _perplexity(spec, dataset, suffix, present, sizes, fit) -> None:
    fig = go.Figure()
    shown: list[float] = []
    labeled: list[tuple[float, float]] = []
    for model, color, dense, series in present:
        short = _label(model, sizes.get(model))
        drawn = None if fit is None else _curve(fit, model)
        if drawn is not None:
            grid, predicted, _exponent = drawn
            shown.extend(float(value) for value in predicted)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=predicted,
                    mode="lines",
                    name=f"{short} fit",
                    legendgroup=short,
                    showlegend=False,
                    line={"color": color, "width": 2},
                )
            )
        xs = [0.0, *[sparsity for sparsity, _ in series]]
        ys = [dense, *[perplexity for _, perplexity in series]]
        shown.extend(ys)
        labeled.extend((sparsity, value) for sparsity, value in zip(xs, ys) if sparsity > 0)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers" if drawn is not None else "lines+markers",
                name=short,
                legendgroup=short,
                marker={"color": color, "size": 7},
                line={"color": color, "width": 2},
            )
        )
    if fit:
        _annotate(fig, _law_text(fit))
    _clip_perplexity(fig, shown, labeled)
    fig.update_layout(
        title=f"Unstructured SparseGPT ({dataset})",
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_perplexity{suffix}.html")


def _clip_perplexity(fig, shown, labeled) -> None:
    finite = [value for value in shown if math.isfinite(value) and value > 0]
    if not finite:
        return
    top = max(finite)
    if top > 800:
        in_frame = [value for value in finite if value <= 800]
        y1 = (max(in_frame) if in_frame else 200) * 1.12
        off = [(sparsity, value) for sparsity, value in labeled if value > y1]
        if off:
            note = ", ".join(f"{sparsity:.2g}: {value:.3g}" for sparsity, value in off[:8])
            fig.add_annotation(
                text="Above axis: " + note,
                x=0.0,
                y=1.02,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                xanchor="left",
                font=dict(size=10),
            )
    else:
        y1 = top * 1.12
    fig.update_xaxes(title_text="Sparsity", range=[-0.02, 0.82], tickmode="array", tickvals=[i / 10 for i in range(9)])
    fig.update_yaxes(title_text="Perplexity", range=[0, y1])


def _loglog(spec, dataset, suffix, present, fit) -> None:
    fig = go.Figure()
    measured: list[float] = []
    below = []
    for model, color, dense, series in present:
        short = model.split("/")[-1]
        positive = []
        clamped = []
        for sparsity, perplexity in series:
            if perplexity <= 0 or not math.isfinite(perplexity):
                continue
            ratio = math.log(perplexity / dense)
            if ratio > 0:
                positive.append((sparsity, ratio))
            else:
                clamped.append((sparsity, perplexity))
        drawn = None if fit is None else _curve(fit, model)
        if drawn is not None:
            grid, predicted, exponent = drawn
            log_ratio = fit["a"] * np.power(grid, exponent)
            mask = np.isfinite(log_ratio) & (log_ratio > 0)
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=grid[mask],
                        y=log_ratio[mask],
                        mode="lines",
                        name=f"{short} fit",
                        showlegend=False,
                        line={"color": color, "width": 2},
                    )
                )
        if positive:
            xs, ys = zip(*positive)
            measured.extend(ys)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=short,
                    showlegend=False,
                    marker={"color": color, "size": 8},
                )
            )
        if clamped:
            below.append((short, color, dense, clamped))
    finite = [value for value in measured if value > 0 and math.isfinite(value)]
    low = high = None
    if finite:
        low = min(finite) / 1.8
        high = max(finite) * 1.8
    if below and low is not None:
        for short, color, dense, clamped in below:
            xs, measured_ppl = zip(*clamped)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[low] * len(xs),
                    mode="markers",
                    name=short,
                    showlegend=False,
                    marker={"color": color, "size": 9, "symbol": "diamond-open", "line": {"width": 1.5, "color": color}},
                )
            )
    note = "Open diamonds are at or below that model's dense perplexity."
    if fit:
        note = _law_text(fit) + "<br>" + note
    _annotate(fig, note)
    fig.update_xaxes(
        type="log",
        title_text="Sparsity",
        range=[math.log10(0.08), math.log10(0.9)],
        tickmode="array",
        tickvals=[0.1, 0.2, 0.4, 0.8],
    )
    y_axis = {"type": "log", "title_text": "ln(perplexity / dense)"}
    if low is not None and high is not None:
        y_axis["range"] = [math.log10(low), math.log10(high)]
    fig.update_yaxes(**y_axis)
    fig.update_layout(
        title=f"Unstructured SparseGPT, log-log ({dataset})",
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        showlegend=False,
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_loglog{suffix}.html")


def _normalized(spec, dataset, suffix, present, fit) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-0.02, 0.82],
            y=[1.0, 1.0],
            mode="lines",
            name="Dense",
            line={"color": "#666", "width": 1, "dash": "dash"},
        )
    )
    shown = [1.0]
    for model, color, dense, series in present:
        short = model.split("/")[-1]
        rows = [(sparsity, perplexity / dense) for sparsity, perplexity in series if perplexity > 0]
        if not rows:
            continue
        drawn = None if fit is None else _curve(fit, model)
        slope = ""
        if drawn is not None:
            grid, predicted, exponent = drawn
            norm = predicted / dense
            shown.extend(float(value) for value in norm if math.isfinite(value) and value > 0)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=norm,
                    mode="lines",
                    name=f"{short} fit",
                    legendgroup=short,
                    showlegend=False,
                    line={"color": color, "width": 2},
                )
            )
            slope = f"  b={exponent:.2f}"
        xs, ys = zip(*rows)
        shown.extend(ys)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=short + slope,
                legendgroup=short,
                marker={"color": color, "size": 8},
            )
        )
    finite = [value for value in shown if value > 0 and math.isfinite(value)]
    if finite:
        fig.update_yaxes(range=[math.log10(min(finite) / 1.15), math.log10(max(finite) * 1.25)])
    note = "Each series is perplexity / that model's own dense perplexity."
    if fit:
        note += "<br>" + _law_text(fit)
    _annotate(fig, note)
    fig.update_xaxes(
        title_text="Sparsity",
        range=[-0.02, 0.82],
        tickmode="array",
        tickvals=[i / 10 for i in range(9)],
    )
    fig.update_yaxes(type="log", title_text="Perplexity / own dense perplexity")
    fig.update_layout(
        title=f"Unstructured SparseGPT, normalized by own dense ({dataset})",
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )
    _write(fig, spec.result_dir / f"{spec.name}_sparsity_normalized{suffix}.html")


def main() -> None:
    from experiments.sweeps import sweeps

    print(
        "gemma-it WikiText2-cal dense scores are in the hundreds or thousands; "
        "left out of the figures and the fit",
        flush=True,
    )
    for spec in sweeps():
        refresh_family(spec)
        from analysis.principle import write_family_page

        write_family_page(spec)


if __name__ == "__main__":
    main()
