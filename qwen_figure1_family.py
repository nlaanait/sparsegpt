"""Figure 1 for each Qwen3.5 model: raw-WikiText2 perplexity vs sparsity.

Magnitude pruning stops after the first point above 100 perplexity.
Unstructured SparseGPT stops after the first point above 200.
Calibration is 128 C4 segments. Finished points are reused, including the
Qwen3.5-2B sweep and the dense / 50% numbers from the family scaling run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]
SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Half-steps on the rising part of the log-log curve. Kept off SPARSITIES so
# the full sweep does not pick them up, and so main() does not resume 27B.
UNSTRUCTURED_HALF_STEPS = {
    "Qwen/Qwen3.5-0.8B": [0.45, 0.55, 0.65],
    "Qwen/Qwen3.5-2B": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    "Qwen/Qwen3.5-4B": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    "Qwen/Qwen3.5-9B": [0.45, 0.55, 0.65, 0.75],
}
MAGNITUDE_STOP = 100.0
SPARSEGPT_STOP = 200.0
PPL_RE = re.compile(
    r"Perplexity:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?|inf|nan)",
    re.I,
)
FIGURE1 = ROOT / "figures" / "qwen3.5-2b_wikitext2.json"
FIGURE2 = ROOT / "figures" / "qwen3.5_family_wikitext2.json"


def result_path(output: Path) -> Path:
    return output / "qwen3.5_family_sparsity_wikitext2.json"


def load_results(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"eval": "raw-WikiText2", "calibration": "c4", "points": []}


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(path)


def lookup(results: dict, model: str, method: str, sparsity: float) -> float | None:
    for point in results["points"]:
        if (
            point["model"] == model
            and point["method"] == method
            and abs(point["sparsity"] - sparsity) < 1e-9
        ):
            return point["perplexity"]
    return None


def add_point(results: dict, model: str, method: str, sparsity: float, perplexity: float) -> None:
    if lookup(results, model, method, sparsity) is None and math.isfinite(perplexity):
        results["points"].append(
            {
                "model": model,
                "method": method,
                "sparsity": sparsity,
                "perplexity": perplexity,
            }
        )


def seed_known_points(results: dict) -> None:
    if FIGURE1.is_file():
        data = json.loads(FIGURE1.read_text())
        for point in data["points"]:
            add_point(
                results,
                "Qwen/Qwen3.5-2B",
                point["method"],
                point["sparsity"],
                point["perplexity"],
            )
    if FIGURE2.is_file():
        data = json.loads(FIGURE2.read_text())
        for point in data["points"]:
            if point["method"] == "Dense":
                add_point(results, point["model"], "dense", 0.0, point["perplexity"])
            elif point["method"] == "50% Unstructured":
                add_point(results, point["model"], "sparsegpt", 0.5, point["perplexity"])


def sparsity_token(sparsity: float) -> str:
    """CLI token that keeps half-steps (0.15) distinct from tenths (0.1)."""
    return f"{sparsity:.2f}".rstrip("0").rstrip(".")


def run_qwen(model: str, sparsity: float, gmp: bool, log_path: Path) -> float:
    short = model.split("/")[-1].lower()
    cmd = [
        sys.executable,
        "-m",
        "sparsegpt.qwen",
        model,
        "c4",
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        "wikitext2",
    ]
    if sparsity > 0 and not gmp:
        cmd.extend(["--save", f"sparse_opt/{short}"])
    if gmp:
        cmd.append("--gmp")
    print("+", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            lines.append(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"qwen.py exited {code}. Log: {log_path}")
    matches = PPL_RE.findall("".join(lines))
    if not matches:
        raise RuntimeError(f"No perplexity line in {log_path}")
    return float(matches[-1])


def stored_sparsities(results: dict, model: str, method: str) -> list[float]:
    """Every sparsity saved for this series, including half-steps such as 0.15."""
    values = [
        float(point["sparsity"])
        for point in results["points"]
        if point["model"] == model
        and point["method"] == method
        and float(point["sparsity"]) > 0
    ]
    values.sort()
    unique: list[float] = []
    for value in values:
        if not unique or abs(unique[-1] - value) > 1e-9:
            unique.append(value)
    return unique


def stopped_series(
    results: dict, model: str, method: str, stop: float
) -> list[tuple[float, float]]:
    """Saved points for one series, in sparsity order.

    ``stop`` is the sweep's early-exit threshold. Points already saved past it
    stay on the figure. A half-step such as 2B at 0.75 can cross the threshold
    while 0.8 was measured earlier and must still be drawn.
    """
    del stop
    rows = []
    for sparsity in stored_sparsities(results, model, method):
        perplexity = lookup(results, model, method, sparsity)
        if perplexity is None or not math.isfinite(perplexity):
            continue
        rows.append((sparsity, perplexity))
    return rows


_BLUE = (33, 102, 172)
_RED = (178, 24, 43)


def _mix(t: float) -> str:
    rgb = tuple(round(_BLUE[i] + (_RED[i] - _BLUE[i]) * t) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


# Same parameter count as Qwen3.5-27B. A partial grid is drawn from the points
# on disk and kept out of the Qwen3.5 JSON. Purple and teal so they stay
# visible against the 27B red.
_EXTRA_COLOR = {
    "Qwen/Qwen3.6-27B": "#6d28d9",
    "Qwen/Qwen3.8-27B": "#0f766e",
}


def model_colors() -> dict[str, str]:
    """Blue for the smallest model, red for the largest, on a log parameter scale."""
    sizes: dict[str, float] = {}
    if FIGURE2.is_file():
        sizes = json.loads(FIGURE2.read_text()).get("params_b") or {}
    known = [(model, sizes[model]) for model in MODELS if model in sizes and sizes[model] > 0]
    if len(known) < 2:
        count = max(len(MODELS) - 1, 1)
        colors = {model: _mix(index / count) for index, model in enumerate(MODELS)}
    else:
        logs = [math.log10(billions) for _, billions in known]
        low, high = min(logs), max(logs)
        span = high - low or 1.0
        colors = {model: _mix((math.log10(billions) - low) / span) for model, billions in known}
    colors.update(_EXTRA_COLOR)
    return colors


def _param_billions(models: list[str] | None = None) -> dict[str, float]:
    models = list(MODELS if models is None else models)
    sizes: dict[str, float] = {}
    if FIGURE2.is_file():
        raw = json.loads(FIGURE2.read_text()).get("params_b") or {}
        sizes = {model: float(raw[model]) for model in models if raw.get(model, 0) > 0}
    from sparsity_principle import EXTRA_PARAMS_B

    for model in models:
        sizes.setdefault(model, EXTRA_PARAMS_B)
    return {model: sizes[model] for model in models if model in sizes and sizes[model] > 0}


def _with_ready_extras(results: dict, tag: str) -> dict:
    """Copy saved Qwen3.6 and Qwen3.8 points without writing the family JSON."""
    from sparsity_principle import EXTRA_FILES, ready_extra_models

    dataset = "ptb" if tag == "ptb" else "wikitext2"
    points = list(results["points"])
    present = {(point["model"], point["method"], float(point["sparsity"])) for point in points}
    for model in ready_extra_models():
        path = EXTRA_FILES[model]
        for point in json.loads(path.read_text()).get("points", []):
            if point.get("dataset") != dataset or point.get("model") != model:
                continue
            if point["method"] not in ("dense", "sparsegpt"):
                continue
            key = (model, point["method"], float(point["sparsity"]))
            if key in present:
                continue
            points.append(
                {
                    "model": model,
                    "method": point["method"],
                    "sparsity": float(point["sparsity"]),
                    "perplexity": float(point["perplexity"]),
                }
            )
            present.add(key)
    merged = dict(results)
    merged["points"] = points
    return merged


def series_models(results: dict) -> list[str]:
    present = {point["model"] for point in results["points"]}
    return [model for model in [*MODELS, *_EXTRA_COLOR] if model in present]


def sparsegpt_exponent_fit(results: dict) -> dict | None:
    """ln(ppl / dense) = a * (N / N*)^(b1 * log10 sparsity).

    One a, N*, and b1 for every model. N and N* are parameter counts in
    billions. Each drawn curve is a * s^(b1 log10(N/N*)). Only unstructured
    SparseGPT points above that model's own dense perplexity enter the fit.
    """
    import numpy as np

    models = series_models(results)
    sizes = _param_billions(models)
    usable = []
    sparsities, log_ratios, billions = [], [], []
    for model in models:
        dense = lookup(results, model, "dense", 0.0)
        if model not in sizes or dense is None or not math.isfinite(dense) or dense <= 0:
            continue
        rows = []
        for sparsity in stored_sparsities(results, model, "sparsegpt"):
            perplexity = lookup(results, model, "sparsegpt", sparsity)
            if perplexity is None or not math.isfinite(perplexity) or perplexity <= dense:
                continue
            rows.append((sparsity, perplexity))
            sparsities.append(sparsity)
            log_ratios.append(math.log(perplexity / dense))
            billions.append(sizes[model])
        if rows:
            usable.append((model, dense, sizes[model], rows))
    if len(usable) < 2 or len(sparsities) < 4:
        return None

    from analysis.fit import solve_reference_law

    amplitude, n_star, slope, rmse = solve_reference_law(
        np.asarray(sparsities, dtype=float),
        np.asarray(log_ratios, dtype=float),
        np.asarray(billions, dtype=float),
    )
    print(
        f"shared law ln(ppl/dense) = {amplitude:.6f} * (N / {n_star:.6f})"
        f"^( {slope:.6f} log10 s )  RMSE {rmse:.4f} over {len(sparsities)} points",
        flush=True,
    )
    curves = {}
    for model, dense, n_billions, rows in usable:
        curves[model] = {
            "a": amplitude,
            "b": slope * math.log10(n_billions / n_star),
            "dense": dense,
            "s_max": max(sparsity for sparsity, _ in rows),
        }
    return {
        "a": amplitude,
        "b1": slope,
        "n_star": n_star,
        "curves": curves,
    }


def shared_law_label(fit: dict) -> str:
    """Fitted reference-size law. N and N* are billions of parameters."""
    return (
        "ln(ppl / dense) = "
        f"{fit['a']:.2f} · (N / {fit['n_star']:.3f})"
        f"<sup>{fit['b1']:.2f} log10 s</sup>"
        "<br>N, N* = #params in billions"
    )


def _fit_hover(short: str, curve: dict, fit: dict, value_name: str) -> str:
    return (
        f"{short} fit<br>a={fit['a']:.2f}, N*={fit['n_star']:.3f}B, b1={fit['b1']:.2f}"
        f"<br>exponent here={curve['b']:.2f}"
        f"<br>sparsity=%{{x:.2f}}<br>{value_name}=%{{y:.3g}}<extra></extra>"
    )


def _style_panel(
    fig,
    shown: list[float],
    labeled: list[tuple[float, float]],
    title: str,
    y_title: str = "Perplexity on raw-WikiText2",
) -> None:
    finite = [value for value in shown if math.isfinite(value)]
    if not finite:
        return
    top = max(finite)
    if top > 800:
        in_frame = [value for value in finite if value <= 800]
        y1 = (max(in_frame) if in_frame else 200) * 1.12
        off = [(sparsity, value) for sparsity, value in labeled if value > y1]
        if off:
            note = ", ".join(f"{sparsity:.1f}: {value:.3g}" for sparsity, value in off)
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
    fig.update_yaxes(title_text=y_title, range=[0, y1])
    fig.update_xaxes(
        title_text="Sparsity",
        range=[-0.02, 0.82],
        tickmode="array",
        tickvals=[i / 10 for i in range(9)],
    )
    fig.update_layout(
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        title=dict(text=title, x=0),
        legend=dict(
            title="Model",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
    )


def plot_results(
    results: dict,
    output: Path,
    *,
    y_title: str = "Perplexity on raw-WikiText2",
    tag: str = "",
    methods: tuple[str, ...] = ("sparsegpt", "magnitude"),
) -> Path:
    import numpy as np
    import plotly.graph_objects as go

    results = _with_ready_extras(results, tag)
    colors = model_colors()
    try:
        fit = sparsegpt_exponent_fit(results)
    except Exception as exc:
        print(f"power-law fit skipped: {exc}", flush=True)
        fit = None
    suffix = f"_{tag}" if tag else ""
    dataset = tag.upper() if tag else ""
    sparse_title = "Unstructured SparseGPT" + (f" on {dataset}" if dataset else "")
    magnitude_title = "Magnitude" + (f" on {dataset}" if dataset else "")
    panels = (
        ("sparsegpt", SPARSEGPT_STOP, sparse_title, f"qwen3.5_family_sparsity_perplexity{suffix}.html"),
        ("magnitude", MAGNITUDE_STOP, magnitude_title, f"qwen3.5_family_magnitude_perplexity{suffix}.html"),
    )
    written = []
    for method, stop, title, filename in panels:
        fig = go.Figure()
        shown: list[float] = []
        labeled: list[tuple[float, float]] = []
        if method not in methods:
            continue
        for model in series_models(results):
            series = stopped_series(results, model, method, stop)
            dense = lookup(results, model, "dense", 0.0)
            if dense is not None and math.isfinite(dense):
                series = [(0.0, dense)] + series
            if not series:
                continue
            xs, ys = zip(*series)
            shown.extend(ys)
            labeled.extend((sparsity, value) for sparsity, value in series if sparsity > 0)
            short = model.split("/")[-1]
            color = colors.get(model, "#333")
            curve = None if fit is None or method != "sparsegpt" else fit["curves"].get(model)
            if dense is not None and math.isfinite(dense):
                fig.add_trace(
                    go.Scatter(
                        x=[-0.02, 0.82],
                        y=[dense, dense],
                        mode="lines",
                        name=f"{short} dense",
                        legendgroup=short,
                        showlegend=False,
                        hovertemplate=f"{short} dense: %{{y:.3f}}<extra></extra>",
                        line=dict(color=color, width=1, dash="dash"),
                    )
                )
            if curve is not None:
                grid = np.linspace(0.0, curve["s_max"], 80)
                predicted = curve["dense"] * np.exp(curve["a"] * np.power(grid, curve["b"]))
                shown.extend(float(value) for value in predicted)
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=predicted,
                        mode="lines",
                        name=f"{short} fit",
                        legendgroup=short,
                        showlegend=False,
                        hovertemplate=_fit_hover(short, curve, fit, "perplexity"),
                        line=dict(color=color, width=2),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers" if curve is not None else "lines+markers",
                    name=short,
                    legendgroup=short,
                    showlegend=True,
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color),
                )
            )
        if method == "sparsegpt" and fit is not None:
            fig.add_annotation(
                text=shared_law_label(fit),
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
        if not labeled:
            continue
        _style_panel(fig, shown, labeled, title, y_title)
        output.mkdir(parents=True, exist_ok=True)
        html = output / filename
        fig.write_html(html, include_plotlyjs=True)
        print(f"Wrote {html}")
        written.append(html)
    written.append(plot_loglog(results, output, colors, fit, tag=tag))
    written.append(plot_normalized(results, output, colors, tag=tag))
    return written[0]


def plot_loglog(
    results: dict,
    output: Path,
    colors: dict[str, str],
    fit: dict | None,
    *,
    tag: str = "",
) -> Path:
    """ln(perplexity / dense) versus sparsity, both axes logarithmic.

    Positive log ratios stay circles. A non-positive ratio is below that model's
    own dense perplexity, so ln is 0 or negative and cannot sit on a log axis.
    Those points are drawn as open diamonds on the bottom edge of the y-range.
    The JSON perplexities are left unchanged.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    measured: list[float] = []
    below_dense: list[tuple[str, str, float, list[tuple[float, float]]]] = []
    for model in series_models(results):
        dense = lookup(results, model, "dense", 0.0)
        if dense is None or not math.isfinite(dense) or dense <= 0:
            continue
        rows = []
        clamped = []
        for sparsity, perplexity in stopped_series(results, model, "sparsegpt", SPARSEGPT_STOP):
            if perplexity <= 0 or not math.isfinite(perplexity):
                continue
            log_ratio = math.log(perplexity / dense)
            if log_ratio > 0:
                rows.append((sparsity, log_ratio))
            else:
                clamped.append((sparsity, perplexity))
        short = model.split("/")[-1]
        color = colors.get(model, "#333")
        curve = None if fit is None else fit["curves"].get(model)
        if curve is not None:
            grid = np.geomspace(0.1, 0.8, 60)
            predicted = curve["a"] * np.power(grid, curve["b"])
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=predicted,
                    mode="lines",
                    name=f"{short} fit",
                    showlegend=False,
                    hovertemplate=_fit_hover(short, curve, fit, "ln(ppl/dense)"),
                    line=dict(color=color, width=2),
                )
            )
        if rows:
            xs, ys = zip(*rows)
            measured.extend(ys)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=short,
                    showlegend=False,
                    marker=dict(size=8, color=color),
                    hovertemplate=(
                        f"{short}<br>sparsity=%{{x:.2f}}<br>ln(ppl/dense)=%{{y:.3g}}<extra></extra>"
                    ),
                )
            )
        if clamped:
            below_dense.append((short, color, dense, clamped))
    finite = [value for value in measured if value > 0 and math.isfinite(value)]
    low = high = None
    if finite:
        low = min(finite) / 1.8
        high = max(finite) * 1.8
    if below_dense and low is not None:
        for short, color, dense, clamped in below_dense:
            xs, measured_ppl = zip(*clamped)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[low] * len(xs),
                    mode="markers",
                    name=short,
                    showlegend=False,
                    marker=dict(
                        size=9,
                        color=color,
                        symbol="diamond-open",
                        line=dict(width=1.5, color=color),
                    ),
                    customdata=[[ppl, dense] for ppl in measured_ppl],
                    hovertemplate=(
                        f"{short}<br>sparsity=%{{x:.2f}}"
                        "<br>measured below dense"
                        "<br>measured perplexity=%{customdata[0]:.6f}"
                        "<br>plotted value reset to dense (%{customdata[1]:.6f})"
                        "<extra></extra>"
                    ),
                )
            )
    note = "" if fit is None else shared_law_label(fit)
    floor_clause = "Open diamonds were below dense and are drawn at the dense floor."
    note = f"{note}<br>{floor_clause}" if note else floor_clause
    fig.add_annotation(
        text=note,
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
    fig.update_xaxes(
        type="log",
        title_text="Sparsity",
        range=[math.log10(0.08), math.log10(0.9)],
        tickmode="array",
        tickvals=[0.1, 0.2, 0.4, 0.8],
    )
    y_axis = dict(type="log", title_text="ln(perplexity / dense)")
    if low is not None and high is not None:
        y_axis["range"] = [math.log10(low), math.log10(high)]
    fig.update_yaxes(**y_axis)
    fig.update_layout(
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        showlegend=False,
        title=dict(
            text="Unstructured SparseGPT, log-log"
            + (f" ({tag.upper()})" if tag else ""),
            x=0,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    html = output / f"qwen3.5_family_sparsity_loglog{suffix}.html"
    fig.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")
    return html


def per_model_exponent_fits(results: dict) -> dict[str, dict]:
    """Fit ln(ppl / dense) = a * sparsity ** b separately for each model.

    Only points above that model's own dense perplexity are used. The form is
    positive, so a point below dense cannot lie on the curve.
    """
    import numpy as np
    from scipy.optimize import least_squares

    fits = {}
    for model in series_models(results):
        dense = lookup(results, model, "dense", 0.0)
        if dense is None or not math.isfinite(dense) or dense <= 0:
            continue
        rows = []
        for sparsity in stored_sparsities(results, model, "sparsegpt"):
            perplexity = lookup(results, model, "sparsegpt", sparsity)
            if perplexity is not None and math.isfinite(perplexity) and perplexity > dense:
                rows.append((sparsity, perplexity))
        if len(rows) < 4:
            continue
        sparsities = np.asarray([sparsity for sparsity, _ in rows])
        perplexities = np.asarray([perplexity for _, perplexity in rows])

        def residual(theta: np.ndarray, sparsities=sparsities, perplexities=perplexities, dense=dense) -> np.ndarray:
            predicted = dense * np.exp(theta[0] * np.power(sparsities, theta[1]))
            return np.log(predicted) - np.log(perplexities)

        fitted = least_squares(residual, np.array([12.0, 4.0]), bounds=([0.0, 0.2], [1e4, 20.0]))
        fits[model] = {
            "a": float(fitted.x[0]),
            "b": float(fitted.x[1]),
            "dense": dense,
            "s_max": max(sparsity for sparsity, _ in rows),
        }
    return fits


def plot_normalized(
    results: dict, output: Path, colors: dict[str, str], *, tag: str = ""
) -> Path:
    """Perplexity divided by each model's own dense perplexity."""
    import numpy as np
    import plotly.graph_objects as go

    try:
        fit = sparsegpt_exponent_fit(results)
    except Exception as exc:
        print(f"shared fit skipped: {exc}", flush=True)
        fit = None
    curves = {} if fit is None else fit["curves"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-0.02, 0.82],
            y=[1.0, 1.0],
            mode="lines",
            name="Dense",
            showlegend=True,
            hovertemplate="dense: 1<extra></extra>",
            line=dict(color="#666", width=1, dash="dash"),
        )
    )
    shown: list[float] = [1.0]
    for model in series_models(results):
        dense = lookup(results, model, "dense", 0.0)
        series = stopped_series(results, model, "sparsegpt", SPARSEGPT_STOP)
        if dense is None or not math.isfinite(dense) or dense <= 0 or not series:
            continue
        rows = [(sparsity, perplexity / dense) for sparsity, perplexity in series if perplexity > 0]
        if not rows:
            continue
        short = model.split("/")[-1]
        color = colors.get(model, "#333")
        curve = curves.get(model)
        slope = "" if curve is None else f"  b={curve['b']:.2f}"
        if curve is not None:
            grid = np.linspace(0.0, curve["s_max"], 80)
            predicted = np.exp(curve["a"] * np.power(grid, curve["b"]))
            shown.extend(float(value) for value in predicted if math.isfinite(value))
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=predicted,
                    mode="lines",
                    name=f"{short} fit",
                    legendgroup=short,
                    showlegend=False,
                    hovertemplate=_fit_hover(short, curve, fit, "ppl/dense"),
                    line=dict(color=color, width=2),
                )
            )
        xs, ys = zip(*rows)
        shown.extend(ys)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=short + slope,
                legendgroup=short,
                showlegend=True,
                marker=dict(size=8, color=color),
                hovertemplate=(
                    f"{short}<br>sparsity=%{{x:.2f}}<br>ppl/dense=%{{y:.3f}}<extra></extra>"
                ),
            )
        )
    finite = [value for value in shown if value > 0 and math.isfinite(value)]
    if finite:
        fig.update_yaxes(range=[math.log10(min(finite) / 1.15), math.log10(max(finite) * 1.25)])
    note = "Each series is perplexity / that model's own dense perplexity."
    if fit is not None:
        note += "<br>" + shared_law_label(fit)
    fig.add_annotation(
        text=note,
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
    fig.update_xaxes(
        title_text="Sparsity",
        range=[-0.02, 0.82],
        tickmode="array",
        tickvals=[i / 10 for i in range(9)],
    )
    fig.update_yaxes(type="log", title_text="Perplexity / own dense perplexity")
    fig.update_layout(
        template="plotly_white",
        width=760,
        height=480,
        margin=dict(t=56, b=48, l=72, r=24),
        title=dict(
            text="Unstructured SparseGPT, normalized by own dense"
            + (f" ({tag.upper()})" if tag else ""),
            x=0,
        ),
        legend=dict(
            title="Model",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    html = output / f"qwen3.5_family_sparsity_normalized{suffix}.html"
    fig.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")
    return html


def _same_sparsity(left: float, right: float) -> bool:
    return abs(left - right) < 1e-9


def sweep_unstructured_half_steps(results: dict, path: Path, output: Path) -> None:
    """Unstructured SparseGPT half-steps for 0.8B, 2B, 4B, and 9B.

    Existing grid points are visited only so a perplexity already above
    SPARSEGPT_STOP ends that model before a higher half-step. Magnitude
    pruning and Qwen3.5-27B are not run.
    """
    logs = output / "logs"
    for model, extras in UNSTRUCTURED_HALF_STEPS.items():
        order = sorted(set(SPARSITIES).union(extras))
        for sparsity in order:
            existing = lookup(results, model, "sparsegpt", sparsity)
            token = sparsity_token(sparsity)
            is_extra = any(_same_sparsity(sparsity, extra) for extra in extras)
            if not is_extra:
                if existing is not None and existing > SPARSEGPT_STOP:
                    print(
                        f"stop {model} sparsegpt: perplexity {existing:.1f} > {SPARSEGPT_STOP:.0f}",
                        flush=True,
                    )
                    break
                continue
            if existing is not None:
                print(
                    f"skip {model} sparsegpt sparsity={token} perplexity={existing:.4f}",
                    flush=True,
                )
                if existing > SPARSEGPT_STOP:
                    print(
                        f"stop {model} sparsegpt: perplexity {existing:.1f} > {SPARSEGPT_STOP:.0f}",
                        flush=True,
                    )
                    break
                continue
            short = model.split("/")[-1].lower()
            log_path = logs / f"{short}_sparsegpt_sparsity{token}.log"
            try:
                perplexity = run_qwen(model, sparsity, False, log_path)
            except RuntimeError as exc:
                print(f"failed {model} sparsegpt sparsity={token}: {exc}", flush=True)
                break
            add_point(results, model, "sparsegpt", sparsity, perplexity)
            save_results(path, results)
            print(
                f"{model} sparsegpt sparsity={token} perplexity={perplexity:.4f}",
                flush=True,
            )
            try:
                plot_results(results, output)
            except ModuleNotFoundError:
                print("plotly is not installed; results were saved, plot skipped")
            if perplexity > SPARSEGPT_STOP:
                print(
                    f"stop {model} sparsegpt: perplexity {perplexity:.1f} > {SPARSEGPT_STOP:.0f}",
                    flush=True,
                )
                break


def sweep_method(
    results: dict,
    path: Path,
    output: Path,
    model: str,
    method: str,
    gmp: bool,
    stop: float,
) -> None:
    logs = output / "logs"
    short = model.split("/")[-1].lower()
    for sparsity in SPARSITIES:
        existing = lookup(results, model, method, sparsity)
        if existing is not None:
            print(f"skip {model} {method} sparsity={sparsity:.1f} perplexity={existing:.4f}", flush=True)
            if existing > stop:
                print(f"stop {model} {method}: perplexity {existing:.1f} > {stop:.0f}", flush=True)
                return
            continue
        log_path = logs / f"{short}_{method}_sparsity{sparsity:.1f}.log"
        try:
            perplexity = run_qwen(model, sparsity, gmp, log_path)
        except RuntimeError as exc:
            print(f"failed {model} {method} sparsity={sparsity:.1f}: {exc}", flush=True)
            return
        add_point(results, model, method, sparsity, perplexity)
        save_results(path, results)
        print(f"{model} {method} sparsity={sparsity:.1f} perplexity={perplexity:.4f}", flush=True)
        try:
            plot_results(results, output)
        except ModuleNotFoundError:
            print("plotly is not installed; results were saved, plot skipped")
        if perplexity > stop:
            print(f"stop {model} {method}: perplexity {perplexity:.1f} > {stop:.0f}", flush=True)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--half-steps",
        action="store_true",
        help="Unstructured SparseGPT half-steps for 0.8B, 2B, 4B, and 9B only.",
    )
    args = parser.parse_args()
    path = result_path(args.output)
    results = load_results(path)
    results.setdefault("points", [])
    if args.half_steps and not args.plot_only:
        seed_known_points(results)
        save_results(path, results)
        sweep_unstructured_half_steps(results, path, args.output)
    elif not args.plot_only:
        seed_known_points(results)
        save_results(path, results)
        for model in MODELS:
            if lookup(results, model, "dense", 0.0) is None:
                log_path = args.output / "logs" / f"{model.split('/')[-1].lower()}_dense.log"
                try:
                    perplexity = run_qwen(model, 0.0, False, log_path)
                except RuntimeError as exc:
                    print(f"failed {model} dense: {exc}", flush=True)
                    continue
                add_point(results, model, "dense", 0.0, perplexity)
                save_results(path, results)
                print(f"{model} dense perplexity={perplexity:.4f}", flush=True)
            sweep_method(results, path, args.output, model, "magnitude", True, MAGNITUDE_STOP)
            sweep_method(results, path, args.output, model, "sparsegpt", False, SPARSEGPT_STOP)
    if not results["points"]:
        raise SystemExit(f"No perplexities in {path}")
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
