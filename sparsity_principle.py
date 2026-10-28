"""Refit the shared sparsity law on WikiText2 and PTB and rewrite the page.

Points at or below each model's own dense perplexity are left out. The chosen
form is the scale-free law α · N^β · s^b, fit separately on each eval. Other
rows in the comparison table are alternatives, including the older
a · (N / N*)^(b1 log10 s) reference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PAGE = FIGURES / "sparsity_principle.html"
SIZES_PATH = FIGURES / "qwen3.5_family_wikitext2.json"
WIKI_PATH = FIGURES / "qwen3.5_family_sparsity_wikitext2.json"
PTB_PATH = FIGURES / "qwen3.5_family_sparsity_ptb.json"
BUNDLE_SOURCE = FIGURES / "qwen3.5_family_sparsity_perplexity.html"

ORDER = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]
SHORT = {
    "Qwen/Qwen3.5-0.8B": "0.8B",
    "Qwen/Qwen3.5-2B": "2B",
    "Qwen/Qwen3.5-4B": "4B",
    "Qwen/Qwen3.5-9B": "9B",
    "Qwen/Qwen3.5-27B": "27B",
}
# Same bf16 parameter count as Qwen3.5-27B (shard bytes / 2).
EXTRA_PARAMS_B = 27.781427952
EXTRA_FILES = {
    "Qwen/Qwen3.6-27B": FIGURES / "qwen3.6-27b_sparsity.json",
    "Qwen/Qwen3.8-27B": FIGURES / "qwen3.8-27b_sparsity.json",
}
SWEEP_SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MODEL_COLORS = {
    "Qwen/Qwen3.5-0.8B": "#1d4ed8",
    "Qwen/Qwen3.5-2B": "#0284c7",
    "Qwen/Qwen3.5-4B": "#0d9488",
    "Qwen/Qwen3.5-9B": "#7c3aed",
    "Qwen/Qwen3.5-27B": "#15803d",
    "Qwen/Qwen3.6-27B": "#65a30d",
    "Qwen/Qwen3.8-27B": "#ca8a04",
}
FORMS = (
    ("power", "α · N^β · s^b", "3", True),
    ("ref", "a · (N / N*)^(b1 log10 s)", "3", False),
    ("slog", "a · s^(b0 + b1 log10 N)", "3", False),
    ("slog0", "a · s^(b log10 N)", "2", False),
    ("log1s", "a · (−log(1−s))^b · N^c", "3", False),
    ("exp", "a · exp(b · s · N^c)", "3", False),
    ("nosize", "a · s^b", "2", False),
)


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(pred - y)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit(residual, guesses, bounds):
    best = None
    for guess in guesses:
        fitted = least_squares(
            residual, np.asarray(guess, dtype=float), bounds=bounds, max_nfev=4000
        )
        err = _rmse(residual(fitted.x))
        if best is None or err < best[0]:
            best = (err, fitted.x.copy())
    return best


def _has_sparsity(values: set[float], sparsity: float) -> bool:
    return any(abs(value - sparsity) < 1e-9 for value in values)


def _saved_sparsities(model: str) -> dict[str, set[float]]:
    saved = {"ptb": set(), "wikitext2": set()}
    path = EXTRA_FILES.get(model)
    if path is None or not path.is_file():
        return saved
    for point in json.loads(path.read_text()).get("points", []):
        if point.get("model") != model or point.get("method") not in ("dense", "sparsegpt"):
            continue
        dataset = point.get("dataset")
        if dataset in saved:
            saved[dataset].add(float(point["sparsity"]))
    return saved


def ready_extra_models() -> list[str]:
    """Extras with saved dense or SparseGPT points, including a partial grid."""
    ready = []
    for model, path in EXTRA_FILES.items():
        if not path.is_file():
            continue
        saved = _saved_sparsities(model)
        if saved["ptb"] or saved["wikitext2"]:
            ready.append(model)
    return ready


def _partial_sentence(models: list[str]) -> str:
    """Say which extra sparsities are not on disk yet."""
    notes = []
    for model in models:
        if model not in EXTRA_FILES:
            continue
        saved = _saved_sparsities(model)
        gaps = []
        for dataset, label in (("wikitext2", "WikiText2"), ("ptb", "PTB")):
            missing = [sparsity for sparsity in SWEEP_SPARSITIES if not _has_sparsity(saved[dataset], sparsity)]
            if missing:
                gaps.append((label, missing))
        if not gaps:
            continue
        name = model.split("/")[-1]
        if len({tuple(missing) for _, missing in gaps}) == 1:
            missing = gaps[0][1]
            where = " and ".join(label for label, _ in gaps)
            listed = ", ".join(f"{sparsity:.0%}" for sparsity in missing)
            notes.append(f"{name} is partial on {where}: still missing {listed}.")
        else:
            parts = []
            for label, missing in gaps:
                listed = ", ".join(f"{sparsity:.0%}" for sparsity in missing)
                parts.append(f"{label} missing {listed}")
            notes.append(f"{name} is partial: {'; '.join(parts)}.")
    return " ".join(notes)


def model_order() -> list[str]:
    return [*ORDER, *ready_extra_models()]


def load_sizes(models: list[str]) -> dict[str, float]:
    sizes = {
        key: float(value)
        for key, value in json.loads(SIZES_PATH.read_text())["params_b"].items()
    }
    for model in models:
        sizes.setdefault(model, EXTRA_PARAMS_B)
    return sizes


def _eval_points(path: Path, models: list[str]) -> list[dict]:
    points = list(json.loads(path.read_text())["points"])
    dataset = "ptb" if path.name.endswith("_ptb.json") else "wikitext2"
    for model in models:
        extra = EXTRA_FILES.get(model)
        if extra is None or not extra.is_file():
            continue
        for point in json.loads(extra.read_text()).get("points", []):
            if point.get("dataset") == dataset and point.get("model") == model:
                points.append(point)
    return points


def _rows(path: Path, sizes: dict[str, float], models_order: list[str] | None = None):
    models_order = models_order or ORDER
    data_points = _eval_points(path, models_order)
    dense = {
        point["model"]: float(point["perplexity"])
        for point in data_points
        if point["method"] == "dense"
    }
    sparsities, targets, billions, models = [], [], [], []
    above = {model: 0 for model in models_order}
    for point in data_points:
        if point["method"] != "sparsegpt":
            continue
        model = point["model"]
        base = dense.get(model)
        perplexity = float(point["perplexity"])
        if model not in sizes or base is None or base <= 0 or perplexity <= base:
            continue
        sparsities.append(float(point["sparsity"]))
        targets.append(math.log(perplexity / base))
        billions.append(float(sizes[model]))
        models.append(model)
        if model in above:
            above[model] += 1
    return (
        np.asarray(sparsities, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(billions, dtype=float),
        models,
        above,
        dense,
    )


def _per_model_sonly(sparsities, targets, models, models_order):
    """Fit y = β · s^b inside each model (N fixed)."""
    out = {}
    for model in models_order:
        idx = np.asarray([m == model for m in models])
        if int(idx.sum()) < 4:
            continue
        s = sparsities[idx]
        y = targets[idx]

        def resid(theta, s=s, y=y):
            amp, exp = theta
            return amp * np.power(s, exp) - y

        err, coef = _fit(
            resid,
            [[8.0, 4.0], [12.0, 5.0], [5.0, 3.0], [15.0, 6.0]],
            ([1.0e-6, 0.1], [1.0e4, 15.0]),
        )
        pred = float(coef[0]) * np.power(s, float(coef[1]))
        out[model] = {
            "beta": float(coef[0]),
            "b": float(coef[1]),
            "rmse": err,
            "r2": _r2(y, pred),
            "n": int(idx.sum()),
            "s": [float(v) for v in s],
            "y": [float(v) for v in y],
        }
    return out


def analyze(path: Path, sizes: dict[str, float], models_order: list[str] | None = None) -> dict:
    models_order = list(models_order or ORDER)
    sparsities, targets, billions, models, above, dense = _rows(path, sizes, models_order)

    def power(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude, exponent, scale = theta
        return amplitude * np.power(billions, scale) * np.power(sparsities, exponent) - targets

    err, coef = _fit(
        power,
        [[18.0, 5.0, -0.4], [16.0, 4.7, -0.38], [12.0, 4.0, -0.3], [15.0, 4.6, -0.35]],
        ([0.0, 0.0, -5.0], [1.0e4, 30.0, 5.0]),
    )
    amplitude, exponent, scale = (float(coef[0]), float(coef[1]), float(coef[2]))
    power_pred = amplitude * np.power(billions, scale) * np.power(sparsities, exponent)
    power_r2 = _r2(targets, power_pred)
    var_y = float(np.mean(np.square(targets - np.mean(targets))))

    mask = np.asarray([not str(model).endswith("27B") for model in models])

    def held(theta, sparsities=sparsities[mask], targets=targets[mask], billions=billions[mask]):
        amplitude_h, exponent_h, scale_h = theta
        return (
            amplitude_h * np.power(billions, scale_h) * np.power(sparsities, exponent_h)
            - targets
        )

    _, held_coef = _fit(held, [[amplitude, exponent, scale]], ([0.0, 0.0, -5.0], [1.0e4, 30.0, 5.0]))

    def slog(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude_s, b0, b1 = theta
        return amplitude_s * np.power(sparsities, b0 + b1 * np.log10(billions)) - targets

    def slog0(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude_0, slope = theta
        safe = (sparsities > 0.0) & (billions > 0.0)
        prediction = np.empty_like(targets)
        prediction[safe] = amplitude_0 * np.power(
            sparsities[safe], slope * np.log10(billions[safe])
        )
        prediction[~safe] = 1.0e6
        prediction[~np.isfinite(prediction)] = 1.0e6
        return prediction - targets

    def log1s(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude_l, exponent_l, scale_l = theta
        return (
            amplitude_l
            * np.power(-np.log(1.0 - sparsities), exponent_l)
            * np.power(billions, scale_l)
            - targets
        )

    def exponential(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude_e, exponent_e, scale_e = theta
        return amplitude_e * np.exp(exponent_e * sparsities * np.power(billions, scale_e)) - targets

    def nosize(theta, sparsities=sparsities, targets=targets):
        amplitude_n, exponent_n = theta
        return amplitude_n * np.power(sparsities, exponent_n) - targets

    def slog_held(
        theta,
        sparsities=sparsities[mask],
        targets=targets[mask],
        billions=billions[mask],
    ):
        amplitude_h, offset_h, slope_h = theta
        return (
            amplitude_h * np.power(sparsities, offset_h + slope_h * np.log10(billions))
            - targets
        )

    slog_err, slog_coef = _fit(
        slog,
        [[10.0, 4.0, 1.0], [18.0, 5.0, -1.0], [12.0, 6.0, -0.5], [8.0, 4.5, 0.8]],
        ([1.0e-6, -5.0, -10.0], [1.0e4, 30.0, 10.0]),
    )
    _, held_slog = _fit(
        slog_held,
        [slog_coef.tolist()],
        ([1.0e-6, -5.0, -10.0], [1.0e4, 30.0, 10.0]),
    )
    ref_a = float(slog_coef[0])
    ref_b0 = float(slog_coef[1])
    ref_b1 = float(slog_coef[2])
    ref_nstar = 10.0 ** (-ref_b0 / ref_b1)
    held_nstar = 10.0 ** (-float(held_slog[1]) / float(held_slog[2]))
    positive = (sparsities > 0.0) & (billions > 0.0) & (targets > 0.0)
    column = np.log10(billions[positive]) * np.log(sparsities[positive])
    slope, intercept = np.linalg.lstsq(
        np.column_stack([column, np.ones(column.size)]),
        np.log(targets[positive]),
        rcond=None,
    )[0]
    seeded = float(np.clip(np.exp(intercept), 1.0e-5, 9.0e3))
    slope = float(np.clip(slope, -39.0, 39.0))
    slog0_err, slog0_coef = _fit(
        slog0,
        [
            [seeded, slope],
            [10.0, 1.0],
            [18.0, 5.0],
            [12.0, 8.0],
            [8.0, -1.0],
            [20.0, 3.0],
            [5.0, 12.0],
            [15.0, -4.0],
        ],
        ([1.0e-6, -40.0], [1.0e4, 40.0]),
    )
    forms = {
        "power": err,
        "slog": slog_err,
        "slog0": slog0_err,
        "log1s": _fit(
            log1s,
            [[8.0, 3.0, -0.4], [18.0, 5.0, -0.4], [4.0, 2.0, -0.3]],
            ([0.0, 0.0, -5.0], [1.0e4, 30.0, 5.0]),
        )[0],
        "exp": _fit(
            exponential,
            [[0.05, 4.0, -0.4], [0.2, 2.0, -0.3], [0.01, 8.0, -0.2], [0.1, 3.0, 0.0]],
            ([1.0e-8, -20.0, -5.0], [1.0e3, 40.0, 5.0]),
        )[0],
        "nosize": _fit(
            nosize,
            [[12.0, 4.0], [18.0, 5.0], [8.0, 3.0]],
            ([0.0, 0.0], [1.0e4, 30.0]),
        )[0],
    }

    per_model = {}
    for model in models_order:
        idx = np.asarray([m == model for m in models])
        if int(idx.sum()) < 3:
            continue
        y = targets[idx]
        pred = power_pred[idx]
        per_model[model] = {
            "n": int(idx.sum()),
            "rmse": _rmse(pred - y),
            "r2": _r2(y, pred),
            "var_y": float(np.mean(np.square(y - np.mean(y)))),
            "s": [float(v) for v in sparsities[idx]],
            "y": [float(v) for v in y],
            "N": float(sizes[model]),
        }

    return {
        "n": int(len(targets)),
        "a": amplitude,
        "b": exponent,
        "c": scale,
        "rmse": err,
        "r2": power_r2,
        "var_y": var_y,
        "median": float(np.median(np.abs(power_pred - targets))),
        "models": models_order,
        "pref": {model: amplitude * (sizes[model] ** scale) for model in models_order},
        "held": [float(value) for value in held_coef],
        "above": above,
        "forms": {**forms, "ref": slog_err},
        "per_model": per_model,
        "per_model_sonly": _per_model_sonly(sparsities, targets, models, models_order),
        "points": {
            "s": [float(v) for v in sparsities],
            "y": [float(v) for v in targets],
            "N": [float(v) for v in billions],
            "model": list(models),
        },
        "dense": {model: float(value) for model, value in dense.items()},
        "ref": {
            "a": ref_a,
            "b0": ref_b0,
            "b1": ref_b1,
            "n_star": ref_nstar,
            "rmse": slog_err,
            "r2": _r2(targets, ref_a * np.power(sparsities, ref_b0 + ref_b1 * np.log10(billions))),
            "exponent": {
                model: ref_b1 * math.log10(sizes[model] / ref_nstar) for model in models_order
            },
            "held": [float(held_slog[0]), held_nstar, float(held_slog[2])],
        },
        "slog_fit": {
            "a": float(slog_coef[0]),
            "b0": float(slog_coef[1]),
            "b1": float(slog_coef[2]),
            "rmse": slog_err,
        },
        "slog0_fit": {
            "a": float(slog0_coef[0]),
            "b": float(slog0_coef[1]),
            "rmse": slog0_err,
        },
        "ten": 10.0 ** scale,
    }


def _bundle() -> str:
    text = BUNDLE_SOURCE.read_text()
    scripts = []
    cursor = 0
    while len(scripts) < 2:
        start = text.find("<script", cursor)
        end = text.find("</script>", start)
        scripts.append(text[start : end + len("</script>")])
        cursor = end + len("</script>")
    if "plotly.js" not in scripts[1][:300]:
        raise RuntimeError("Plotly bundle was not the second script in the family figure")
    return scripts[0] + "\n" + scripts[1]


def _label(model: str, models: list[str]) -> str:
    later = [item for item in models if item in EXTRA_FILES]
    if later and model.endswith("27B"):
        return model.split("/")[-1].removeprefix("Qwen")
    return SHORT.get(model, model.split("/")[-1])


def _s_grid(lo: float = 0.08, hi: float = 0.82, n: int = 120) -> np.ndarray:
    return np.linspace(lo, hi, n)


def _chart(wiki: dict, ptb: dict) -> tuple[str, str]:
    """Interactive collapse: family law + per-model points; dropdown to isolate a model fit."""
    models = wiki["models"]
    labels = {model: _label(model, models) for model in models}
    s_grid = _s_grid()
    fig = go.Figure()

    # Family master curves on the collapsed y = α · s^b  (after dividing out N^β)
    for name, row, color, dash in (
        ("WikiText2 family", wiki, "#1d4ed8", "solid"),
        ("PTB family", ptb, "#c2410c", "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=s_grid,
                y=row["a"] * np.power(s_grid, row["b"]),
                mode="lines",
                name=f"{name}: {row['a']:.2f}·N^({row['c']:.2f})·s^{row['b']:.2f}",
                line=dict(color=color, width=2.6, dash=dash),
                hovertemplate="%{fullData.name}<br>s=%{x:.2f}<br>α·s^b=%{y:.3f}<extra></extra>",
                legendgroup="family",
            )
        )

    # Per-model collapsed points (both evals); model-specific s-only curves start hidden
    for model in models:
        color = MODEL_COLORS.get(model, "#334155")
        label = labels[model]
        for row, symbol, ds_name in (
            (wiki, "circle", "WikiText2"),
            (ptb, "diamond-open", "PTB"),
        ):
            info = row["per_model"].get(model)
            if not info:
                continue
            collapsed = [
                y * (info["N"] ** (-row["c"])) for y in info["y"]
            ]
            fig.add_trace(
                go.Scatter(
                    x=info["s"],
                    y=collapsed,
                    mode="markers",
                    name=f"{label} · {ds_name}",
                    marker=dict(
                        color=color,
                        symbol=symbol,
                        size=9,
                        line=dict(width=1.2, color=color),
                    ),
                    hovertemplate=(
                        f"{label} · {ds_name}<br>s=%{{x:.2f}}<br>"
                        f"N^(-β)·ln(ppl/dense)=%{{y:.3f}}<extra></extra>"
                    ),
                    legendgroup=model,
                )
            )
        # Model-specific WikiText2 s-only fit in collapsed coordinates of the WikiText2 family β
        sonly = wiki["per_model_sonly"].get(model)
        info = wiki["per_model"].get(model)
        if sonly and info:
            # y_model = β_m · s^{b_m}; collapsed with family scale: y / N^β_family
            y_curve = (sonly["beta"] * np.power(s_grid, sonly["b"])) * (
                info["N"] ** (-wiki["c"])
            )
            fig.add_trace(
                go.Scatter(
                    x=s_grid,
                    y=y_curve,
                    mode="lines",
                    name=f"{label} own fit (WT2): {sonly['beta']:.2f}·s^{sonly['b']:.2f}",
                    line=dict(color=color, width=2.0, dash="dot"),
                    visible=False,
                    legendgroup=f"{model}-fit",
                    hovertemplate=(
                        f"{label} own WT2 fit<br>s=%{{x:.2f}}<br>collapsed=%{{y:.3f}}<extra></extra>"
                    ),
                )
            )

    # Trace inventory: 2 family curves, then for each model: up to 2 point traces + 1 own-fit.
    n_family = 2
    idx = n_family
    model_trace_ranges: dict[str, tuple[int, int]] = {}
    model_fit_index: dict[str, int | None] = {}
    for model in models:
        start = idx
        for row in (wiki, ptb):
            if model in row["per_model"]:
                idx += 1
        fit_idx = None
        if model in wiki["per_model_sonly"] and model in wiki["per_model"]:
            fit_idx = idx
            idx += 1
        model_trace_ranges[model] = (start, idx)
        model_fit_index[model] = fit_idx
    n_traces = idx
    if n_traces != len(fig.data):
        raise RuntimeError(f"trace count mismatch: planned {n_traces}, fig has {len(fig.data)}")

    family_title = (
        f"Scale-free collapse · WikiText2 {wiki['a']:.2f}·N^({wiki['c']:.2f})·s^{wiki['b']:.2f}"
        f" · PTB {ptb['a']:.2f}·N^({ptb['c']:.2f})·s^{ptb['b']:.2f}"
    )
    # Default: family + all points, hide own-fit curves
    default_visible = [True] * n_traces
    for fit_idx in model_fit_index.values():
        if fit_idx is not None:
            default_visible[fit_idx] = False

    buttons = [
        dict(
            label="All models",
            method="update",
            args=[{"visible": default_visible}, {"title": family_title}],
        )
    ]
    for model in models:
        label = labels[model]
        visible = [False] * n_traces
        visible[0] = True
        visible[1] = True
        start, stop = model_trace_ranges[model]
        for t in range(start, stop):
            visible[t] = True
        sonly = wiki["per_model_sonly"].get(model, {})
        pm = wiki["per_model"].get(model, {})
        title = (
            f"{label} · family law + own WT2 fit "
            f"({sonly.get('beta', float('nan')):.2f}·s^{sonly.get('b', float('nan')):.2f}, "
            f"R²={sonly.get('r2', float('nan')):.3f}, family R²={pm.get('r2', float('nan')):.3f})"
        )
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visible}, {"title": title}],
            )
        )

    fig.update_layout(
        template="plotly_white",
        width=860,
        height=480,
        margin=dict(t=80, b=60, l=70, r=30),
        title=dict(
            text=(
                f"Scale-free collapse · WikiText2 {wiki['a']:.2f}·N^({wiki['c']:.2f})·s^{wiki['b']:.2f}"
                f" · PTB {ptb['a']:.2f}·N^({ptb['c']:.2f})·s^{ptb['b']:.2f}"
            ),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(15,23,42,0.12)",
            borderwidth=1,
        ),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0f172a"),
        xaxis=dict(title_text="Unstructured sparsity s (log)", type="log"),
        yaxis=dict(title_text="N^(−β) · ln(perplexity / dense) (log)", type="log"),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.0,
                xanchor="right",
                y=1.18,
                yanchor="top",
                buttons=buttons,
                bgcolor="#f8fafc",
                bordercolor="#cbd5e1",
                font=dict(size=12),
                active=0,
            )
        ],
        annotations=[
            dict(
                text="Show fit for",
                x=1.0,
                xref="paper",
                y=1.18,
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#475569"),
                xshift=-8,
                yshift=18,
            )
        ],
    )
    # Apply default visibility (hide own-fit curves until a model is selected)
    for i, vis in enumerate(default_visible):
        fig.data[i].visible = vis

    chart = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="prefactor-chart",
        config={"displaylogo": False, "responsive": True},
    )
    start = chart.find('<div id="prefactor-chart"')
    end = chart.find("</div>", start) + len("</div>")
    div = chart[start:end].replace(
        'style="height:100%; width:100%;"',
        'style="height:480px; width:100%;"',
        1,
    )
    script = chart[chart.find("<script") :]
    script = script.split("</script>", 1)[0] + "</script>"
    return div, script


def _law_power(row: dict) -> str:
    return (
        f"ln(perplexity / dense) = {row['a']:.2f} · N<sup>{row['c']:.2f}</sup>"
        f" · s<sup>{row['b']:.2f}</sup>"
    )


def _sample_rows(models: list[str]) -> str:
    later = [model for model in models if model in EXTRA_FILES]
    if not later:
        return ""
    rows = []
    for model in later:
        points = json.loads(EXTRA_FILES[model].read_text())["points"]
        dense = {
            point["dataset"]: float(point["perplexity"])
            for point in points
            if point["method"] == "dense"
        }
        ordered = sorted(
            points,
            key=lambda point: (point["dataset"], float(point["sparsity"])),
        )
        for point in ordered:
            if point["method"] not in ("dense", "sparsegpt"):
                continue
            dataset = point["dataset"]
            perplexity = float(point["perplexity"])
            base = dense.get(dataset)
            ratio = "" if base is None or base <= 0 else f"{perplexity / base:.3f}"
            rows.append(
                "<tr>"
                f"<td>{model.split('/')[-1]}</td>"
                f"<td>{'WikiText2' if dataset == 'wikitext2' else 'PTB'}</td>"
                f"<td class=\"num\">{float(point['sparsity']):.1f}</td>"
                f"<td class=\"num\">{perplexity:.3f}</td>"
                f"<td class=\"num\">{ratio}</td>"
                "</tr>"
            )
    body = "".join(rows)
    names = [model.split("/")[-1] for model in later]
    heading = " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + ", and " + names[-1]
    return (
        f"<h2>{heading}</h2>"
        "<p class=\"small muted\">Same protocol and the same parameter count as Qwen3.5-27B. "
        "Sparsity 0 is the dense baseline. These rows enter the fit when perplexity is above that baseline. "
        f"{_partial_sentence(later)}</p>"
        "<table><thead><tr><th>Model</th><th>Eval</th><th class=\"num\">Sparsity</th>"
        "<th class=\"num\">Perplexity</th><th class=\"num\">Perplexity / dense</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render(wiki: dict, ptb: dict) -> str:
    models = wiki["models"]
    sizes = load_sizes(models)
    later_names = [model.split("/")[-1] for model in models if model in EXTRA_FILES]
    later_sentence = ""
    if later_names:
        later_sentence = (
            " The fit also includes "
            + " and ".join(later_names)
            + ", at the same parameter count as Qwen3.5-27B."
        )
    sample_table = _sample_rows(models)
    form_rows = []
    for key, formula, params, chosen in FORMS:
        cls = " class=\"chosen\"" if chosen else ""
        tag = "<span class=\"chosen-tag\">chosen</span>" if chosen else ""
        if key == "power":
            w_r2, p_r2 = f"{wiki['r2']:.3f}", f"{ptb['r2']:.3f}"
        elif key == "ref":
            w_r2, p_r2 = f"{wiki['ref']['r2']:.3f}", f"{ptb['ref']['r2']:.3f}"
        else:
            w_r2, p_r2 = "—", "—"
        form_rows.append(
            f"<tr{cls}><td><code>{formula}</code>{tag}</td>"
            f"<td class=\"num\">{params}</td>"
            f"<td class=\"num\">{wiki['forms'][key]:.3f}</td>"
            f"<td class=\"num\">{ptb['forms'][key]:.3f}</td>"
            f"<td class=\"num\">{w_r2}</td>"
            f"<td class=\"num\">{p_r2}</td></tr>"
        )

    per_model_rows = []
    for model in models:
        label = _label(model, models)
        w = wiki["per_model"].get(model)
        p = ptb["per_model"].get(model)
        ws = wiki["per_model_sonly"].get(model)
        ps = ptb["per_model_sonly"].get(model)
        if not w and not p:
            continue
        per_model_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td class=\"num\">{sizes[model]:.2f}</td>"
            f"<td class=\"num\">{(w['r2'] if w else float('nan')):.3f}</td>"
            f"<td class=\"num\">{(p['r2'] if p else float('nan')):.3f}</td>"
            f"<td class=\"num\">{(w['rmse'] if w else float('nan')):.3f}</td>"
            f"<td class=\"num\">{(p['rmse'] if p else float('nan')):.3f}</td>"
            f"<td class=\"num\">{(ws['r2'] if ws else float('nan')):.3f}</td>"
            f"<td class=\"num\">{(ps['r2'] if ps else float('nan')):.3f}</td>"
            "</tr>"
        )

    amp_rows = []
    for model in models:
        label = _label(model, models)
        amp_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td class=\"num\">{sizes[model]:.2f}</td>"
            f"<td class=\"num\">{wiki['pref'][model]:.3f}</td>"
            f"<td class=\"num\">{ptb['pref'][model]:.3f}</td>"
            "</tr>"
        )

    div, chart_script = _chart(wiki, ptb)
    bundle = _bundle()
    from universal_scaling import other_family_section

    other_section = other_family_section()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scale-free sparsity law for the Qwen family</title>
<style>
  :root {{
    --ink: #0f172a; --muted: #475569; --faint: #64748b; --line: #e2e8f0;
    --info: #1d4ed8; --info-bg: #eff6ff; --ok: #166534; --ok-bg: #ecfdf3;
    --card: #f8fafc; --ptb: #9a3412;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: var(--ink); background: #f1f5f9;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.5; }}
  main {{ max-width: 920px; margin: 32px auto 64px; padding: 32px 36px 40px;
    background: #fff; border: 1px solid var(--line); border-radius: 12px; }}
  h1 {{ margin: 0 0 12px; font-size: 28px; line-height: 1.25; letter-spacing: -0.02em; }}
  h2 {{ margin: 28px 0 8px; font-size: 18px; }}
  p {{ margin: 0 0 10px; }}
  .lead {{ font-size: 16px; }}
  .principle {{ font-size: 16px; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 13px; }}
  .faint {{ color: var(--faint); font-size: 13px; }}
  .law {{ margin: 14px 0 8px; padding: 12px 14px; background: var(--card);
    border-left: 3px solid var(--info); font-weight: 650; font-size: 17px; }}
  .law.ptb {{ border-left-color: #c2410c; }}
  .ds {{ display: inline-block; margin-right: 8px; font-size: 12px; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--info); }}
  .law.ptb .ds {{ color: var(--ptb); }}
  .callout {{ margin: 16px 0 8px; padding: 14px 16px; background: var(--info-bg);
    border: 1px solid #bfdbfe; border-radius: 10px; }}
  .callout h3 {{ margin: 0 0 6px; font-size: 15px; color: #1e3a8a; }}
  hr {{ border: 0; border-top: 1px solid var(--line); margin: 22px 0 8px; }}
  .chart-wrap {{ margin-top: 8px; border: 1px solid var(--line); border-radius: 10px; padding: 4px 4px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0 6px; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
  th {{ font-size: 13px; color: var(--muted); font-weight: 650; }}
  td.num, th.num {{ text-align: right; }}
  code {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13.5px; }}
  tr.chosen td {{ background: var(--ok-bg); color: #14532d; }}
  tr.chosen td:first-child {{ box-shadow: inset 3px 0 0 var(--ok); }}
  .chosen-tag {{ margin-left: 8px; font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ok); }}
  .caption {{ margin-top: 8px; }}
  .identity {{ margin: 10px 0 12px; padding: 10px 14px; background: var(--card);
    border-radius: 8px; font-weight: 650; }}
  @media (max-width: 720px) {{
    main {{ margin: 0; padding: 20px 16px 32px; border-radius: 0; }}
    h1 {{ font-size: 24px; }}
    .law {{ font-size: 15px; }}
  }}
</style>
</head>
<body>
<main>
  <h1>Scale-free sparsity law for the Qwen family</h1>
  <p class="lead">Unstructured SparseGPT, C4 calibration. The same expression is fit separately on raw WikiText2 ({wiki['n']} points) and PTB ({ptb['n']} points). Each point has SparseGPT perplexity above that model’s own dense value, from 0.8B through 27B.{later_sentence}</p>
  <p class="principle">The chosen form is scale-free: <code>ln(ppl / dense) = α · N<sup>β</sup> · s<sup>b</sup></code>. Size and sparsity factorize; changing the unit of N only rescales α.</p>
  <p class="law"><span class="ds">WikiText2</span> {_law_power(wiki)}</p>
  <p class="law ptb"><span class="ds">PTB</span> {_law_power(ptb)}</p>
  <p class="small muted">s is unstructured sparsity. N is stored parameters in billions (safetensors bytes / 2). Points at or below dense are left out. R² = 1 − RMSE² / Var(y) on y = ln(ppl / dense).</p>
  <table>
    <thead>
      <tr><th>Eval</th><th class="num">α</th><th class="num">β</th><th class="num">b</th><th class="num">RMSE</th><th class="num">Var(y)</th><th class="num">R²</th></tr>
    </thead>
    <tbody>
      <tr><td>WikiText2</td><td class="num">{wiki['a']:.2f}</td><td class="num">{wiki['c']:.3f}</td><td class="num">{wiki['b']:.2f}</td><td class="num">{wiki['rmse']:.3f}</td><td class="num">{wiki['var_y']:.3f}</td><td class="num">{wiki['r2']:.3f}</td></tr>
      <tr><td>PTB</td><td class="num">{ptb['a']:.2f}</td><td class="num">{ptb['c']:.3f}</td><td class="num">{ptb['b']:.2f}</td><td class="num">{ptb['rmse']:.3f}</td><td class="num">{ptb['var_y']:.3f}</td><td class="num">{ptb['r2']:.3f}</td></tr>
    </tbody>
  </table>
  <div class="callout">
    <h3>What the law says</h3>
    <p>Doubling N multiplies ln(ppl / dense) by 2<sup>β</sup> ≈ {2 ** wiki['c']:.2f} on WikiText2 and {2 ** ptb['c']:.2f} on PTB, at every sparsity. Holding cost fixed requires N ∝ s<sup>(b / |β|)</sup> ≈ s<sup>{abs(wiki['b'] / wiki['c']):.1f}</sup> (WikiText2). A rounded fixed rule near these fits is <code>15 · N<sup>(−1/3)</sup> · s<sup>4.6</sup></code>.</p>
  </div>
  {sample_table}
  <hr>
  <h2>Universal collapse (interactive)</h2>
  <p class="small muted">Vertical axis: N<sup>(−β)</sup> · ln(ppl / dense) with that eval’s family β, so every model should fall on α · s<sup>b</sup>. Use the dropdown to isolate one model: its points stay, and its own WikiText2 fit <code>β · s<sup>b</sup></code> appears as a dotted curve. Click legend entries to toggle series.</p>
  <div class="chart-wrap">
    {div}
  </div>
  <p class="caption faint">Filled circles = WikiText2 · open diamonds = PTB. Solid / dashed curves = family laws. Dotted = selected model’s own WT2 fit.</p>
  <h2>R² by model</h2>
  <p class="small muted">Family columns use the shared α · N<sup>β</sup> · s<sup>b</sup> predictions. Own-fit columns refit <code>β · s<sup>b</sup></code> on that model alone (N fixed).</p>
  <table>
    <thead>
      <tr>
        <th>Model</th><th class="num">N (B)</th>
        <th class="num">WT2 family R²</th><th class="num">PTB family R²</th>
        <th class="num">WT2 family RMSE</th><th class="num">PTB family RMSE</th>
        <th class="num">WT2 own R²</th><th class="num">PTB own R²</th>
      </tr>
    </thead>
    <tbody>
      {''.join(per_model_rows)}
    </tbody>
  </table>
  <h2>Amplitude α · N<sup>β</sup> at each size</h2>
  <p class="small muted">With sparsity factored out, each model’s prefactor is α · N<sup>β</sup>. Larger models have smaller prefators.</p>
  <table>
    <thead>
      <tr><th>Model</th><th class="num">Parameters (billions)</th><th class="num">WikiText2 α·N<sup>β</sup></th><th class="num">PTB α·N<sup>β</sup></th></tr>
    </thead>
    <tbody>
      {''.join(amp_rows)}
    </tbody>
  </table>
  <h2>Same points, other forms</h2>
  <p class="small muted">RMSE is on ln(perplexity / dense). R² is shown for the scale-free law and the older reference law. The highlighted row is the form used above.</p>
  <table>
    <thead>
      <tr><th>Form</th><th class="num">Parameters</th><th class="num">WikiText2 RMSE</th><th class="num">PTB RMSE</th><th class="num">WikiText2 R²</th><th class="num">PTB R²</th></tr>
    </thead>
    <tbody>
      {''.join(form_rows)}
    </tbody>
  </table>
  {other_section}
  <p class="faint">Median absolute error on ln(perplexity / dense) is {wiki['median']:.3f} across the {wiki['n']} WikiText2 points and {ptb['median']:.3f} across the {ptb['n']} PTB points. Holding out every 27B model leaves α, β, b at {wiki['held'][0]:.2f}, {wiki['held'][2]:.3f}, {wiki['held'][1]:.2f} on WikiText2 and {ptb['held'][0]:.2f}, {ptb['held'][2]:.3f}, {ptb['held'][1]:.2f} on PTB. Source: C4 calibration; WikiText2 from figures/qwen3.5_family_sparsity_wikitext2.json; PTB from figures/qwen3.5_family_sparsity_ptb.json.{' ' + ', '.join(model.split('/')[-1] + ' from figures/' + EXTRA_FILES[model].name for model in models if model in EXTRA_FILES) + '.' if any(model in EXTRA_FILES for model in models) else ''}</p>
</main>
{bundle}
{chart_script}
</body>
</html>
"""


def write_page() -> dict:
    models = model_order()
    sizes = load_sizes(models)
    wiki = analyze(WIKI_PATH, sizes, models)
    ptb = analyze(PTB_PATH, sizes, models)
    text = render(wiki, ptb)
    if text.count("<div") != text.count("</div>"):
        raise RuntimeError("sparsity principle page has unbalanced divs")
    if "plotly.js" not in text or "Plotly.newPlot" not in text:
        raise RuntimeError("sparsity principle page is missing Plotly")
    PAGE.write_text(text)
    print(
        f"wiki ln(ppl/dense) = {wiki['a']:.2f} * N^({wiki['c']:.3f}) * s^{wiki['b']:.2f} "
        f"RMSE {wiki['rmse']:.3f} R² {wiki['r2']:.3f} n={wiki['n']}",
        flush=True,
    )
    print(
        f"ptb  ln(ppl/dense) = {ptb['a']:.2f} * N^({ptb['c']:.3f}) * s^{ptb['b']:.2f} "
        f"RMSE {ptb['rmse']:.3f} R² {ptb['r2']:.3f} n={ptb['n']}",
        flush=True,
    )
    for model in models:
        w = wiki["per_model"].get(model)
        p = ptb["per_model"].get(model)
        if not w and not p:
            continue
        w_r2 = f"{w['r2']:.3f}" if w else "  —  "
        p_r2 = f"{p['r2']:.3f}" if p else "  —  "
        print(f"  {_label(model, models):12s}  WT2 R²={w_r2}  PTB R²={p_r2}", flush=True)
    print(f"Wrote {PAGE}", flush=True)
    return {"wiki": wiki, "ptb": ptb}


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()
