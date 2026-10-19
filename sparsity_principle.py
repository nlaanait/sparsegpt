"""Refit the shared sparsity law on WikiText2 and PTB and rewrite the page.

Points at or below each model's own dense perplexity are left out. The chosen
form is a · (N / N*)^(b1 log10 s), fit separately on each eval. It is the same
fit as a · s^(b0 + b1 log10 N). The other rows are comparisons.
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
FORMS = (
    ("ref", "a · (N / N*)^(b1 log10 s)", "3", True),
    ("slog", "a · s^(b0 + b1 log10 N)", "3", False),
    ("power", "a · N^c · s^b", "3", False),
    ("slog0", "a · s^(b log10 N)", "2", False),
    ("log1s", "a · (−log(1−s))^b · N^c", "3", False),
    ("exp", "a · exp(b · s · N^c)", "3", False),
    ("nosize", "a · s^b", "2", False),
)


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


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
    )


def analyze(path: Path, sizes: dict[str, float], models_order: list[str] | None = None) -> dict:
    models_order = list(models_order or ORDER)
    sparsities, targets, billions, models, above = _rows(path, sizes, models_order)

    def power(theta, sparsities=sparsities, targets=targets, billions=billions):
        amplitude, exponent, scale = theta
        return amplitude * np.power(billions, scale) * np.power(sparsities, exponent) - targets

    err, coef = _fit(
        power,
        [[18.0, 5.0, -0.4], [16.0, 4.7, -0.38], [12.0, 4.0, -0.3]],
        ([0.0, 0.0, -5.0], [1.0e4, 30.0, 5.0]),
    )
    amplitude, exponent, scale = (float(coef[0]), float(coef[1]), float(coef[2]))
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

    mask = np.asarray([not str(model).endswith("27B") for model in models])

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
    return {
        "n": int(len(targets)),
        "a": amplitude,
        "b": exponent,
        "c": scale,
        "rmse": err,
        "median": float(np.median(np.abs(slog(slog_coef)))),
        "models": models_order,
        "pref": {model: amplitude * (sizes[model] ** scale) for model in models_order},
        "held": [float(value) for value in held_coef],
        "above": above,
        "forms": {**forms, "ref": slog_err},
        "ref": {
            "a": ref_a,
            "b0": ref_b0,
            "b1": ref_b1,
            "n_star": ref_nstar,
            "rmse": slog_err,
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
    return SHORT[model]


def _chart(wiki: dict, ptb: dict) -> tuple[str, str]:
    models = wiki["models"]
    labels = [_label(model, models) for model in models]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[wiki["ref"]["exponent"][model] for model in models],
            mode="lines+markers",
            name="WikiText2",
            hovertemplate="WikiText2 %{x}<br>exponent b1 log10(N/N*) = %{y:.2f}<extra></extra>",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(color="#2563eb", size=10, line=dict(color="#1e3a8a", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[ptb["ref"]["exponent"][model] for model in models],
            mode="lines+markers",
            name="PTB",
            hovertemplate="PTB %{x}<br>exponent b1 log10(N/N*) = %{y:.2f}<extra></extra>",
            line=dict(color="#c2410c", width=2.5),
            marker=dict(color="#c2410c", size=10, line=dict(color="#7c2d12", width=1)),
        )
    )
    fig.update_layout(
        template="plotly_white",
        width=760,
        height=380,
        margin=dict(t=48, b=64, l=78, r=28),
        legend=dict(
            title_text="Exponent b1 log10(N/N*)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(15,23,42,0.18)",
            borderwidth=1,
        ),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0f172a"),
        yaxis=dict(title_text="Exponent b1 log10(N/N*)", range=[0, 10], tickformat=".1f"),
        xaxis=dict(
            title_text="Model, ordered by parameter count",
            type="category",
            categoryorder="array",
            categoryarray=labels,
        ),
    )
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
        'style="height:380px; width:100%;"',
        1,
    )
    script = chart[chart.find("<script") :]
    script = script.split("</script>", 1)[0] + "</script>"
    return div, script


def _signed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def _law(row: dict) -> str:
    ref = row["ref"]
    return (
        f"ln(perplexity / dense) = {ref['a']:.2f} · (N / {ref['n_star']:.3f})"
        f"<sup>{ref['b1']:.2f} log10 s</sup>"
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
    pref_rows = []
    for model in models:
        billions = float(sizes[model])
        billions_txt = f"{billions:.2f}" if billions < 20 else f"{billions:.1f}"
        pref_rows.append(
            "<tr>"
            f"<td>{_label(model, models)}</td>"
            f"<td class=\"num\">{billions_txt}</td>"
            f"<td class=\"num\">{wiki['ref']['exponent'][model]:.2f}</td>"
            f"<td class=\"num\">{ptb['ref']['exponent'][model]:.2f}</td>"
            "</tr>"
        )
    smallest = float(sizes[models[0]])
    later_names = [model.split("/")[-1] for model in models if model in EXTRA_FILES]
    later_sentence = ""
    if later_names:
        later_sentence = (
            " The fit also includes "
            + " and ".join(later_names)
            + ", at the same parameter count as Qwen3.5-27B."
        )
    sample_table = _sample_rows(models)
    above_txt = "; ".join(
        f"{_label(model, models)} has {wiki['above'][model]} WikiText2 points above dense and {ptb['above'][model]} on PTB"
        for model in models
        if model.endswith("27B")
    )
    wiki_ratio = smallest / wiki["ref"]["n_star"]
    ptb_ratio = smallest / ptb["ref"]["n_star"]
    wiki_penalty = math.exp(wiki["ref"]["a"])
    ptb_penalty = math.exp(ptb["ref"]["a"])
    wiki_base = 10.0 ** (1.0 / wiki["ref"]["b1"])
    ptb_base = 10.0 ** (1.0 / ptb["ref"]["b1"])
    form_rows = []
    for key, formula, params, chosen in FORMS:
        cls = " class=\"chosen\"" if chosen else ""
        tag = "<span class=\"chosen-tag\">chosen</span>" if chosen else ""
        form_rows.append(
            f"<tr{cls}><td><code>{formula}</code>{tag}</td>"
            f"<td class=\"num\">{params}</td>"
            f"<td class=\"num\">{wiki['forms'][key]:.3f}</td>"
            f"<td class=\"num\">{ptb['forms'][key]:.3f}</td></tr>"
        )
    div, chart_script = _chart(wiki, ptb)
    bundle = _bundle()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One sparsity law for the Qwen3.5 family</title>
<style>
  :root {{
    --ink: #0f172a; --muted: #475569; --faint: #64748b; --line: #e2e8f0;
    --info: #1d4ed8; --info-bg: #eff6ff; --ok: #166534; --ok-bg: #ecfdf3;
    --card: #f8fafc; --ptb: #9a3412;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: var(--ink); background: #f1f5f9;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.5; }}
  main {{ max-width: 880px; margin: 32px auto 64px; padding: 32px 36px 40px;
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
  <h1>One sparsity law for the Qwen3.5 family</h1>
  <p class="lead">Unstructured SparseGPT, C4 calibration. The same expression is fit separately on raw WikiText2 ({wiki['n']} points) and PTB ({ptb['n']} points). Each point has perplexity above that model’s own dense value, from 0.8B through 27B.{later_sentence}</p>
  <p class="principle">The sparsity exponent is b1 log10(N / N*). It is zero at N* and rises by b1 each time the model grows tenfold.</p>
  <p class="law"><span class="ds">WikiText2</span> {_law(wiki)}</p>
  <p class="law ptb"><span class="ds">PTB</span> {_law(ptb)}</p>
  <p class="small muted">s is sparsity. N and N* are parameter counts in billions. Points at or below dense are left out: this form is positive for N &gt; N*, and those measurements are a small improvement rather than a cost.</p>
  <table>
    <thead>
      <tr><th>Eval</th><th class="num">a</th><th class="num">N* (billions)</th><th class="num">b1</th><th class="num">RMSE</th><th class="num">Exponent at 27B</th></tr>
    </thead>
    <tbody>
      <tr><td>WikiText2</td><td class="num">{wiki['ref']['a']:.2f}</td><td class="num">{wiki['ref']['n_star']:.3f}</td><td class="num">{wiki['ref']['b1']:.2f}</td><td class="num">{wiki['ref']['rmse']:.3f}</td><td class="num">{wiki['ref']['exponent']['Qwen/Qwen3.5-27B']:.2f}</td></tr>
      <tr><td>PTB</td><td class="num">{ptb['ref']['a']:.2f}</td><td class="num">{ptb['ref']['n_star']:.3f}</td><td class="num">{ptb['ref']['b1']:.2f}</td><td class="num">{ptb['ref']['rmse']:.3f}</td><td class="num">{ptb['ref']['exponent']['Qwen/Qwen3.5-27B']:.2f}</td></tr>
    </tbody>
  </table>
  <div class="callout">
    <h3>What the law says</h3>
    <p>Each model follows a · s raised to b1 log10(N / N*). The amplitude a is shared. On WikiText2 the exponent is {wiki['ref']['exponent']['Qwen/Qwen3.5-0.8B']:.2f} at 0.8B and {wiki['ref']['exponent']['Qwen/Qwen3.5-27B']:.2f} at 27B. On PTB it is {ptb['ref']['exponent']['Qwen/Qwen3.5-0.8B']:.2f} at 0.8B and {ptb['ref']['exponent']['Qwen/Qwen3.5-27B']:.2f} at 27B. Writing the same fit as a · s<sup>(b0 + b1 log10 N)</sup> gives the same RMSE, {wiki['forms']['slog']:.3f} on WikiText2 and {ptb['forms']['slog']:.3f} on PTB.</p>
  </div>
  {sample_table}
  <h2>What N* means</h2>
  <p>The coefficients that are estimated are a, b0, and b1 in a · s<sup>(b0 + b1 log10 N)</sup>. On WikiText2 they are a = {wiki['ref']['a']:.2f}, b0 = {wiki['ref']['b0']:.2f}, b1 = {wiki['ref']['b1']:.2f}. On PTB they are a = {ptb['ref']['a']:.2f}, b0 = {ptb['ref']['b0']:.2f}, b1 = {ptb['ref']['b1']:.2f}.</p>
  <p>b1 is a change of base. b1 log10 N = log<sub>β</sub> N with β = 10<sup>(1 / b1)</sup>, which is {wiki_base:.2f} on WikiText2 and {ptb_base:.2f} on PTB. The exponent is then b0 + log<sub>β</sub> N, and b0 is still there. A rescaling of N absorbs b0, because log10(kN) = log10 k + log10 N and b1 factors out of both terms. Choosing the unit so that the additive constant vanishes gives</p>
  <p class="identity">b0 + b1 log10 N = b1 log10(N / N*), &nbsp; N* = 10<sup>(−b0 / b1)</sup></p>
  <p>That is the law at the top of the page. N* is {wiki['ref']['n_star'] * 1000:.0f} million parameters on WikiText2 and {ptb['ref']['n_star'] * 1000:.0f} million on PTB. The smallest measured model has {smallest:.2f} billion parameters, {wiki_ratio:.0f}× the WikiText2 value and {ptb_ratio:.0f}× the PTB value.</p>
  <p>The exponent of sparsity is k(N) = b1 log10(N / N*), and the predicted log-cost is a · s<sup>k(N)</sup>. Hold N fixed and let s approach 0 from above.</p>
  <table>
    <thead>
      <tr><th>Size</th><th>Exponent k(N)</th><th>Limit of ln(perplexity / dense)</th></tr>
    </thead>
    <tbody>
      <tr><td>N &gt; N*</td><td>positive</td><td>0</td></tr>
      <tr><td>N = N*</td><td>0</td><td>a, at every s &gt; 0</td></tr>
      <tr><td>N &lt; N*</td><td>negative</td><td>grows without bound</td></tr>
    </tbody>
  </table>
  <p>N* is the greatest lower bound on size for which that limit is 0. The condition is N &gt; N*: at the bound itself the size ratio is 1, and 1 to any power is 1, so the formula returns a for every positive sparsity. On WikiText2 that constant is a perplexity about {wiki_penalty:,.0f} times dense. On PTB it is about {ptb_penalty:,.0f} times dense. Dense, s = 0, is defined separately as cost 0, because log10 0 is undefined. The jump from 0 at s = 0 to a at every s &gt; 0 is the formula past the measured range.</p>
  <p>Every model in the sweep is above N*, from {smallest:.2f}B to 27B, so each has a positive exponent. An infinitesimal prune on those models is predicted to cost nothing in the limit. N* itself was not trained or pruned. It is the pivot read off by rewriting the fit.</p>
  <p>At N = N* the predicted cost does not change with s. Remaining parameters are N*(1 − s), so a search that minimizes remaining weights among sparsities the formula scores the same way selects the largest sparsity, and keeps selecting it out to a model with no weights left. That preference is what a flat extrapolation does.</p>
  <p>A finite cutoff makes a size. Require the predicted cost at sparsity ε to stay at most δ. The smallest N that meets it is</p>
  <p class="identity">N<sub>min</sub> = N* · 10<sup>ln(δ / a) / (b1 ln ε)</sup></p>
  <p class="small muted">For a small ε and a small allowed cost δ, N<sub>min</sub> is still below {smallest:.2f}B. The expression extrapolates a fit made on models from {smallest:.2f}B to 27B.</p>
  <hr>
  <h2>Exponent at each model size</h2>
  <p class="small muted">Horizontal axis: model, ordered by parameter count. Vertical axis: b1 log10(N / N*) from that eval’s own fit. The amplitude a is the same for every model.</p>
  <div class="chart-wrap">
    {div}
  </div>
  <p class="caption faint">Source: C4 calibration, points with perplexity above that model’s dense value. WikiText2 from figures/qwen3.5_family_sparsity_wikitext2.json. PTB from figures/qwen3.5_family_sparsity_ptb.json.{' ' + ', '.join(model.split('/')[-1] + ' from figures/' + EXTRA_FILES[model].name for model in models if model in EXTRA_FILES) + '.' if any(model in EXTRA_FILES for model in models) else ''}</p>
  <h2>Same points, other forms</h2>
  <p class="small muted">RMSE is on ln(perplexity / dense). Each column is fit on that eval alone. The highlighted row is the form used above. The next row is that same fit written as a · s^(b0 + b1 log10 N), so the two RMSE values match.</p>
  <table>
    <thead>
      <tr><th>Form</th><th class="num">Parameters</th><th class="num">WikiText2 RMSE</th><th class="num">PTB RMSE</th></tr>
    </thead>
    <tbody>
      {''.join(form_rows)}
    </tbody>
  </table>
  <h2>Exponent the law assigns</h2>
  <table>
    <thead>
      <tr><th>Model</th><th class="num">Parameters (billions)</th><th class="num">WikiText2 b1 log10(N/N*)</th><th class="num">PTB b1 log10(N/N*)</th></tr>
    </thead>
    <tbody>
      {''.join(pref_rows)}
    </tbody>
  </table>
  <p class="faint">Median absolute error on ln(perplexity / dense) is {wiki['median']:.3f} across the {wiki['n']} WikiText2 points and {ptb['median']:.3f} across the {ptb['n']} PTB points. Holding out every 27B model leaves a, N*, and b1 at {wiki['ref']['held'][0]:.2f}, {wiki['ref']['held'][1]:.3f}, {wiki['ref']['held'][2]:.2f} on WikiText2 and {ptb['ref']['held'][0]:.2f}, {ptb['ref']['held'][1]:.3f}, {ptb['ref']['held'][2]:.2f} on PTB. {above_txt}.</p>
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
        f"wiki ln(ppl/dense) = {wiki['ref']['a']:.2f} * (N / {wiki['ref']['n_star']:.4f})"
        f"^( {wiki['ref']['b1']:.2f} log10 s ) RMSE {wiki['ref']['rmse']:.3f} n={wiki['n']}",
        flush=True,
    )
    print(
        f"ptb  ln(ppl/dense) = {ptb['ref']['a']:.2f} * (N / {ptb['ref']['n_star']:.4f})"
        f"^( {ptb['ref']['b1']:.2f} log10 s ) RMSE {ptb['ref']['rmse']:.3f} n={ptb['n']}",
        flush=True,
    )
    for name, row in (("wiki", wiki), ("ptb", ptb)):
        fit = row["slog0_fit"]
        print(
            f"{name} slog0 a={fit['a']:.4f} b={fit['b']:.4f} RMSE {fit['rmse']:.3f} | "
            f"power RMSE {row['forms']['power']:.3f} slog RMSE {row['forms']['slog']:.3f}",
            flush=True,
        )
    print(f"Wrote {PAGE}", flush=True)
    return {"wiki": wiki, "ptb": ptb}


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()
