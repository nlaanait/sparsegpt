"""Universal scaling figure: one row per family, collapse beside ppl/dense.

Qwen and Gemma-4 are the panels already on results/universal_scaling_law.html.
OLMo 2 Instruct and Llama 3 are fit on C4 calibration, WikiText2 and PTB,
with the same markers: filled circles for WikiText2, open diamonds for PTB.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.fit import solve_scale_free_law
from analysis.series import load_rows
from experiments.llama import LLAMA3, stored_params as llama_stored_params
from experiments.olmo import OLMO2, stored_params as olmo_stored_params
from experiments.protocol import ROOT

PAGE = ROOT / "results" / "universal_scaling_law.html"
CARRIED = ROOT / "results" / "universal_qwen_gemma_traces.json"
CURVE_S = np.linspace(0.08, 0.82, 80)
PANEL_WIDTH = 730
FAMILY_COLORS = {
    "OLMo 2 Instruct": ("#047857", "#0f766e", "#115e59", "#134e4a"),
    "Llama 3": ("#4338ca", "#6d28d9", "#7e22ce", "#a21caf", "#be185d", "#9d174d"),
}


def _grab(text: str, start: int, open_c: str, close_c: str) -> str:
    depth = 0
    in_str = False
    esc = False
    for index in range(start, len(text)):
        char = text[index]
        if in_str:
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
            continue
        if char == open_c:
            depth += 1
        elif char == close_c:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise RuntimeError("unbalanced plotly payload")


def _decode(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        code = {"f8": "d", "f4": "f", "i4": "i"}[value["dtype"]]
        count = len(raw) // struct.calcsize(code)
        return list(struct.unpack("<" + code * count, raw))
    raise TypeError(f"cannot decode plot array of type {type(value).__name__}")


def _existing() -> dict[str, list[dict]]:
    """Qwen and Gemma traces from the original two-row page.

    The written page is not a source: re-reading it swaps axes once OLMo and
    Llama rows are added.
    """
    traces = json.loads(CARRIED.read_text())
    groups = {"qwen-collapse": [], "gemma-collapse": [], "qwen-norm": [], "gemma-norm": []}
    for trace in traces:
        axis = trace.get("xaxis", "x")
        if axis in ("x", "x1"):
            groups["qwen-collapse"].append(trace)
        elif axis == "x2":
            groups["gemma-collapse"].append(trace)
        elif axis == "x3":
            groups["qwen-norm"].append(trace)
        elif axis == "x4":
            groups["gemma-norm"].append(trace)
    return groups


def _replay(fig: go.Figure, traces: list[dict], kind: str) -> None:
    seen_markers: set[str] = set()
    for trace in traces:
        mode = trace.get("mode", "markers")
        name = trace.get("name", "")
        show = trace.get("showlegend", True)
        if kind == "norm":
            if "lines" in mode:
                show = True
            elif name in seen_markers:
                show = False
            else:
                seen_markers.add(name)
                show = True
        kwargs = {
            "x": _decode(trace["x"]),
            "y": _decode(trace["y"]),
            "mode": mode,
            "name": name,
            "showlegend": show,
        }
        if trace.get("marker"):
            kwargs["marker"] = trace["marker"]
        if trace.get("line"):
            kwargs["line"] = trace["line"]
        fig.add_trace(go.Scatter(**kwargs))


def _usable(dense: float) -> bool:
    return 0.5 < dense < 80.0


def _collect(spec, sizes: dict[str, float]) -> dict[str, list[dict]]:
    loaded = load_rows(spec)
    dense = {
        (point["model"], point["dataset"]): float(point["perplexity"])
        for point in loaded
        if point["sparsity"] == 0.0
    }
    buckets = {"wikitext2": [], "ptb": [], "above": []}
    for point in loaded:
        sparsity = float(point["sparsity"])
        if sparsity <= 0:
            continue
        key = (point["model"], point["dataset"])
        base = dense.get(key)
        perplexity = float(point["perplexity"])
        if base is None or not _usable(base) or perplexity <= 0 or not math.isfinite(perplexity):
            continue
        if point["model"] not in sizes:
            continue
        row = {
            "model": point["model"],
            "label": point["model"].split("/")[-1],
            "dataset": point["dataset"],
            "s": sparsity,
            "ratio": perplexity / base,
            "log_ratio": math.log(perplexity / base),
            "n": float(sizes[point["model"]]),
        }
        if point["dataset"] in buckets:
            buckets[point["dataset"]].append(row)
        if row["ratio"] > 1.0:
            buckets["above"].append(row)
    return buckets


def _fit(rows: list[dict]):
    models = {row["model"] for row in rows}
    if len(models) < 2 or len(rows) < 4:
        return None
    alpha, beta, exponent, rmse = solve_scale_free_law(
        np.asarray([row["s"] for row in rows]),
        np.asarray([row["log_ratio"] for row in rows]),
        np.asarray([row["n"] for row in rows]),
    )
    return alpha, beta, exponent, rmse


def _law_name(prefix: str, law) -> str:
    alpha, beta, exponent, rmse = law
    rmse_bit = f"  (RMSE={rmse:.3f})" if prefix.endswith("pooled") or "pooled" in prefix else ""
    if prefix.endswith("pooled"):
        rmse_bit = f"  (RMSE={rmse:.3f})"
    else:
        rmse_bit = ""
    return f"{prefix}: {alpha:.2f}·N^({beta:.2f})·s^{exponent:.2f}{rmse_bit}"


def _draw_family(collapse: go.Figure, normalized: go.Figure, title: str, buckets: dict) -> str:
    palette = FAMILY_COLORS[title]
    above = buckets["above"]
    pooled = _fit(above)
    wt = _fit(buckets["wikitext2"])
    ptb = _fit([row for row in buckets["above"] if row["dataset"] == "ptb"])
    colors = {}
    models = []
    for row_point in above:
        if row_point["model"] not in colors:
            colors[row_point["model"]] = palette[len(models) % len(palette)]
            models.append(row_point["model"])
    if pooled is not None:
        alpha, _beta, exponent, rmse = pooled
        fig.add_trace(
            go.Scatter(
                x=CURVE_S,
                y=alpha * np.power(CURVE_S, exponent),
                mode="lines",
                name=_law_name(f"{title} pooled", pooled),
                line=dict(color="#111827", width=2.4),
            ),
        )
    for dataset, law, dash in (
        ("wikitext2", wt, "dash"),
        ("ptb", ptb, "dot"),
    ):
        if law is None:
            continue
        alpha, _beta, exponent, _rmse = law
        tag = "WT2" if dataset == "wikitext2" else "PTB"
        fig.add_trace(
            go.Scatter(
                x=CURVE_S,
                y=alpha * np.power(CURVE_S, exponent),
                mode="lines",
                name=_law_name(f"{title} {tag}", law),
                line=dict(color="#334155", width=1.6, dash=dash),
            ),
        )
    for model in models:
        color = colors[model]
        label = model.split("/")[-1]
        for dataset, symbol, tag in (
            ("wikitext2", "circle", "WT2"),
            ("ptb", "diamond-open", "PTB"),
        ):
            points = [point for point in buckets["above"] if point["model"] == model and point["dataset"] == dataset]
            if not points or pooled is None:
                continue
            beta = pooled[1]
            fig.add_trace(
                go.Scatter(
                    x=[point["s"] for point in points],
                    y=[point["log_ratio"] * point["n"] ** (-beta) for point in points],
                    mode="markers",
                    name=f"{label} · {tag}",
                    marker=dict(color=color, size=8, symbol=symbol, line=dict(color=color, width=1.2)),
                ),
            )
        sample = next((point for point in above if point["model"] == model), None)
        if sample is None or pooled is None:
            continue
        alpha, beta, exponent, _rmse = pooled
        fig.add_trace(
            go.Scatter(
                x=CURVE_S,
                y=np.exp(alpha * sample["n"] ** beta * np.power(CURVE_S, exponent)),
                mode="lines",
                name=f"{label} law",
                line=dict(color=color, width=1.5),
            ),
        )
        for dataset, symbol in (("wikitext2", "circle"), ("ptb", "diamond-open")):
            points = [point for point in buckets[dataset] if point["model"] == model]
            if not points:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[point["s"] for point in points],
                    y=[point["ratio"] for point in points],
                    mode="markers",
                    name=f"{label} · {'WT2' if dataset == 'wikitext2' else 'PTB'}",
                    showlegend=dataset == "wikitext2",
                    marker=dict(color=color, size=8, symbol=symbol, line=dict(color=color, width=1.2)),
                ),
            )
    if pooled is None:
        return f"{title}: no fit"
    alpha, beta, exponent, rmse = pooled
    text = f"{title}: {alpha:.2f} · N^({beta:.3f}) · s^{exponent:.2f} (RMSE={rmse:.3f})"
    print(text, flush=True)
    if wt:
        print(f"  WT2 {_law_name(title, wt)} n={len(buckets['wikitext2'])}", flush=True)
    if ptb:
        print(f"  PTB {_law_name(title, ptb)} n={sum(1 for row in above if row['dataset']=='ptb')}", flush=True)
    return text


def _place_legends(fig: go.Figure) -> None:
    """One horizontal legend directly under each panel."""
    for row in range(1, 5):
        for col in range(1, 3):
            number = (row - 1) * 2 + col
            x_key = "xaxis" if number == 1 else f"xaxis{number}"
            y_key = "yaxis" if number == 1 else f"yaxis{number}"
            x0, x1 = fig.layout[x_key].domain
            y0, _y1 = fig.layout[y_key].domain
            fig.layout[_legend_name(row, col)] = dict(
                xref="paper",
                yref="paper",
                x=(x0 + x1) / 2,
                y=y0 - 0.008,
                xanchor="center",
                yanchor="top",
                orientation="h",
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="rgba(15,23,42,0.12)",
                borderwidth=1,
                itemsizing="constant",
            )


def write_page() -> None:
    kept = _existing()
    titles = (
        "Qwen · Collapse",
        "Qwen · Normalized ppl/dense",
        "Gemma-4 · Collapse",
        "Gemma-4 · Normalized ppl/dense",
        "OLMo 2 Instruct · Collapse",
        "OLMo 2 Instruct · Normalized ppl/dense",
        "Llama 3 · Collapse",
        "Llama 3 · Normalized ppl/dense",
    )
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    _replay(fig, kept["qwen-collapse"], 1, 1, "Qwen")
    _replay(fig, kept["qwen-norm"], 1, 2, "Qwen")
    _replay(fig, kept["gemma-collapse"], 2, 1, "Gemma-4")
    _replay(fig, kept["gemma-norm"], 2, 2, "Gemma-4")
    olmo_line = _draw_family(fig, "OLMo 2 Instruct", _collect(OLMO2, olmo_stored_params()), 3)
    llama_line = _draw_family(fig, "Llama 3", _collect(LLAMA3, llama_stored_params()), 4)
    for row in range(1, 5):
        fig.update_xaxes(type="log", title_text="Unstructured sparsity s (log)", row=row, col=1)
        fig.update_yaxes(type="log", title_text="N^(−β) · ln(ppl/dense) (log)", row=row, col=1)
        fig.update_xaxes(title_text="Unstructured sparsity s", range=[0, 0.85], row=row, col=2)
        fig.update_yaxes(type="log", title_text="perplexity / dense (log)", row=row, col=2)
    fig.update_layout(
        template="plotly_white",
        width=1480,
        height=2480,
        margin=dict(t=120, b=200, l=70, r=30),
        title=dict(
            text=(
                "<b>Scale-Free Sparsity Law by Model Family</b><br><sup>"
                "Qwen: 15.92 · N^(-0.357) · s^4.74 (RMSE=0.178) &nbsp;|&nbsp; "
                "Gemma-4 WT2: 17.15 · N^(-0.309) · s^4.48 (RMSE=0.137) &nbsp;|&nbsp; "
                f"{olmo_line} &nbsp;|&nbsp; {llama_line}<br>"
                "filled = WikiText2 · open diamond = PTB · C4 calibration</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0f172a"),
    )
    _place_legends(fig)
    fig.write_html(PAGE, include_plotlyjs=True, full_html=True, config={"displaylogo": False, "responsive": True})
    print(f"Wrote {PAGE}", flush=True)


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()
