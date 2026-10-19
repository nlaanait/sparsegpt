"""Perplexity vs model size for the Qwen3.5 family.

Matches Figure 2 of SparseGPT (Frantar & Alistarh, 2023): raw-WikiText2
perplexity for the dense model and for SparseGPT at 50% unstructured,
4:8, and 2:4 sparsity. Calibration is 128 C4 segments.
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
# Legend order matches Figure 2.
METHODS = ["2:4", "4:8", "50% Unstructured", "Dense"]
METHOD_STYLE = {
    "2:4": dict(color="#d62728", symbol="circle"),
    "4:8": dict(color="#9467bd", symbol="diamond"),
    "50% Unstructured": dict(color="#ff7f0e", symbol="square"),
    "Dense": dict(color="#2ca02c", symbol="circle-open"),
}
PPL_RE = re.compile(
    r"Perplexity:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?|inf|nan)",
    re.I,
)
FIGURE1 = ROOT / "figures" / "qwen3.5-2b_wikitext2.json"


def result_path(output: Path) -> Path:
    return output / "qwen3.5_family_wikitext2.json"


def load_results(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "eval": "raw-WikiText2",
        "calibration": "c4",
        "params_b": {},
        "points": [],
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(path)


def has_point(results: dict, model: str, method: str) -> bool:
    return any(
        point["model"] == model and point["method"] == method
        for point in results["points"]
    )


def params_billions(model: str) -> float:
    from huggingface_hub import model_info

    info = model_info(model)
    total = info.safetensors.total
    return total / 1e9


def seed_known_points(results: dict) -> None:
    """Reuse the Qwen3.5-2B dense and 50% numbers from the Figure 1 sweep."""
    if not FIGURE1.is_file():
        return
    data = json.loads(FIGURE1.read_text())
    wanted = {
        ("dense", 0.0): "Dense",
        ("sparsegpt", 0.5): "50% Unstructured",
    }
    model = "Qwen/Qwen3.5-2B"
    for point in data["points"]:
        method = wanted.get((point["method"], point["sparsity"]))
        if method is None or has_point(results, model, method):
            continue
        if not math.isfinite(point["perplexity"]):
            continue
        results["points"].append(
            {"model": model, "method": method, "perplexity": point["perplexity"]}
        )


def run_qwen(model: str, method: str, log_path: Path) -> float:
    short = model.split("/")[-1].lower()
    cmd = [
        sys.executable,
        "-m",
        "sparsegpt.qwen",
        model,
        "c4",
        "--eval-datasets",
        "wikitext2",
    ]
    if method == "Dense":
        cmd.extend(["--sparsity", "0"])
    elif method == "50% Unstructured":
        cmd.extend(["--sparsity", "0.5", "--save", f"sparse_opt/{short}"])
    elif method == "4:8":
        cmd.extend(["--prunen", "4", "--prunem", "8", "--save", f"sparse_opt/{short}"])
    elif method == "2:4":
        cmd.extend(["--prunen", "2", "--prunem", "4", "--save", f"sparse_opt/{short}"])
    else:
        raise ValueError(method)
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


def _rows(results: dict, method: str) -> list[tuple[float, float]]:
    params = results.get("params_b", {})
    rows = []
    for point in results["points"]:
        if point["method"] != method or not math.isfinite(point["perplexity"]):
            continue
        billions = params.get(point["model"])
        if billions is None:
            continue
        rows.append((billions, point["perplexity"]))
    return sorted(rows)


def plot_results(results: dict, output: Path) -> Path:
    import plotly.graph_objects as go

    fig = go.Figure()
    ys = []
    for method in METHODS:
        rows = _rows(results, method)
        if not rows:
            continue
        xs, method_ys = zip(*rows)
        ys.extend(method_ys)
        style = METHOD_STYLE[method]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=method_ys,
                mode="lines+markers",
                name=method,
                line=dict(color=style["color"]),
                marker=dict(symbol=style["symbol"], size=9),
            )
        )
    params = results.get("params_b", {})
    tickvals = [params[model] for model in MODELS if model in params]
    ticktext = [f"{value:.3g}" for value in tickvals]
    y0 = min(ys) * 0.9 if ys else 0
    y1 = max(ys) * 1.08 if ys else 1
    fig.update_layout(
        template="plotly_white",
        width=720,
        height=480,
        margin=dict(t=36, b=48, l=72, r=24),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        xaxis=dict(
            title="#Params in Billions",
            type="log",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(title="Perplexity on raw-WikiText2", range=[y0, y1]),
    )
    output.mkdir(parents=True, exist_ok=True)
    html = output / "qwen3.5_family_perplexity.html"
    fig.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the figure from saved perplexities.",
    )
    args = parser.parse_args()
    path = result_path(args.output)
    results = load_results(path)
    results.setdefault("params_b", {})
    results.setdefault("points", [])
    if not args.plot_only:
        seed_known_points(results)
        for model in MODELS:
            if model not in results["params_b"]:
                results["params_b"][model] = params_billions(model)
                print(f"{model} params_b={results['params_b'][model]:.4f}", flush=True)
        save_results(path, results)
        logs = args.output / "logs"
        for model in MODELS:
            short = model.split("/")[-1].lower()
            for method in ("Dense", "50% Unstructured", "4:8", "2:4"):
                if has_point(results, model, method):
                    print(f"skip {model} {method}", flush=True)
                    continue
                slug = method.lower().replace("%", "").replace(" ", "").replace(":", "")
                log_path = logs / f"{short}_{slug}.log"
                try:
                    perplexity = run_qwen(model, method, log_path)
                except RuntimeError as exc:
                    print(f"failed {model} {method}: {exc}", flush=True)
                    continue
                results["points"].append(
                    {"model": model, "method": method, "perplexity": perplexity}
                )
                save_results(path, results)
                print(f"{model} {method} perplexity={perplexity:.4f}", flush=True)
                try:
                    plot_results(results, args.output)
                except ModuleNotFoundError:
                    print("plotly is not installed; results were saved, plot skipped")
    if not results["points"]:
        raise SystemExit(f"No perplexities in {path}")
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
