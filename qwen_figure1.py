"""Sparsity vs raw-WikiText2 perplexity for Qwen3.5-2B.

Matches Figure 1 of SparseGPT (Frantar & Alistarh, 2023): uniform
per-layer sparsity for magnitude pruning and SparseGPT, plus the dense
baseline. Calibration is 128 C4 segments, the same setup as `demo-qwen`.
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
MODEL = "Qwen/Qwen3.5-2B"
SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PPL_RE = re.compile(r"Perplexity:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?|inf|nan)", re.I)


def result_path(output: Path) -> Path:
    return output / "qwen3.5-2b_wikitext2.json"


def load_results(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"model": MODEL, "eval": "raw-WikiText2", "calibration": "c4", "points": []}


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(path)


def has_point(results: dict, method: str, sparsity: float) -> bool:
    return any(
        point["method"] == method and abs(point["sparsity"] - sparsity) < 1e-9
        for point in results["points"]
    )


def run_qwen(sparsity: float, gmp: bool, log_path: Path) -> float:
    cmd = [
        sys.executable,
        "-m",
        "sparsegpt.qwen",
        MODEL,
        "c4",
        "--sparsity",
        f"{sparsity:.1f}",
        "--eval-datasets",
        "wikitext2",
    ]
    if sparsity > 0 and not gmp:
        cmd.extend(["--save", "sparse_opt/qwen3.5-2b"])
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
        raise SystemExit(f"qwen.py exited {code}. Log: {log_path}")
    matches = PPL_RE.findall("".join(lines))
    if not matches:
        raise SystemExit(f"No perplexity line in {log_path}")
    return float(matches[-1])


def _series(results: dict, method: str) -> list[tuple[float, float]]:
    rows = [
        (point["sparsity"], point["perplexity"])
        for point in results["points"]
        if point["method"] == method and math.isfinite(point["perplexity"])
    ]
    return sorted(rows)


def plot_results(results: dict, output: Path) -> Path:
    import plotly.graph_objects as go

    dense = _series(results, "dense")
    magnitude = _series(results, "magnitude")
    sparsegpt = _series(results, "sparsegpt")
    if dense:
        # Figure 1 draws the dense model as a horizontal reference.
        dense_y = dense[0][1]
        magnitude = [(0.0, dense_y)] + magnitude
        sparsegpt = [(0.0, dense_y)] + sparsegpt

    focus = [y for _, y in dense + sparsegpt]
    mag_y = [y for _, y in magnitude]
    reference = max(focus) if focus else (max(mag_y) if mag_y else 1.0)
    # Keep the axis on the SparseGPT/dense range, as in Figure 1, but include
    # magnitude points that have not yet collapsed.
    cap = reference * 2.5
    offscale = [(s, y) for s, y in magnitude if y > cap]
    in_frame = focus + [y for y in mag_y if y <= cap]
    top = max(in_frame) if in_frame else 1.0
    bottom = min(in_frame) if in_frame else 0.0
    span = top - bottom
    pad = span * 0.12 if span else top * 0.05
    y0 = 0.0 if bottom - pad < 0 else bottom - pad
    y1 = top + pad * 0.35

    fig = go.Figure()
    if magnitude:
        xs, ys = zip(*magnitude)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="Magnitude",
                line=dict(color="#1f77b4"),
                marker=dict(symbol="circle", size=8),
            )
        )
    if sparsegpt:
        xs, ys = zip(*sparsegpt)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="SparseGPT",
                line=dict(color="#ff7f0e"),
                marker=dict(symbol="square", size=8),
            )
        )
    if dense:
        fig.add_trace(
            go.Scatter(
                x=[-0.02, 0.82],
                y=[dense_y, dense_y],
                mode="lines",
                name="Dense",
                line=dict(color="#2ca02c", dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=720,
        height=480,
        margin=dict(t=48, b=48, l=72, r=24),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        xaxis=dict(
            title="Sparsity",
            range=[-0.02, 0.82],
            tickmode="array",
            tickvals=[i / 10 for i in range(9)],
        ),
        yaxis=dict(title="Perplexity on raw-WikiText2", range=[y0, y1]),
    )
    if offscale:
        note = ", ".join(f"{s:.1f}: {y:.3g}" for s, y in offscale)
        fig.add_annotation(
            text="Magnitude above axis: " + note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.06,
            showarrow=False,
            xanchor="left",
            font=dict(size=11),
        )
    output.mkdir(parents=True, exist_ok=True)
    html = output / "qwen3.5-2b_sparsity_perplexity.html"
    fig.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")
    return html


def jobs() -> list[tuple[str, float, bool]]:
    planned = [("dense", 0.0, False)]
    # Magnitude evals are forwards only. SparseGPT reuses sparse_opt on a match.
    planned.extend(("magnitude", sparsity, True) for sparsity in SPARSITIES)
    planned.append(("sparsegpt", 0.5, False))
    planned.extend(
        ("sparsegpt", sparsity, False) for sparsity in SPARSITIES if sparsity != 0.5
    )
    return planned


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
    if not args.plot_only:
        logs = args.output / "logs"
        for method, sparsity, gmp in jobs():
            if has_point(results, method, sparsity):
                print(f"skip {method} sparsity={sparsity:.1f}", flush=True)
                continue
            log_path = logs / f"{method}_sparsity{sparsity:.1f}.log"
            perplexity = run_qwen(sparsity, gmp, log_path)
            results["points"].append(
                {"method": method, "sparsity": sparsity, "perplexity": perplexity}
            )
            save_results(path, results)
            print(f"{method} sparsity={sparsity:.1f} perplexity={perplexity:.4f}", flush=True)
            try:
                plot_results(results, args.output)
            except ModuleNotFoundError:
                print("plotly is not installed; results were saved, plot skipped")
    if not results["points"]:
        raise SystemExit(f"No perplexities in {path}")
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
