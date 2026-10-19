"""PTB perplexity versus sparsity for cached Qwen3.5 SparseGPT checkpoints.

Calibration stays the C4 run (128 x 2048, seed 0). Weights are loaded from
sparse_opt and evaluated on PTB. A missing checkpoint is an error; this
script does not prune. Magnitude pruning is omitted because those runs were
not saved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from sparsegpt.prune_cache import find_pruned_model, prune_config_from_args
from qwen_figure1_family import (
    MODELS,
    PPL_RE,
    ROOT,
    add_point,
    load_results,
    lookup,
    plot_results,
    save_results,
    sparsity_token,
)


def result_path(output: Path) -> Path:
    return output / "qwen3.5_family_sparsity_ptb.json"


def load_ptb_results(path: Path) -> dict:
    results = load_results(path)
    results["eval"] = "PTB"
    results["calibration"] = "c4"
    results.setdefault("points", [])
    return results


def _prune_args(model: str, sparsity: float) -> Namespace:
    return Namespace(
        model=model,
        dataset="c4",
        seed=0,
        nsamples=128,
        percdamp=0.01,
        sparsity=sparsity,
        prunen=0,
        prunem=0,
        blocksize=128,
        gmp=False,
        wbits=16,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        true_sequential=False,
        geometry_alpha=0.0,
        geometry_probes=None,
    )


def cached_sparse_points() -> list[tuple[str, float, Path]]:
    """Unstructured C4 SparseGPT checkpoints, smallest model and sparsity first."""
    rows: list[tuple[str, float, Path]] = []
    seen: set[tuple[str, float]] = set()
    for cfg_path in sorted((ROOT / "sparse_opt").glob("*/prune_config.json")):
        config = json.loads(cfg_path.read_text())
        model = config.get("model")
        sparsity = float(config.get("sparsity") or 0.0)
        if model not in MODELS or sparsity <= 0:
            continue
        if config.get("prunen") or config.get("prunem") or config.get("gmp"):
            continue
        if config.get("dataset") != "c4" or config.get("nsamples") != 128:
            continue
        if config.get("seed") != 0 or config.get("wbits", 16) < 16:
            continue
        if config.get("geometry_alpha"):
            continue
        key = (model, sparsity)
        if key in seen:
            continue
        located = find_pruned_model(prune_config_from_args(_prune_args(model, sparsity)))
        if located is None:
            raise SystemExit(
                f"Checkpoint {cfg_path.parent} does not match the loader config."
            )
        seen.add(key)
        rows.append((model, sparsity, located))
    rows.sort(key=lambda row: (MODELS.index(row[0]), row[1]))
    return rows


def run_qwen(model: str, sparsity: float, log_path: Path) -> float:
    cmd = [
        sys.executable,
        "-m",
        "sparsegpt.qwen",
        model,
        "c4",
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        "ptb",
    ]
    if sparsity > 0:
        cmd.append("--load-cached")
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


def refresh_plots(results: dict, output: Path) -> None:
    if not any(point["method"] == "sparsegpt" for point in results["points"]):
        return
    plot_results(results, output, y_title="Perplexity on PTB", tag="ptb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    path = result_path(args.output)
    results = load_ptb_results(path)
    points = cached_sparse_points()
    if not points:
        raise SystemExit("No unstructured SparseGPT checkpoints under sparse_opt.")
    print(f"{len(points)} cached unstructured checkpoints", flush=True)
    if not args.plot_only:
        logs = args.output / "logs"
        by_model: dict[str, list[tuple[float, Path]]] = {model: [] for model in MODELS}
        for model, sparsity, located in points:
            by_model[model].append((sparsity, located))
        for model in MODELS:
            short = model.split("/")[-1].lower()
            if lookup(results, model, "dense", 0.0) is None:
                log_path = logs / f"{short}_ptb_dense.log"
                perplexity = run_qwen(model, 0.0, log_path)
                add_point(results, model, "dense", 0.0, perplexity)
                save_results(path, results)
                print(f"{model} dense perplexity={perplexity:.4f}", flush=True)
            else:
                print(
                    f"skip {model} dense perplexity={lookup(results, model, 'dense', 0.0):.4f}",
                    flush=True,
                )
            for sparsity, located in by_model[model]:
                existing = lookup(results, model, "sparsegpt", sparsity)
                token = sparsity_token(sparsity)
                if existing is not None:
                    print(
                        f"skip {model} sparsegpt sparsity={token} perplexity={existing:.4f}",
                        flush=True,
                    )
                    continue
                print(f"checkpoint {located}", flush=True)
                log_path = logs / f"{short}_ptb_sparsegpt_sparsity{token}.log"
                perplexity = run_qwen(model, sparsity, log_path)
                if not math.isfinite(perplexity):
                    raise RuntimeError(f"Non-finite perplexity for {model} at {token}")
                add_point(results, model, "sparsegpt", sparsity, perplexity)
                save_results(path, results)
                print(
                    f"{model} sparsegpt sparsity={token} perplexity={perplexity:.4f}",
                    flush=True,
                )
                refresh_plots(results, args.output)
        save_results(path, results)
    if not any(point["method"] == "sparsegpt" for point in results["points"]):
        raise SystemExit(f"No PTB perplexities in {path}")
    refresh_plots(results, args.output)


if __name__ == "__main__":
    main()
