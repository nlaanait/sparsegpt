"""Qwen 3.5-0.8B sparsity sweep across the implemented pruners.

SparseGPT is the baseline. The other arms are Wanda, Variant A
(input-side pairs), Variant B rescaling, the score-only heuristic, and
Variant D sparse-plus-low-rank. The Fisher rung is omitted: the layer
loop does not collect loss gradients, so ``tangent_fisher`` cannot prune.

Sparsities run from 0.5 through 0.9 in tenths, then 0.99. Each point is
one C4-calibrated prune and a WikiText2 perplexity. The JSON is resumed
on a later launch. Host memory is checked before every point and the run
stops inside the cushion under the 110 GiB brick.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from sparsifier.eval.live_stages import BRICK_GIB, CUSHION_GIB, read_host_memory_gib
from sparsifier.metrics import PPL_RE
from sparsifier.sweep import repo_root, sparsity_token

MODEL = "Qwen/Qwen3.5-0.8B"
CALIBRATION = "c4"
DATASET = "wikitext2"
SPARSITIES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
METHODS = (
    "sparsegpt",
    "wanda",
    "tangent_input_side",
    "tangent_rescale",
    "tangent_heuristic",
    "tangent_sparse_lowrank",
)
PAIRS = Path("results/tangent/qwen35-0.8b-concept-pairs")
RESULT = Path("results/tangent/qwen35-0.8b-method-sweep.json")
LOG_DIR = Path("results/tangent/logs")


def method_flags(method: str) -> list[str]:
    """Extra CLI flags for one pruner. SparseGPT uses the default."""
    if method == "sparsegpt":
        return []
    if method == "wanda":
        return ["--pruner", "wanda"]
    if method == "tangent_input_side":
        return [
            "--pruner",
            "tangent_input_side",
            "--alpha-in",
            "1",
            "--pair-deltas",
            str(PAIRS),
        ]
    if method == "tangent_rescale":
        return ["--pruner", "tangent_rescale", "--tangent-alpha", "1"]
    if method == "tangent_heuristic":
        return ["--pruner", "tangent_heuristic", "--tangent-alpha", "1"]
    if method == "tangent_sparse_lowrank":
        return ["--pruner", "tangent_sparse_lowrank"]
    raise ValueError(f"unknown method {method}")


def point_command(method: str, sparsity: float, *, python: str) -> list[str]:
    """One prune-and-eval invocation."""
    return [
        python,
        "-u",
        "-m",
        "drivers.capped_sparsifier",
        MODEL,
        CALIBRATION,
        "--seed",
        "0",
        "--nsamples",
        "128",
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        DATASET,
        "--pytorch-attention",
        "--forward-batch",
        "1",
        *method_flags(method),
    ]


def pending_points(results: dict) -> list[tuple[str, float]]:
    """Methods then sparsities, skipping points that already have a perplexity."""
    have = {
        (point["method"], float(point["sparsity"]))
        for point in results.get("points", [])
        if point.get("dataset") == DATASET and point.get("perplexity") == point.get("perplexity")
    }
    return [
        (method, sparsity)
        for method in METHODS
        for sparsity in SPARSITIES
        if (method, sparsity) not in have
    ]


def load_results(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": MODEL,
        "calibration": CALIBRATION,
        "dataset": DATASET,
        "sparsities": SPARSITIES,
        "methods": list(METHODS),
        "points": [],
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def assert_room(where: str) -> None:
    total, available = read_host_memory_gib()
    used = total - available
    print(
        f"memory [{where}]: used={used:.1f} GiB available={available:.1f} GiB brick={BRICK_GIB:.0f}",
        flush=True,
    )
    if used >= BRICK_GIB - CUSHION_GIB:
        raise RuntimeError(
            f"stopping at {where}: host is using {used:.1f} GiB, inside the cushion under {BRICK_GIB:.0f} GiB"
        )


def _perplexity(text: str) -> float:
    found = None
    dataset = None
    for line in text.splitlines():
        if line.startswith("Dataset:"):
            dataset = line.split(":", 1)[1].strip()
            continue
        match = PPL_RE.search(line)
        if match and dataset == DATASET:
            found = float(match.group(1))
    if found is None or found != found or found == float("inf"):
        raise RuntimeError("WikiText2 perplexity line was missing or not a finite number")
    return found


def run_point(method: str, sparsity: float, log_path: Path) -> float:
    python = os.environ.get("SPARSIFIER_PYTHON", sys.executable)
    command = point_command(method, sparsity, python=python)
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONPATH", "src:.")
    process = subprocess.Popen(
        command,
        cwd=repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines: list[str] = []
    stop = threading.Event()

    def watch_memory() -> None:
        while not stop.wait(30):
            total, available = read_host_memory_gib()
            used = total - available
            print(
                f"memory [{method} {sparsity}]: used={used:.1f} GiB available={available:.1f} GiB",
                flush=True,
            )
            if used >= BRICK_GIB - CUSHION_GIB:
                print(f"stopping {method} {sparsity}: host is using {used:.1f} GiB", flush=True)
                os.killpg(process.pid, signal.SIGTERM)

    watcher = threading.Thread(target=watch_memory, daemon=True)
    watcher.start()
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
    code = process.wait()
    stop.set()
    watcher.join(timeout=1)
    if code != 0:
        raise RuntimeError(f"{method} sparsity {sparsity} exited {code}. Log: {log_path}")
    return _perplexity("".join(lines))


def main() -> None:
    os.chdir(repo_root())
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    lock_path = RESULT.with_suffix(".lock")
    lock = lock_path.open("a", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    results = load_results(RESULT)
    missing = pending_points(results)
    print(f"{len(missing)} points remaining", flush=True)
    for method, sparsity in missing:
        assert_room(f"{method}-{sparsity}")
        log_path = LOG_DIR / f"qwen35-0.8b_{method}_{sparsity_token(sparsity)}_wikitext2.log"
        perplexity = run_point(method, sparsity, log_path)
        results["points"].append(
            {
                "model": MODEL,
                "dataset": DATASET,
                "method": method,
                "sparsity": sparsity,
                "perplexity": perplexity,
            }
        )
        save_results(RESULT, results)
        print(f"saved {method} {sparsity} perplexity={perplexity:.4f}", flush=True)
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
