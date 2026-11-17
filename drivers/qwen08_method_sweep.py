"""Qwen 3.5-0.8B sparsity sweep across the implemented pruners.

SparseGPT is the baseline. The other arms are Wanda, Variant A
(input-side pairs), Variant B rescaling, the score-only heuristic, and
Variant D sparse-plus-low-rank. The Fisher rung is omitted: the layer
loop does not collect loss gradients, so ``tangent_fisher`` cannot prune.

Sparsities run from 0.5 through 0.9 in tenths, then 0.99. Each point is
one C4-calibrated prune and a WikiText2 perplexity. Resumption requires matching
source, child-runtime, and pair-bank provenance in a versioned result namespace.
Historical unversioned results are retained separately. Host memory is checked before every point and the run
stops inside the cushion under the 110 GiB brick.
"""

from __future__ import annotations

import fcntl
import argparse
import hashlib
import json
import math
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
PROTOCOL = "qwen08-method-sweep-curvature-trace-v2"
RESULT = Path(f"results/tangent/{PROTOCOL}.json")


def current_provenance() -> dict:
    """Bind cached points to the actual child runtime and pruning sources.

    This guards code/runtime/pair-bank reuse. It does not establish immutable
    Hub references or identical GPU hardware and dispatched backends.
    """
    root = repo_root()
    paths = set()
    for directory in ("prune", "models", "data", "metrics"):
        paths.update((root / "src/sparsifier" / directory).rglob("*.py"))
    paths.update(root / p for p in (
        "src/sparsifier/cli.py", "src/sparsifier/eval/live_stages.py",
        "drivers/capped_sparsifier.py", "drivers/qwen08_method_sweep.py"))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(paths) if p.is_file()}
    pair_files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((root / PAIRS).rglob("*"))
                  if p.is_file() and p.suffix in {".npy", ".json"}}
    python = os.environ.get("SPARSIFIER_PYTHON", sys.executable)
    script = """import importlib.metadata as m, json, sys
packages = {}
for name in ('torch','transformers','numpy','scipy','datasets','safetensors','accelerate','kernels','flash-linear-attention','causal-conv1d'):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':packages}))
"""
    runtime = json.loads(subprocess.check_output([python, "-c", script], text=True))
    return {"protocol": PROTOCOL, "sources": sources, "pair_files": pair_files,
            "runtime": runtime,
            "commands": {m: point_command(m, .7, python=python) for m in METHODS}}


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
    """Plan points after load_results has validated implementation provenance."""
    have = {
        (point["method"], float(point["sparsity"]))
        for point in results.get("points", [])
        if point.get("dataset") == DATASET
        and isinstance(point.get("perplexity"), (int, float))
        and not isinstance(point.get("perplexity"), bool)
        and math.isfinite(point["perplexity"]) and point["perplexity"] > 0
    }
    return [
        (method, sparsity)
        for method in METHODS
        for sparsity in SPARSITIES
        if (method, sparsity) not in have
    ]


def load_results(path: Path) -> dict:
    provenance = current_provenance()
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {"model": MODEL, "calibration": CALIBRATION, "dataset": DATASET,
                    "sparsities": SPARSITIES, "methods": list(METHODS)}
        if result.get("provenance") != provenance or any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("method-sweep provenance differs or is missing; use a new --output path; "
                             "historical points cannot be relabeled as current measurements")
        return result
    return {
        "model": MODEL,
        "calibration": CALIBRATION,
        "dataset": DATASET,
        "sparsities": SPARSITIES,
        "methods": list(METHODS),
        "provenance": provenance,
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
    if found is None or not math.isfinite(found) or found <= 0:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT,
                        help="Versioned result namespace; legacy unversioned results cannot resume.")
    args = parser.parse_args()
    os.chdir(repo_root())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(".lock")
    lock = lock_path.open("a", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    results = load_results(args.output)
    missing = pending_points(results)
    print(f"{len(missing)} points remaining", flush=True)
    for method, sparsity in missing:
        if current_provenance() != results["provenance"]:
            raise RuntimeError("pruning code, pair bank, or runtime changed during the sweep; use a new --output")
        assert_room(f"{method}-{sparsity}")
        log_path = args.output.with_name(args.output.name + ".artifacts") / "logs" / f"{method}_{sparsity_token(sparsity)}_wikitext2.log"
        perplexity = run_point(method, sparsity, log_path)
        if current_provenance() != results["provenance"]:
            raise RuntimeError("pruning code, pair bank, or runtime changed while the point ran; result not saved")
        results["points"].append(
            {
                "model": MODEL,
                "dataset": DATASET,
                "method": method,
                "sparsity": sparsity,
                "perplexity": perplexity,
            }
        )
        save_results(args.output, results)
        print(f"saved {method} {sparsity} perplexity={perplexity:.4f}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
