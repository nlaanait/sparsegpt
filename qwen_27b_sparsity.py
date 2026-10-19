"""Missing Qwen3.5-27B unstructured SparseGPT points.

The tenth-grid points 0.3, 0.4, 0.6, 0.7, and 0.8 never landed: WikiText2
eval of 0.3 was killed, and PTB was only run for checkpoints that already
existed (0.1, 0.2, 0.5). Calibration stays C4. PTB is measured first. Each
job uses the Atlas-Inference/gdn SM121 kernel (transformers use_kernels).

The watcher kills a run whose resident memory is large, whose log has gone
quiet, and whose GPU SMs have stayed at 0% utilization. That is the FLA
stall: memory held, no kernels running.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from sparsegpt.prune_cache import find_pruned_model, prune_config_from_args
from qwen_figure1_family import (
    PPL_RE,
    ROOT,
    add_point,
    load_results,
    lookup,
    save_results,
    sparsity_token,
)
from qwen_ptb import _prune_args, load_ptb_results

MODEL = "Qwen/Qwen3.5-27B"
MISSING = [0.3, 0.4, 0.6, 0.7, 0.8]
# Four polls of 10s. A resident model, a quiet log, and 0% SM utilization
# across that window is the stall: memory held, no kernels running.
POLL_S = 10
STALL_POLLS = 4
RSS_GIB = 8
ARM_MARKERS = ("Starting...", "Pruning ...", "Evaluating ...", "Dataset:")


def result_path(dataset: str) -> Path:
    if dataset == "ptb":
        return ROOT / "figures" / "qwen3.5_family_sparsity_ptb.json"
    if dataset == "wikitext2":
        return ROOT / "figures" / "qwen3.5_family_sparsity_wikitext2.json"
    raise ValueError(dataset)


def open_results(dataset: str) -> dict:
    path = result_path(dataset)
    if dataset == "ptb":
        return load_ptb_results(path)
    results = load_results(path)
    results.setdefault("points", [])
    return results


def checkpoint(sparsity: float) -> Path | None:
    return find_pruned_model(prune_config_from_args(_prune_args(MODEL, sparsity)))


def gpu_util() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    line = out.strip().splitlines()
    if not line:
        return None
    try:
        return int(line[0].strip())
    except ValueError:
        return None


def rss_gib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return 0.0
    return 0.0


def run_one(dataset: str, sparsity: float, log_path: Path) -> float:
    cached = checkpoint(sparsity)
    cmd = [
        sys.executable,
        "-m",
        "sparsegpt.qwen",
        MODEL,
        "c4",
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        dataset,
    ]
    if cached is None:
        cmd.extend(["--save", "sparse_opt/qwen3.5-27b"])
    else:
        cmd.append("--load-cached")
        print(f"checkpoint {cached}", flush=True)
    print("+", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: list[str] = []
    state = {"last": time.time(), "armed": False}

    def reader() -> None:
        assert proc.stdout is not None
        with log_path.open("w") as log:
            for line in proc.stdout:
                state["last"] = time.time()
                if any(marker in line for marker in ARM_MARKERS):
                    state["armed"] = True
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                lines.append(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    utils: list[int] = []
    while proc.poll() is None:
        time.sleep(POLL_S)
        util = gpu_util()
        rss = rss_gib(proc.pid)
        silent = time.time() - state["last"]
        if util is not None:
            utils.append(util)
            utils = utils[-STALL_POLLS:]
        print(
            f"monitor util={util} rss={rss:.1f}GiB silent={silent:.0f}s armed={state['armed']}",
            flush=True,
        )
        sm_idle = len(utils) == STALL_POLLS and all(sample == 0 for sample in utils)
        if state["armed"] and sm_idle and rss >= RSS_GIB and silent >= POLL_S * STALL_POLLS:
            message = (
                "STALL: GPU utilization stayed at 0% with the model resident "
                f"and no log progress for {POLL_S * STALL_POLLS}s. Killing.\n"
            )
            print(message, flush=True)
            with log_path.open("a") as log:
                log.write(message)
            os.killpg(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            thread.join(timeout=10)
            raise RuntimeError(message.strip())
    thread.join(timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"qwen.py exited {proc.returncode}. Log: {log_path}")
    matches = PPL_RE.findall("".join(lines))
    if not matches:
        raise RuntimeError(f"No perplexity line in {log_path}")
    return float(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["ptb", "wikitext2", "all"], default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    phases = ["ptb", "wikitext2"] if args.phase == "all" else [args.phase]
    logs = ROOT / "figures" / "logs"
    for dataset in phases:
        path = result_path(dataset)
        results = open_results(dataset)
        for sparsity in MISSING:
            token = sparsity_token(sparsity)
            existing = lookup(results, MODEL, "sparsegpt", sparsity)
            if existing is not None:
                print(
                    f"skip {dataset} sparsity={token} perplexity={existing:.4f}",
                    flush=True,
                )
                continue
            log_path = logs / f"qwen3.5-27b_{dataset}_pytorch_sparsity{token}.log"
            perplexity = run_one(dataset, sparsity, log_path)
            add_point(results, MODEL, "sparsegpt", sparsity, perplexity)
            save_results(path, results)
            print(
                f"{MODEL} {dataset} sparsity={token} perplexity={perplexity:.4f}",
                flush=True,
            )
            from sparsity_principle import write_page

            write_page()


if __name__ == "__main__":
    main()
