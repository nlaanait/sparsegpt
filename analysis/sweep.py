"""Calibration then eval sweeps for one model family.

Default protocol is C4 calibration with PTB and WikiText2 eval. A SweepSpec
may swap those (for example WikiText2 calibration with C4 and PTB eval).
A prune is saved with --skip-eval and evaluated with --load-cached. The
watcher kills a run only when it is armed, the SMs stay at 0%, the log is
quiet, resident memory is flat, and the process is not writing a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from qwen_figure1_family import PPL_RE, ROOT
from qwen_ptb import _prune_args
from sparsegpt.prune_cache import find_pruned_model, prune_config_from_args

from experiments.qwen import SweepSpec

SPARSITIES = [round(i / 10, 1) for i in range(1, 9)]
# Legacy alias for the default Gemma / C4-calibrated Qwen eval pair.
DATASETS = ["ptb", "wikitext2"]
POLL_S = 10
STALL_POLLS = 12
RSS_GIB = 8
ARM_MARKERS = ("Starting...", "Pruning ...", "Evaluating ...", "Dataset:")


def short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


def sparsity_token(sparsity: float) -> str:
    return f"{sparsity:.2f}".rstrip("0").rstrip(".")


def result_path(spec: SweepSpec, model: str) -> Path:
    return spec.result_dir / f"{short_name(model)}_sparsity.json"


def load_results(path: Path, spec: SweepSpec) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "calibration": spec.calibration,
        "attention": spec.attention,
        "points": [],
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(path)


def lookup(results: dict, dataset: str, method: str, sparsity: float) -> float | None:
    for point in results["points"]:
        if (
            point["dataset"] == dataset
            and point["method"] == method
            and abs(point["sparsity"] - sparsity) < 1e-9
        ):
            return point["perplexity"]
    return None


def add_point(results, model, dataset, method, sparsity, perplexity) -> None:
    if lookup(results, dataset, method, sparsity) is not None:
        return
    if perplexity != perplexity:
        return
    results["points"].append(
        {
            "model": model,
            "dataset": dataset,
            "method": method,
            "sparsity": sparsity,
            "perplexity": perplexity,
        }
    )


def checkpoint(model: str, sparsity: float, calibration: str = "c4") -> Path | None:
    return find_pruned_model(
        prune_config_from_args(_prune_args(model, sparsity, dataset=calibration))
    )


def release_finished_checkpoints(spec: SweepSpec) -> None:
    """Remove a checkpoint once every eval corpus is saved.

    Large 27B / Gemma caches do not fit for a full sparsity ladder. The JSON
    point is the result. A missing checkpoint is pruned again only when an
    eval point is still absent.
    """
    if spec.name not in (
        "gemma",
        "olmo2",
        "qwen_wikitext2_cal",
        "qwen3.5_wikitext2_cal",
    ):
        return
    for model in spec.models:
        path = result_path(spec, model)
        if not path.exists():
            continue
        results = load_results(path, spec)
        for sparsity in SPARSITIES:
            if any(
                lookup(results, dataset, "sparsegpt", sparsity) is None
                for dataset in spec.eval_datasets
            ):
                continue
            cached = checkpoint(model, sparsity, spec.calibration)
            if cached is None:
                continue
            print(f"removing {cached} after both evals", flush=True)
            shutil.rmtree(cached)


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


def parse_perplexities(text: str) -> dict[str, float]:
    found: dict[str, float] = {}
    dataset = None
    for line in text.splitlines():
        if line.startswith("Dataset:"):
            dataset = line.split(":", 1)[1].strip()
            continue
        match = PPL_RE.search(line)
        if match and dataset is not None:
            found[dataset] = float(match.group(1))
            dataset = None
    return found


def run_job(cmd: list[str], log_path: Path, *, expect_perplexity: bool = True) -> dict[str, float]:
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
    state = {"last": time.time(), "armed": False, "line": ""}

    def reader() -> None:
        assert proc.stdout is not None
        with log_path.open("w") as log:
            for line in proc.stdout:
                state["last"] = time.time()
                state["line"] = line
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
    rss_samples: list[float] = []
    while proc.poll() is None:
        time.sleep(POLL_S)
        util = gpu_util()
        rss = rss_gib(proc.pid)
        silent = time.time() - state["last"]
        if util is not None:
            utils.append(util)
            utils = utils[-STALL_POLLS:]
        rss_samples.append(rss)
        rss_samples = rss_samples[-STALL_POLLS:]
        print(
            f"monitor util={util} rss={rss:.1f}GiB silent={silent:.0f}s armed={state['armed']}",
            flush=True,
        )
        sm_idle = len(utils) == STALL_POLLS and all(sample == 0 for sample in utils)
        rss_flat = len(rss_samples) == STALL_POLLS and max(rss_samples) - min(rss_samples) < 1.0
        saving = any(token in state["line"] for token in ("Saving pruned", "Writing model shards"))
        if (
            state["armed"]
            and sm_idle
            and rss_flat
            and not saving
            and rss >= RSS_GIB
            and silent >= POLL_S * STALL_POLLS
        ):
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
        raise RuntimeError(f"{cmd[2]} exited {proc.returncode}. Log: {log_path}")
    if not expect_perplexity:
        return {}
    found = parse_perplexities("".join(lines))
    if not found:
        raise RuntimeError(f"No perplexity line in {log_path}")
    return found


def family_cmd(spec: SweepSpec, model: str, datasets: list[str], sparsity: float) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        spec.runner,
        model,
        spec.calibration,
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        *datasets,
        *spec.extra_args,
    ]
    if sparsity > 0:
        cached = checkpoint(model, sparsity, spec.calibration)
        if cached is None:
            cmd.extend(["--save", f"sparse_opt/{short_name(model)}"])
        else:
            cmd.append("--load-cached")
            print(f"checkpoint {cached}", flush=True)
    return cmd


def run_smoke_dense(
    spec: SweepSpec,
    models: list[str] | None = None,
    *,
    dataset: str = "wikitext2",
    ppl_range: tuple[float, float] = (5.0, 50.0),
) -> dict[str, float]:
    """Dense perplexity gate before any prune.

    Runs (or reuses) dense eval on ``dataset`` for each model. Raises if any
    score is outside ``ppl_range``. Returns the dense scores that cleared.
    """
    chosen = list(spec.models if models is None else models)
    for model in chosen:
        if model not in spec.models:
            raise SystemExit(f"Unknown model {model}")
    if dataset not in spec.eval_datasets:
        raise SystemExit(f"Unknown smoke dataset {dataset} for {spec.name}")
    low, high = ppl_range
    if not (low < high):
        raise SystemExit(f"invalid smoke range {ppl_range}")
    spec.result_dir.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    cleared: dict[str, float] = {}
    for model in chosen:
        path = result_path(spec, model)
        results = load_results(path, spec)
        name = short_name(model)
        existing = lookup(results, dataset, "dense", 0.0)
        if existing is None:
            log_path = spec.log_dir / f"{name}_{dataset}_{spec.log_infix}_dense.log"
            found = run_job(family_cmd(spec, model, [dataset], 0.0), log_path)
            existing = found[dataset]
            add_point(results, model, dataset, "dense", 0.0, existing)
            save_results(path, results)
            print(f"{model} {dataset} dense perplexity={existing:.4f}", flush=True)
            _refresh(spec)
        else:
            print(f"skip {name} {dataset} dense perplexity={existing:.4f}", flush=True)
        if not (low <= existing <= high):
            raise SystemExit(
                f"smoke failed for {model}: {dataset} dense perplexity "
                f"{existing:.4f} outside [{low}, {high}]"
            )
        cleared[model] = existing
    print(
        f"smoke cleared for {len(cleared)} model(s) on {dataset} "
        f"in [{low}, {high}]",
        flush=True,
    )
    return cleared


def run_sweep(spec: SweepSpec, models: list[str] | None = None, phase: str = "all") -> None:
    chosen = list(spec.models if models is None else models)
    for model in chosen:
        if model not in spec.models:
            raise SystemExit(f"Unknown model {model}")
    phases = list(spec.eval_datasets) if phase == "all" else [phase]
    for dataset in phases:
        if dataset not in spec.eval_datasets:
            raise SystemExit(f"Unknown eval dataset {dataset} for {spec.name}")
    spec.result_dir.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    for model in chosen:
        path = result_path(spec, model)
        results = load_results(path, spec)
        name = short_name(model)
        for dataset in phases:
            existing = lookup(results, dataset, "dense", 0.0)
            if existing is not None:
                print(f"skip {name} {dataset} dense perplexity={existing:.4f}", flush=True)
                continue
            log_path = spec.log_dir / f"{name}_{dataset}_{spec.log_infix}_dense.log"
            found = run_job(family_cmd(spec, model, [dataset], 0.0), log_path)
            perplexity = found[dataset]
            add_point(results, model, dataset, "dense", 0.0, perplexity)
            save_results(path, results)
            print(f"{model} {dataset} dense perplexity={perplexity:.4f}", flush=True)
            _refresh(spec)
        for sparsity in SPARSITIES:
            token = sparsity_token(sparsity)
            for dataset in phases:
                existing = lookup(results, dataset, "sparsegpt", sparsity)
                if existing is not None:
                    print(f"skip {name} {dataset} sparsity={token} perplexity={existing:.4f}", flush=True)
                    continue
                if checkpoint(model, sparsity, spec.calibration) is None:
                    prune_log = spec.log_dir / f"{name}_{spec.log_infix}_sparsity{token}_prune.log"
                    prune_cmd = [
                        sys.executable,
                        "-m",
                        spec.runner,
                        model,
                        spec.calibration,
                        "--sparsity",
                        token,
                        "--skip-eval",
                        *spec.extra_args,
                        "--save",
                        f"sparse_opt/{name}",
                    ]
                    run_job(prune_cmd, prune_log, expect_perplexity=False)
                    if checkpoint(model, sparsity, spec.calibration) is None:
                        raise RuntimeError(f"prune finished without a checkpoint for {name} {token}")
                log_path = spec.log_dir / f"{name}_{dataset}_{spec.log_infix}_sparsity{token}.log"
                found = None
                for attempt in range(3):
                    try:
                        found = run_job(family_cmd(spec, model, [dataset], sparsity), log_path)
                        break
                    except RuntimeError as exc:
                        if "STALL" not in str(exc) or attempt == 2:
                            raise
                        print(f"retry {name} {dataset} sparsity={token} after stall ({attempt + 1})", flush=True)
                perplexity = found[dataset]
                add_point(results, model, dataset, "sparsegpt", sparsity, perplexity)
                save_results(path, results)
                print(f"{model} {dataset} sparsity={token} perplexity={perplexity:.4f}", flush=True)
                _refresh(spec)
            release_finished_checkpoints(spec)
        _page(spec)
    _page(spec)


def _refresh(spec: SweepSpec) -> None:
    if not spec.plot:
        return
    try:
        from analysis.plots import refresh_family

        refresh_family(spec)
    except Exception as exc:
        print(f"plot refresh failed: {exc}", flush=True)


def _page(spec: SweepSpec) -> None:
    if not spec.plot:
        return
    from analysis.principle import write_family_page

    write_family_page(spec)


def main() -> None:
    from experiments.gemma import GEMMA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(GEMMA.models))
    parser.add_argument(
        "--phase",
        choices=[*GEMMA.eval_datasets, "all"],
        default="all",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    run_sweep(GEMMA, args.models, args.phase)


if __name__ == "__main__":
    main()
