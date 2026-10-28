"""Collect sum_nll_nats / token-byte-char-word counts per sparsity.

Writes one JSON per model and eval dataset under
``figures/<family>/likelihood/<short>_<dataset>.json``, with sparsity
strings as keys (``0``, ``0.1``, …).

Prune in-process and never write ``sparse_opt/`` checkpoints. Each sparsity
reloads the dense weights, SparseGPT-prunes them, evaluates, then drops the
model. Resume-friendly: existing sparsity keys are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch

from analysis.sweep import (
    parse_sparsities,
    result_path as sparsity_result_path,
    short_name,
    sparsity_token,
)
from experiments.gemma import GEMMA, GEMMA_IT_WIKITEXT2_CAL
from experiments.llama import LLAMA3, LLAMA3_WIKITEXT2_CAL
from experiments.olmo import OLMO2, OLMO2_WIKITEXT2_CAL
from experiments.qwen import (
    ROOT,
    QWEN_35_WIKITEXT2_CAL,
    QWEN_36_38,
    QWEN_WIKITEXT2_CAL,
    SweepSpec,
)
from qwen_figure1_family import MODELS as QWEN35_MODELS
from qwen_ptb import _prune_args
from sparsegpt.datautils import get_loaders, get_tokenizer
from sparsegpt.likelihood_metrics import (
    has_sparsity,
    load_metrics_file,
    save_metrics_file,
    upsert_sparsity,
)
from sparsegpt.modelutils import DEV

# Qwen3.5 C4-calibrated family (WikiText2 + PTB), historically one JSON each.
QWEN35_C4 = SweepSpec(
    name="qwen3.5",
    models=tuple(QWEN35_MODELS),
    runner="sparsegpt.qwen",
    extra_args=(),
    result_dir=ROOT / "figures",
    log_dir=ROOT / "figures" / "qwen3.5_likelihood_logs",
    log_infix="pytorch",
    attention="atlas-gdn",
    holdout_suffix="27B",
    plot=False,
    calibration="c4",
    eval_datasets=("wikitext2", "ptb"),
)

FAMILIES: dict[str, SweepSpec] = {
    "olmo2": OLMO2,
    "olmo2_wikitext2_cal": OLMO2_WIKITEXT2_CAL,
    "llama3": LLAMA3,
    "llama3_wikitext2_cal": LLAMA3_WIKITEXT2_CAL,
    "gemma": GEMMA,
    "gemma_it_wikitext2_cal": GEMMA_IT_WIKITEXT2_CAL,
    "qwen": QWEN_36_38,
    "qwen_wikitext2_cal": QWEN_WIKITEXT2_CAL,
    "qwen3.5": QWEN35_C4,
    "qwen3.5_wikitext2_cal": QWEN_35_WIKITEXT2_CAL,
}

# module, get_*, *_eval, *_sequential
RUNNERS = {
    "sparsegpt.olmo": ("sparsegpt.olmo", "get_olmo", "olmo_eval", "olmo_sequential"),
    "sparsegpt.llama": ("sparsegpt.llama", "get_llama", "llama_eval", "llama_sequential"),
    "sparsegpt.gemma": ("sparsegpt.gemma", "get_gemma", "gemma_eval", "gemma_sequential"),
    "sparsegpt.qwen": ("sparsegpt.qwen", "get_qwen", "qwen_eval", "qwen_sequential"),
}

PPL_RTOL = 1e-3
PPL_ATOL = 5e-3


def likelihood_dir(spec: SweepSpec) -> Path:
    if spec.name == "qwen3.5":
        return ROOT / "figures" / "qwen3.5" / "likelihood"
    return spec.result_dir / "likelihood"


def metrics_path(spec: SweepSpec, model: str, dataset: str) -> Path:
    return likelihood_dir(spec) / f"{short_name(model)}_{dataset}.json"


def expected_perplexity(spec: SweepSpec, model: str, dataset: str, sparsity: float) -> float | None:
    if spec.name == "qwen3.5":
        if dataset == "wikitext2":
            path = ROOT / "figures" / "qwen3.5_family_sparsity_wikitext2.json"
        elif dataset == "ptb":
            path = ROOT / "figures" / "qwen3.5_family_sparsity_ptb.json"
        else:
            return None
    else:
        path = sparsity_result_path(spec, model)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    method = "dense" if sparsity == 0.0 else "sparsegpt"
    for point in data.get("points", []):
        if point.get("model", model) != model:
            continue
        if "dataset" in point and point["dataset"] != dataset:
            continue
        if point.get("method") != method:
            continue
        if abs(float(point["sparsity"]) - sparsity) > 1e-9:
            continue
        return float(point["perplexity"])
    return None


def _load_runner(spec: SweepSpec):
    module_name, get_name, eval_name, seq_name = RUNNERS[spec.runner]
    module = __import__(module_name, fromlist=[get_name, eval_name, seq_name])
    return module, getattr(module, get_name), getattr(module, eval_name), getattr(module, seq_name)


def disk_report() -> None:
    usage = shutil.disk_usage(ROOT)
    hub = Path.home() / ".cache" / "huggingface"
    hub_gb = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file()) / 1e9 if hub.exists() else 0.0
    print(
        f"disk free={usage.free / 1e9:.1f}GiB used={usage.used / 1e9:.1f}GiB hf_cache≈{hub_gb:.1f}GiB",
        flush=True,
    )
    if usage.free < 200 * 1e9:
        print("WARNING: less than 200GiB free; large base checkpoints may not fit", flush=True)


def pending_datasets(spec: SweepSpec, model: str, sparsity: float, datasets: list[str]) -> list[str]:
    pending = []
    for dataset in datasets:
        data = load_metrics_file(
            metrics_path(spec, model, dataset),
            model=model,
            dataset=dataset,
            calibration=spec.calibration,
            attention=spec.attention,
            seqlen=2048,
        )
        if not has_sparsity(data, sparsity):
            pending.append(dataset)
    return pending


def check_perplexity(
    spec: SweepSpec,
    model: str,
    dataset: str,
    sparsity: float,
    metrics: dict,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    expected = expected_perplexity(spec, model, dataset, sparsity)
    if expected is None or not math.isfinite(expected):
        return
    got = float(metrics["perplexity"])
    if not math.isclose(got, expected, rel_tol=PPL_RTOL, abs_tol=PPL_ATOL):
        # Advisory only: small drifts vs the saved sparsity JSON are common
        # (esp. PTB / Atlas kernels) and should not abort a resume-friendly run.
        print(
            f"perplexity mismatch (advisory) for {model} {dataset} "
            f"sparsity={sparsity}: got {got:.6f} expected {expected:.6f}",
            flush=True,
        )
        return
    print(f"perplexity match {got:.6f} ≈ {expected:.6f}", flush=True)


def eval_one_sparsity(
    spec: SweepSpec,
    model: str,
    sparsity: float,
    datasets: list[str],
    *,
    check_existing: bool,
) -> None:
    module, get_model, evaluate, sequential = _load_runner(spec)
    tokenizer = get_tokenizer(model)
    print(f"loading dense {model}", flush=True)
    torch_model = get_model(model)
    torch_model.eval()
    try:
        if sparsity > 0:
            module.args = _prune_args(model, sparsity, dataset=spec.calibration)
            cal_loader, _ = get_loaders(
                spec.calibration,
                nsamples=module.args.nsamples,
                seed=module.args.seed,
                model=model,
                seqlen=torch_model.seqlen,
            )
            print(
                f"pruning in-memory sparsity={sparsity_token(sparsity)} "
                f"cal={spec.calibration} (no checkpoint)",
                flush=True,
            )
            sequential(torch_model, cal_loader, DEV)
            del cal_loader
            torch.cuda.empty_cache()

        for dataset in datasets:
            path = metrics_path(spec, model, dataset)
            data = load_metrics_file(
                path,
                model=model,
                dataset=dataset,
                calibration=spec.calibration,
                attention=spec.attention,
                seqlen=int(torch_model.seqlen),
            )
            if has_sparsity(data, sparsity):
                print(
                    f"skip {short_name(model)} {dataset} "
                    f"sparsity={sparsity_token(sparsity) if sparsity else '0'}",
                    flush=True,
                )
                continue
            _loader, testloader = get_loaders(
                dataset,
                seed=0,
                model=model,
                seqlen=torch_model.seqlen,
            )
            print("Dataset:", dataset, flush=True)
            metrics: dict = {}
            evaluate(
                torch_model,
                testloader,
                DEV,
                dataset,
                False,
                tokenizer=tokenizer,
                metrics_out=metrics,
            )
            check_perplexity(
                spec, model, dataset, sparsity, metrics, enabled=check_existing
            )
            upsert_sparsity(data, sparsity, metrics)
            save_metrics_file(path, data)
            print(
                f"wrote {path} key={sparsity_token(sparsity) if sparsity else '0'}",
                flush=True,
            )
    finally:
        del torch_model
        torch.cuda.empty_cache()


def run_model(
    spec: SweepSpec,
    model: str,
    sparsities: list[float],
    datasets: list[str],
    *,
    check_existing: bool,
) -> None:
    if model not in spec.models:
        raise SystemExit(f"{model} not in family {spec.name}")
    likelihood_dir(spec).mkdir(parents=True, exist_ok=True)
    disk_report()
    levels = [0.0, *sparsities]
    for sparsity in levels:
        pending = pending_datasets(spec, model, sparsity, datasets)
        if not pending:
            print(
                f"skip all datasets for {short_name(model)} "
                f"sparsity={sparsity_token(sparsity) if sparsity else '0'}",
                flush=True,
            )
            continue
        print(
            f"=== {model} sparsity={sparsity_token(sparsity) if sparsity else '0'} "
            f"datasets={pending} ===",
            flush=True,
        )
        eval_one_sparsity(
            spec,
            model,
            sparsity,
            pending,
            check_existing=check_existing,
        )
        disk_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Eval corpora. Default: the family's eval_datasets.",
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=None,
        help="Sparse levels in (0,1). Dense 0 is always included. Default 0.1..0.8.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Only collect sparsity 0 (no prune).",
    )
    parser.add_argument(
        "--no-check-existing",
        action="store_true",
        help="Do not compare against figures/*_sparsity.json perplexities.",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    spec = FAMILIES[args.family]
    models = list(spec.models if args.models is None else args.models)
    datasets = list(spec.eval_datasets if args.datasets is None else args.datasets)
    for dataset in datasets:
        if dataset not in ("wikitext2", "ptb", "c4"):
            raise SystemExit(f"unknown dataset {dataset}")
    sparsities = [] if args.dense_only else parse_sparsities(args.sparsities)
    for model in models:
        run_model(
            spec,
            model,
            sparsities,
            datasets,
            check_existing=not args.no_check_existing,
        )


if __name__ == "__main__":
    main()
