"""Run a matched SparseGPT versus output-tangent pruning comparison.

This driver runs the two full-model variants sequentially in one process group
so they can be submitted as one scheduled GPU job. It writes separate logs and
a machine-readable report. The report intentionally records measurements; it
does not call a difference an improvement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.protocol import PPL_RE, ROOT


def _dataset_perplexities(log: str) -> dict[str, float]:
    values: dict[str, float] = {}
    dataset = None
    for line in log.splitlines():
        if line.startswith("Dataset:"):
            dataset = line.split(":", 1)[1].strip()
        match = PPL_RE.search(line)
        if match and dataset is not None:
            values[dataset] = float(match.group(1))
            dataset = None
    return values


def _command(args, arm: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sparsifier",
        args.model,
        args.calibration,
        "--seed",
        str(args.seed),
        "--nsamples",
        str(args.nsamples),
        "--percdamp",
        str(args.percdamp),
        "--sparsity",
        str(args.sparsity),
        "--minlayer",
        str(args.minlayer),
        "--maxlayer",
        str(args.maxlayer),
        "--blocksize",
        str(args.blocksize),
        "--forward-batch",
        str(args.forward_batch),
    ]
    for dataset in args.eval_dataset:
        command.extend(("--eval-datasets", dataset))
    if args.true_sequential:
        command.append("--true-sequential")
    if args.skip_eval:
        command.append("--skip-eval")
    if args.pytorch_attention:
        command.append("--pytorch-attention")
    if arm == "baseline":
        command.extend(("--tangent-alpha", "0"))
    else:
        command.extend(
            (
                "--tangent-alpha",
                str(args.tangent_alpha),
                "--tangent-charts",
                str(args.tangent_charts),
                "--tangent-rank",
                str(args.tangent_rank),
                "--tangent-samples",
                str(args.tangent_samples),
            )
        )
    if args.save_dir:
        command.extend(("--save", str(args.save_dir / arm)))
    return command


def run_arm(args, arm: str) -> dict:
    command = _command(args, arm)
    log_path = args.output_dir / f"{arm}.log"
    print(f"[{arm}] {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log = log_path.read_text(encoding="utf-8", errors="replace")
    record = {
        "command": command,
        "returncode": result.returncode,
        "log": str(log_path),
        "perplexity": _dataset_perplexities(log),
    }
    print(f"[{arm}] exit={result.returncode} log={log_path}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", choices=("wikitext2", "ptb", "c4"), default="c4")
    parser.add_argument(
        "--eval-dataset",
        action="append",
        choices=("wikitext2", "ptb", "c4"),
        default=[],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--forward-batch", type=int, default=0)
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--pytorch-attention",
        action="store_true",
        help=(
            "Force Qwen PyTorch attention/causal-conv fallbacks. Use while a "
            "causal-conv1d wheel build is still running."
        ),
    )
    parser.add_argument("--tangent-alpha", type=float, default=0.5)
    parser.add_argument("--tangent-charts", type=int, default=8)
    parser.add_argument("--tangent-rank", type=int, default=2)
    parser.add_argument("--tangent-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional root for saving both pruned checkpoints; may require substantial disk space.",
    )
    args = parser.parse_args()
    if args.tangent_alpha <= 0 or args.tangent_alpha > 1:
        parser.error("tangent alpha must lie in (0, 1]")
    if args.tangent_charts < 1 or args.tangent_rank < 1 or args.tangent_samples < 3:
        parser.error("tangent charts/rank must be positive and samples >= 3")
    if not 0 < args.sparsity < 1:
        parser.error("sparsity must lie in (0, 1)")
    if not args.eval_dataset and not args.skip_eval:
        args.eval_dataset = ["wikitext2"]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        args.save_dir = args.save_dir.resolve()
        args.save_dir.mkdir(parents=True, exist_ok=True)

    variants = ("baseline", "tangent")
    results = {}
    for arm in variants:
        results[arm] = run_arm(args, arm)
        if results[arm]["returncode"] != 0:
            break

    report = {
        "design": {
            "comparison": "matched sparsity, calibration, seed, layer selection, and evaluation",
            "baseline": "SparseGPT ambient mask and ambient compensation",
            "tangent": (
                "output tangent weighted mask scores with ambient SparseGPT compensation; "
                "a heuristic rather than projected OBS"
            ),
            "settings": {
                key: getattr(args, key)
                for key in (
                    "model",
                    "calibration",
                    "eval_dataset",
                    "seed",
                    "nsamples",
                    "percdamp",
                    "sparsity",
                    "minlayer",
                    "maxlayer",
                    "blocksize",
                    "forward_batch",
                    "true_sequential",
                    "skip_eval",
                    "pytorch_attention",
                    "tangent_alpha",
                    "tangent_charts",
                    "tangent_rank",
                    "tangent_samples",
                )
            },
        },
        "variants": results,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    return 0 if all(record["returncode"] == 0 for record in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
