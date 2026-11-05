"""Sweep sparsity on one model: unstructured SparseGPT vs output-tangent.

For each sparsity, runs the matched pair (ambient baseline, then tangent mask)
and records WikiText2/PTB likelihood (nats) and perplexity. Resumes from the
JSON report so a requeue does not redo finished points. Dense (s=0) is one
baseline eval with tangent disabled.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from analysis.sweep import SPARSITIES, sparsity_token
from experiments.protocol import PPL_RE, ROOT

NATS_RE = re.compile(
    r"Likelihood nats:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?|inf|nan)",
    re.I,
)
TOKENS_RE = re.compile(
    r"Tokens:\s*([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?\d+)?)",
    re.I,
)


def _parse_eval(log: str) -> dict[str, dict[str, float]]:
    """Per dataset: perplexity, nats, tokens (when the log printed them)."""
    out: dict[str, dict[str, float]] = {}
    dataset = None
    pending: dict[str, float] = {}
    for line in log.splitlines():
        if line.startswith("Dataset:"):
            if dataset is not None and pending:
                out[dataset] = pending
            dataset = line.split(":", 1)[1].strip()
            pending = {}
            continue
        if dataset is None:
            continue
        if (m := PPL_RE.search(line)) is not None:
            pending["perplexity"] = float(m.group(1))
        if (m := NATS_RE.search(line)) is not None:
            pending["likelihood_nats"] = float(m.group(1))
        if (m := TOKENS_RE.search(line)) is not None:
            pending["tokens"] = float(m.group(1))
            out[dataset] = pending
            dataset = None
            pending = {}
    if dataset is not None and pending:
        out[dataset] = pending
    return out


def _command(args, *, sparsity: float, arm: str) -> list[str]:
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
        sparsity_token(sparsity) if sparsity > 0 else "0",
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
    if args.pytorch_attention:
        command.append("--pytorch-attention")
    if sparsity <= 0 or arm == "baseline":
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
    if args.save_dir and sparsity > 0:
        command.extend(
            ("--save", str(args.save_dir / arm / f"sp{sparsity_token(sparsity)}"))
        )
    return command


def _load_report(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "design": {},
        "points": [],
    }


def _point_done(report: dict, sparsity: float, arm: str) -> bool:
    for point in report.get("points", []):
        if abs(float(point.get("sparsity", -1)) - sparsity) > 1e-9:
            continue
        variant = point.get("variants", {}).get(arm)
        if variant and variant.get("returncode") == 0:
            return True
    return False


def _upsert_point(report: dict, sparsity: float, arm: str, record: dict) -> None:
    for point in report.setdefault("points", []):
        if abs(float(point.get("sparsity", -1)) - sparsity) <= 1e-9:
            point.setdefault("variants", {})[arm] = record
            return
    report["points"].append(
        {"sparsity": sparsity, "variants": {arm: record}}
    )
    report["points"].sort(key=lambda p: float(p["sparsity"]))


def run_arm(args, sparsity: float, arm: str, log_path: Path) -> dict:
    command = _command(args, sparsity=sparsity, arm=arm)
    print(f"[s={sparsity_token(sparsity)} {arm}] {' '.join(command)}", flush=True)
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
        "eval": _parse_eval(log),
    }
    print(
        f"[s={sparsity_token(sparsity)} {arm}] exit={result.returncode} "
        f"eval={record['eval']}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Mid-size default is OLMo-2-7B-Instruct (Qwen3.5 FLA OOMs on GH200).",
    )
    parser.add_argument("--calibration", choices=("wikitext2", "ptb", "c4"), default="c4")
    parser.add_argument(
        "--eval-dataset",
        action="append",
        choices=("wikitext2", "ptb", "c4"),
        default=[],
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=None,
        help="Sparsity grid (default: analysis.sweep.SPARSITIES = 0.1..0.8).",
    )
    parser.add_argument("--include-dense", action="store_true", default=True)
    parser.add_argument("--no-include-dense", action="store_false", dest="include_dense")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--forward-batch", type=int, default=0)
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--pytorch-attention", action="store_true")
    parser.add_argument("--tangent-alpha", type=float, default=0.5)
    parser.add_argument("--tangent-charts", type=int, default=8)
    parser.add_argument("--tangent-rank", type=int, default=2)
    parser.add_argument("--tangent-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.tangent_alpha <= 0 or args.tangent_alpha > 1:
        parser.error("tangent alpha must lie in (0, 1]")
    sparsities = list(args.sparsities) if args.sparsities is not None else list(SPARSITIES)
    for s in sparsities:
        if not 0 < s < 1:
            parser.error(f"sparsity must lie in (0, 1); got {s}")
    if not args.eval_dataset:
        args.eval_dataset = ["wikitext2"]

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        args.save_dir = args.save_dir.resolve()
        args.save_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "sweep_report.json"
    report = _load_report(report_path)
    report["design"] = {
        "comparison": (
            "one mid-size model; unstructured SparseGPT vs output-tangent mask "
            "across a sparsity sweep; likelihood (nats) and perplexity on eval sets"
        ),
        "baseline": "SparseGPT ambient mask and ambient compensation",
        "tangent": (
            "output tangent weighted mask scores with ambient SparseGPT compensation"
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
                "minlayer",
                "maxlayer",
                "blocksize",
                "forward_batch",
                "true_sequential",
                "pytorch_attention",
                "tangent_alpha",
                "tangent_charts",
                "tangent_rank",
                "tangent_samples",
                "include_dense",
            )
        },
        "sparsities": sparsities,
    }

    grid: list[tuple[float, str]] = []
    if args.include_dense:
        grid.append((0.0, "baseline"))
    for sparsity in sparsities:
        grid.append((sparsity, "baseline"))
        grid.append((sparsity, "tangent"))

    failed = False
    for sparsity, arm in grid:
        if _point_done(report, sparsity, arm):
            print(
                f"[s={sparsity_token(sparsity) if sparsity else 0} {arm}] skip (done)",
                flush=True,
            )
            continue
        log_path = (
            args.output_dir
            / f"s{sparsity_token(sparsity) if sparsity else '0.0'}_{arm}.log"
        )
        record = run_arm(args, sparsity, arm, log_path)
        _upsert_point(report, sparsity, arm, record)
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if record["returncode"] != 0:
            failed = True
            break

    print(f"Report: {report_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
