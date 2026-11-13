"""Paired whole-neuron/entrywise comparisons at measured compression budgets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sparsifier.data.gsm8k import digest
from sparsifier.prune.whole_neuron import VERSION


def read_run(path):
    config = json.loads((path/"config.json").read_text())
    report = json.loads((path/"report.json").read_text())
    if not report.get("complete") or report["config_sha256"] != digest(config):
        raise ValueError("incomplete or mismatched comparison run")
    if config.get("method_version", VERSION) != VERSION:
        raise ValueError("whole-neuron method-version mismatch")
    records = [json.loads(line) for line in (path/"answers.jsonl").read_text().splitlines() if line]
    if (len({r["id"] for r in records}) != len(records)
            or any(r["config_sha256"] != digest(config) for r in records)):
        raise ValueError("answer provenance or duplicate-ID mismatch")
    ids = config["evaluation_ids"]
    if set(ids) != {r["id"] for r in records}:
        raise ValueError("comparison answers do not match declared IDs")
    return config, report, {r["id"]: r for r in records}


def compare(paths, *, budget_tolerance=0., seed=0, draws=2000):
    if not np.isfinite(budget_tolerance) or budget_tolerance < 0 or draws < 1:
        raise ValueError("nonnegative budget tolerance and positive bootstrap count required")
    runs = [read_run(path) for path in paths]
    if len(runs) < 2:
        raise ValueError("at least two completed runs required")
    first_config, first_report, baseline = runs[0]
    ids = first_config["evaluation_ids"]
    rng = np.random.default_rng(seed)
    rows = []
    for path, (config, report, answers) in zip(paths, runs):
        for key in ("model", "revision", "manifest_sha256", "max_new_tokens"):
            if config.get(key) != first_config.get(key):
                raise ValueError(f"comparison protocol mismatch: {key}")
        if set(config["evaluation_ids"]) != set(ids) or config.get("thinking", False) != first_config.get("thinking", False):
            raise ValueError("evaluation identity/thinking mismatch")
        delta = np.array([float(answers[i]["correct"])-float(baseline[i]["correct"]) for i in ids])
        samples = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(1)
        fraction = report["removed_weights"]/report["total_parameters"]
        baseline_fraction = first_report["removed_weights"]/first_report["total_parameters"]
        rows.append({"run": str(path), "arm": config["arm"], "accuracy": float(np.mean([answers[i]["correct"] for i in ids])),
            "paired_accuracy_difference": float(delta.mean()), "exploratory_prompt_bootstrap_interval": np.quantile(samples, [.025, .975]).tolist(),
            "whole_model_removed_fraction": fraction,
            "matched_removed_weight_budget": abs(fraction-baseline_fraction) <= budget_tolerance,
            "architectural_removed_slots": report.get("architectural_removed_slots"),
            "eligible_parameters": report["eligible_parameters"],
            "backend_matches_baseline": bool(config.get("backend")) and config.get("backend") == first_config.get("backend"),
            "truncated": sum(answers[i].get("truncated", False) for i in ids),
            "mean_completion_tokens": float(np.mean([answers[i].get("generated_tokens", 0) for i in ids])),
            "representation": "dense_masked; compact bytes/latency require exported-artifact benchmark"})
    return {"baseline": str(paths[0]), "comparisons": rows, "seed": seed, "draws": draws,
            "interpretation": "Complete pipeline comparison, not a causal isolation of pruning structure; no interpolation or retention claim",
            "uncertainty": "paired prompt bootstrap; assumes independent prompts and conditions on calibration; cluster resampling needed for controlled families"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--budget-tolerance", type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, budget_tolerance=args.budget_tolerance), indent=2))


if __name__ == "__main__":
    main()
