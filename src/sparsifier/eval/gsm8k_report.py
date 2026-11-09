"""Summarize completed, paired development arms; never pick a test winner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import atomic_json, read_completed
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION


def paired_accuracy(candidate, baseline, seed=0, draws=2000):
    if set(candidate) != set(baseline):
        raise ValueError("paired comparison requires identical example IDs")
    groups = {}
    for identifier in sorted(candidate):
        a, b = candidate[identifier], baseline[identifier]
        if a["family"] != b["family"] or a["expected_answer"] != b["expected_answer"]:
            raise ValueError("paired family or target mismatch")
        groups.setdefault(a["family"], []).append(float(a["correct"]) - float(b["correct"]))
    keys = sorted(groups)
    if not keys:
        raise ValueError("empty paired comparison")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        samples = []
        for family in rng.choice(keys, size=len(keys), replace=True):
            values = groups[family]
            samples.extend(rng.choice(values, len(values), replace=True))
        means.append(np.mean(samples))
    return {"accuracy_difference": float(np.mean([v for values in groups.values() for v in values])),
            "clustered_ci95": np.quantile(means, [.025, .975]).tolist(),
            "source_groups": len(groups), "problems": len(candidate),
            "interpretation": "exploratory_validation_interval_not_confirmatory"}


def check_protocol(candidate, baseline):
    """Reject comparisons that confound pruning with evaluation changes."""
    for key in ("model", "revision", "capture_sha256", "manifest_sha256",
                "evaluation_split", "max_new_tokens"):
        if key not in candidate or key not in baseline or candidate[key] != baseline[key]:
            raise ValueError(f"paired protocol mismatch: {key}")


def summarize(root):
    records, entries, configs = {}, {}, {}
    for folder in sorted(root.glob("qwen-9b-gsm8k-*-s*-v1")):
        config_file, report_file = folder / "config.json", folder / "report.json"
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text())
        # The glob also matches dense-smoke: it is not a pruning arm.
        if "arm" not in config:
            continue
        configs[folder.name] = config
        entry = {"arm": config["arm"], "alpha": config["alpha"], "seed": config["seed"], "complete": False}
        entries[folder.name] = entry
        if config['arm'] == 'local' and config.get('local_pruning_version') != LOCAL_PRUNING_VERSION:
            entry.update(excluded=True, reason='invalid legacy local normalization; rerun corrected pruning')
            continue
        if not report_file.exists():
            continue
        report = json.loads(report_file.read_text())
        if not report.get("complete"):
            continue
        if report["config_sha256"] != digest(config):
            raise ValueError(f"report/config mismatch: {folder}")
        answers = read_completed(folder / "answers.jsonl", digest(config))
        if set(answers) != set(config["evaluation_ids"]):
            raise ValueError("completed report does not cover its declared evaluation")
        records[folder.name] = answers
        entry.update(complete=True, accuracy=report["behavior"]["accuracy"],
                     answer_nll=report["behavior"]["answer_nll"],
                     truncated=report["behavior"]["truncated"],
                     delta_language_nll=report["language"]["delta_nll"],
                     sparsity=report["local"]["actual_sparsity"])
    baseline = next((name for name, entry in entries.items() if entry["arm"] == "sparsegpt_task"
                     and entry["seed"] == 0 and entry["complete"]), None)
    dense = next((name for name, entry in entries.items() if entry["arm"] == "dense" and entry["complete"]), None)
    for name, answers in records.items():
        if baseline:
            check_protocol(configs[name], configs[baseline])
            entries[name]["paired_vs_task_sparsegpt"] = paired_accuracy(answers, records[baseline])
        if dense:
            check_protocol(configs[name], configs[dense])
            paired_accuracy(answers, records[dense], draws=1)
            correct = [key for key, row in records[dense].items() if row["correct"]]
            entries[name]["dense_correct_retained"] = sum(answers[key]["correct"] for key in correct)
            entries[name]["dense_correct_total"] = len(correct)
    return {"scope": "one_layer_development_validation", "arms": entries,
            "limitations": ["not GSM8K official test", "not multi-layer pruning", "few independent template groups",
                            "output-metric solver is a row-diagonal congruence approximation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/tangent/gsm8k"))
    args = parser.parse_args()
    result = summarize(args.root)
    atomic_json(args.root / "development-comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
