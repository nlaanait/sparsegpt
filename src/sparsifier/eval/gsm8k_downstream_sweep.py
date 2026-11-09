"""Additive, immutable GSM8K downstream-pullback retention sweep and worker.

The existing active sweep plan and its submission journal remain independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k_downstream import ARM, PULLBACK
from sparsifier.eval.gsm8k_sweep import MODULES, SPARSITIES, build_plan as reference_plan


PLAN = "downstream-retention-sweep-plan-v1.json"
SMOKE = "qwen-9b-gsm8k-sequential-downstream-smoke-v1"


def build_plan(calibration, validation):
    reference = reference_plan(calibration, validation)  # Includes source-leakage checks.
    tasks = [{"name": SMOKE, "phase": "smoke", "arm": ARM,
              "eligible_sparsity": .5, "blocks": "15,16", "eval_count": 4, "depends_on": []}]
    for sparsity in SPARSITIES:
        name = f"qwen-9b-gsm8k-mlp-downstream-s{round(100*sparsity)}-seed0-v1"
        tasks.append({"name": name, "phase": "prune", "arm": ARM,
                      "eligible_sparsity": sparsity, "blocks": "all", "eval_count": 32,
                      "depends_on": [SMOKE]})
        tasks.append({"name": name + "-original512", "phase": "original_validation",
                      "arm": ARM, "depends_on": [name]})
    return {key: value for key, value in reference.items() if key not in ("tasks", "limitations")} | {
        "arm": ARM, "pullback": dict(PULLBACK), "tasks": tasks,
        "limitations": ["MLP-only sparsity; measure whole-model removed fraction separately",
                        "32 family-balanced pruning-calibration problems for backward probes; extra compute is recorded",
                        "fit all anchor charts using geometry_fit only; refresh activations/geometry at every site",
                        "next-block residual anchor; final block uses its own post-MLP residual",
                        "diagonal output/token metric and row congruence approximations",
                        "new arm is exploratory; no retention tolerance or final-test configuration selected"]}


def smoke_gate(root: Path, plan: dict):
    source = root / SMOKE
    if not (source / "report.json").exists():
        raise RuntimeError("downstream smoke report missing")
    from sparsifier.eval.gsm8k_original import frozen_stages
    config, report = frozen_stages(source)
    expected = [[block, module] for block in (15, 16) for module in MODULES.split(",")]
    if config["arm"] != ARM or config["pullback"] != plan["pullback"]:
        raise ValueError("downstream smoke method mismatch")
    if config["manifest_sha256"] != plan["calibration_manifest_sha256"]:
        raise ValueError("downstream smoke manifest mismatch")
    if config["alpha"] != plan["alpha"] or config["seed"] != plan["seed"]:
        raise ValueError("downstream smoke hyperparameter mismatch")
    if config["max_new_tokens"] != plan["max_new_tokens"] or [s["site"] for s in report["stages"]] != expected:
        raise ValueError("downstream smoke scope/generation mismatch")
    if report["behavior"]["count"] != 4 or not 0 < report["whole_model_removed_fraction"] < 1:
        raise ValueError("downstream smoke behavior/sparsity missing")
    for stage in report["stages"]:
        diagnostic = stage["diagnostics"]
        if diagnostic.get("geometry_trace", 0) <= 0 or diagnostic.get("sample_count") != 8192:
            raise ValueError("downstream smoke calibration invalid")
        if diagnostic.get("num_probes") != plan["pullback"]["num_probes"]:
            raise ValueError("downstream smoke probe budget mismatch")
        if len(diagnostic.get("probe_ids", [])) != plan["pullback"]["probe_count"]:
            raise ValueError("downstream smoke calibration problem count mismatch")
    return config, report


def worker(root: Path, task_name: str, expected_hash: str, plan_name: str = PLAN):
    plan = json.loads((root / plan_name).read_text())
    if digest(plan) != expected_hash:
        raise ValueError("downstream sweep plan drift")
    for filename, key in (("controlled-v1.json", "calibration_manifest_sha256"),
                          ("original-validation-v1.json", "validation_manifest_sha256")):
        if load_manifest(root / "data" / filename)["manifest_sha256"] != plan[key]:
            raise ValueError("downstream sweep manifest drift")
    matches = [task for task in plan["tasks"] if task["name"] == task_name]
    if len(matches) != 1:
        raise ValueError("task not uniquely identified")
    task = matches[0]
    output = root / task["name"]
    if task["phase"] == "original_validation":
        from sparsifier.eval.gsm8k_original import frozen_stages
        config, report = frozen_stages(root / task["depends_on"][0])
        if config["arm"] != ARM or config["blocks"] != "all" or config["modules"].split(",") != plan["modules"]:
            raise ValueError("downstream source candidate scope mismatch")
        module = "sparsifier.eval.gsm8k_original"
        argv = ["--manifest", str(root / "data/original-validation-v1.json"),
                "--source", str(root / task["depends_on"][0]), "--output", str(output),
                "--max-new-tokens", str(plan["max_new_tokens"])]
    else:
        if task["phase"] == "prune":
            smoke_gate(root, plan)
        module = "sparsifier.eval.gsm8k_downstream"
        argv = ["--manifest", str(root / "data/controlled-v1.json"),
                "--pilot-report", str(root / "qwen-9b-gsm8k-dense-pilot-v2/report.json"),
                "--c4-capture", str(root / "capture-v1"), "--output", str(output),
                "--blocks", task["blocks"], "--modules", ",".join(plan["modules"]),
                "--sparsity", str(task["eligible_sparsity"]), "--alpha", str(plan["alpha"]),
                "--seed", str(plan["seed"]), "--eval-count", str(task["eval_count"]),
                "--max-new-tokens", str(plan["max_new_tokens"]),
                "--num-probes", str(plan["pullback"]["num_probes"]),
                "--probe-count", str(plan["pullback"]["probe_count"])]
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/tangent/gsm8k"))
    parser.add_argument("--task")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--plan", default=PLAN)
    args = parser.parse_args()
    if args.task:
        if not args.plan_sha256:
            parser.error("worker requires --plan-sha256")
        worker(args.root, args.task, args.plan_sha256, args.plan)
    else:
        plan = build_plan(load_manifest(args.root / "data/controlled-v1.json"),
                          load_manifest(args.root / "data/original-validation-v1.json"))
        write_immutable(args.root / PLAN, plan)
        print(json.dumps({"tasks": len(plan["tasks"]), "plan_sha256": digest(plan)}, indent=2))


if __name__ == "__main__":
    main()
