"""Versioned sum-then-square reruns of completed downstream experiments."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

from sparsifier.data.gsm8k import digest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from scripts.submit_gsm8k_downstream_sweep import command

PLAN = "downstream-weightdiag-plan-v1.json"
SMOKE = "qwen-9b-gsm8k-sequential-downstream-weightdiag-smoke-v1"


def build_plan(root):
    plan = copy.deepcopy(json.loads((root / "downstream-retention-sweep-plan-v1.json").read_text()))
    previous = plan["tasks"]
    plan["arm"] = "downstream_weightdiag"
    plan["smoke_name"] = SMOKE
    plan["pullback"]["metric"] = "weight_diagonal"
    plan["pullback"]["approximation"] = "shared_weight_diagonal_then_row_congruence"
    plan["tasks"] = [{"name": SMOKE, "phase": "smoke", "arm": plan["arm"],
                      "eligible_sparsity": .5, "blocks": "15,16", "eval_count": 4, "depends_on": []}]
    plan["comparison_sources"] = {}
    for task in previous:
        source = root / task["name"]
        if task["phase"] != "prune" or not (source / "report.json").exists():
            continue
        report = json.loads((source / "report.json").read_text())
        if not report.get("complete"):
            continue
        config = json.loads((source / "config.json").read_text())
        if report["config_sha256"] != digest(config):
            raise ValueError("historical result provenance mismatch")
        name = task["name"].removesuffix("-v1") + "-weightdiag-v1"
        plan["comparison_sources"][name] = {
            "run": task["name"], "config_sha256": digest(config),
            "report_sha256": hashlib.sha256((source / "report.json").read_bytes()).hexdigest()}
        plan["tasks"].extend([
            dict(task, name=name, arm=plan["arm"], depends_on=[SMOKE]),
            {"name": name + "-original512", "phase": "original_validation", "arm": plan["arm"],
             "depends_on": [name]}])
    if not plan["comparison_sources"]:
        raise ValueError("no completed downstream candidates to rerun")
    plan["eligible_sparsities"] = [t["eligible_sparsity"] for t in plan["tasks"] if t["phase"] == "prune"]
    plan["limitations"] = ["Sum tokens before squaring per sequence and probe; independent sequences/anchors are summed after squaring",
                           "Exact shared-weight geometric diagonal, approximate row-congruence compensation",
                           "Historical backend policy retained; reference-only dispatch remains unverified",
                           "Full 512-problem validation without the newer 128-problem early screen, matching completed 30/50% results"]
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/tangent/gsm8k"
    plan_path = root / PLAN
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else build_plan(root)
    write_immutable(plan_path, plan)
    journal_path = root / "downstream-weightdiag-submissions-v1.json"
    with (root / "downstream-weightdiag-submissions-v1.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            "plan_sha256": digest(plan), "dense_job": 64, "tasks": {}}
        if journal["plan_sha256"] != digest(plan):
            raise ValueError("submission plan drift")
        ids = {}
        for task in plan["tasks"]:
            name = task["name"]
            if name in journal["tasks"]:
                old = journal["tasks"][name]
                if old["state"] != "submitted":
                    raise RuntimeError(f"ambiguous submission: {name}; reconcile with gflow")
                ids[name] = old["job_id"]
                continue
            deps = [ids[parent] for parent in task["depends_on"]]
            if task["phase"] == "original_validation":
                deps.append(64)
            argv = command(repo, task, deps, digest(plan))
            argv = [value.replace(".pixi/envs/default/bin/python", ".pixi/envs/gb10/bin/python") for value in argv]
            if "--screen-128" in argv:
                argv.remove("--screen-128")
            argv += ["--plan", PLAN]
            if not args.submit:
                print(json.dumps({"task": name, "dependencies": deps, "argv": argv}))
                ids[name] = "JOB_FOR_" + name
                continue
            journal["tasks"][name] = {"state": "submitting", "argv": argv}
            atomic_json(journal_path, journal)
            result = subprocess.run(argv, cwd=repo, text=True, capture_output=True)
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if result.returncode or not match:
                journal["tasks"][name].update(stdout=result.stdout, stderr=result.stderr)
                atomic_json(journal_path, journal)
                raise RuntimeError(f"ambiguous submission: {name}")
            ids[name] = int(match.group(1))
            journal["tasks"][name].update(state="submitted", job_id=ids[name], stdout=result.stdout)
            atomic_json(journal_path, journal)
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
