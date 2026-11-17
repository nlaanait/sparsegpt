"""Version and replace the queued GSM8K program with shared batch-eight runs.

Submit replacements before canceling queued predecessors. Keep started jobs
and their artifacts intact. Previously canceled local pilots stay excluded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from scripts.submit_gsm8k_decision_pilot import MANIFEST, build_plan, command

PLAN = "decision64-light128-batch8-rollout-plan-v1.json"
JOURNAL = "decision64-light128-batch8-rollout-submissions-v1.json"
OLD_PILOTS = {
    148: "qwen-9b-gsm8k-pilot-entrywise-downstream-weight-diagonal-b15-s30-seed0-v1",
    149: "qwen-9b-gsm8k-pilot-entrywise-downstream-weight-diagonal-b15-s30-seed0-v1-light128",
    150: "qwen-9b-gsm8k-pilot-whole-neuron-ambient-b15-s30-seed0-v1",
    151: "qwen-9b-gsm8k-pilot-whole-neuron-ambient-b15-s30-seed0-v1-light128",
    154: "qwen-9b-gsm8k-pilot-whole-neuron-downstream-weight-diagonal-b15-s30-seed0-v1",
    155: "qwen-9b-gsm8k-pilot-whole-neuron-downstream-weight-diagonal-b15-s30-seed0-v1-light128",
}
OLD_EVALUATIONS = {
    161: "qwen-9b-gsm8k-mlp-downstream-s30-seed0-weightdiag-v1",
    162: "qwen-9b-gsm8k-mlp-downstream-s50-seed0-weightdiag-v1",
    163: "qwen-9b-gsm8k-mlp-downstream-s70-seed0-weightdiag-v1",
}


def rollout_plan(manifest):
    plan = build_plan(manifest)
    plan["tasks"] = [t for t in plan["tasks"] if "-local-" not in t["name"]]
    replacements = {str(job): {"old_name": name, "new_name": name.replace("seed0-v1", "seed0-batch8-v1")}
                    for job, name in OLD_PILOTS.items()}
    for job, source in OLD_EVALUATIONS.items():
        name = source+"-decision128-seed0-batch8-v1"
        plan["tasks"].append({"name": name, "phase": "evaluate", "source": source,
                              "depends_on": [plan["dense"]], "reuse_frozen_candidate": True})
        replacements[str(job)] = {"old_name": source+"-decision128-seed0-v1", "new_name": name}
    plan.update(purpose="batch8_rollout_of_current_pilot_and_queued_evaluations",
                replacements=replacements, excluded="previously_canceled_local_pilots",
                historical="existing_results_and_started_jobs_keep_their_original_configuration")
    return plan


def inspect_old(job, expected_name):
    result = subprocess.run(["gjob", "show", str(job)], text=True, capture_output=True, check=True)
    command_line = next((line for line in result.stdout.splitlines() if "Command=" in line), "")
    if expected_name not in command_line:
        raise RuntimeError(f"job {job} identity changed; refuse to cancel")
    match = re.search(r"State=([^\n]+)", result.stdout)
    if not match:
        raise RuntimeError(f"cannot determine state of job {job}")
    return match.group(1).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo/"results/tangent/gsm8k"
    plan = rollout_plan(load_manifest(root/MANIFEST))
    write_immutable(root/PLAN, plan)
    if not args.submit:
        print(json.dumps(plan, indent=2))
        return
    with (root/(JOURNAL+".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/JOURNAL
        journal = json.loads(path.read_text()) if path.exists() else {"plan_sha256": digest(plan), "tasks": {}, "legacy": {}}
        if journal["plan_sha256"] != digest(plan):
            raise ValueError("rollout plan drift")
        ids = {}
        for task in plan["tasks"]:
            name = task["name"]
            old = journal["tasks"].get(name)
            if old:
                if old["state"] != "submitted":
                    raise RuntimeError("ambiguous submission; reconcile "+name)
                ids[name] = old["job_id"]
                continue
            argv = command(repo, task, [ids[p] for p in task["depends_on"]], digest(plan), plan_name=PLAN)
            journal["tasks"][name] = {"state": "submitting", "argv": argv}
            atomic_json(path, journal)
            result = subprocess.run(argv, cwd=repo, text=True, capture_output=True)
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if result.returncode or not match:
                journal["tasks"][name].update(stdout=result.stdout, stderr=result.stderr)
                atomic_json(path, journal)
                raise RuntimeError("ambiguous submission "+name)
            ids[name] = int(match.group(1))
            journal["tasks"][name].update(state="submitted", job_id=ids[name])
            atomic_json(path, journal)
            print(result.stdout.strip(), flush=True)
        # Every replacement exists and has a recorded ID before cancellation.
        for job, replacement in plan["replacements"].items():
            state = inspect_old(job, replacement["old_name"])
            if state == "Queued (PD)":
                # Recheck immediately before mutation to avoid canceling a started job.
                if inspect_old(job, replacement["old_name"]) == "Queued (PD)":
                    subprocess.run(["gcancel", job], check=True)
                    state = "Canceled after replacement submitted"
            journal["legacy"][job] = {"state": state, "replacement_job": ids[replacement["new_name"]]}
            atomic_json(path, journal)
            print(json.dumps({"old_job": job, **journal["legacy"][job]}), flush=True)


if __name__ == "__main__":
    main()
