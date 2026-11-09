"""Submit the additive downstream tangent arm once, with smoke/evaluation gates."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_downstream_sweep import PLAN


def command(repo, task, dependencies, plan_hash):
    root = repo / "results/tangent/gsm8k"
    result = ["gbatch", "--gpus", "1", "--gpu-memory", "64G", "--memory", "80G",
              "--name", task["name"], "--project", "qwen-gsm8k", "--time", "12:00:00", "--auto-close"]
    if dependencies:
        result += ["--depends-on-all", ",".join(map(str, dependencies))]
    return result + ["--", "bash", str(repo / "scripts/run_gsm8k_guarded.sh"), task["name"],
                     str(root / task["name"] / "memory.jsonl"),
                     str(repo / ".pixi/envs/default/bin/python"), "-u", "-m",
                     "sparsifier.eval.gsm8k_downstream_sweep", "--root", str(root),
                     "--task", task["name"], "--plan-sha256", plan_hash]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-job", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/tangent/gsm8k"
    plan = json.loads((root / PLAN).read_text())
    plan_hash = digest(plan)
    journal_path = root / "downstream-retention-submissions-v1.json"
    with (root / "downstream-retention-submissions-v1.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            "plan_sha256": plan_hash, "dense_job": args.dense_job, "tasks": {}}
        if journal["plan_sha256"] != plan_hash or journal["dense_job"] != args.dense_job:
            raise ValueError("submission plan or dense dependency drift")
        ids = {}
        for task in plan["tasks"]:
            name = task["name"]
            if name in journal["tasks"]:
                old = journal["tasks"][name]
                if old["state"] != "submitted":
                    raise RuntimeError(f"ambiguous submission for {name}; reconcile with gflow")
                ids[name] = old["job_id"]
                continue
            deps = [ids[parent] for parent in task["depends_on"]]
            if task["phase"] == "original_validation":
                deps.append(args.dense_job)
            argv = command(repo, task, deps, plan_hash)
            if not args.submit:
                print(json.dumps({"task": name, "phase": task["phase"], "dependencies": deps, "argv": argv}))
                ids[name] = "JOB_ID_FOR_" + name
                continue
            journal["tasks"][name] = {"state": "submitting", "argv": argv}
            atomic_json(journal_path, journal)
            result = subprocess.run(argv, cwd=repo, text=True, capture_output=True)
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if result.returncode or not match:
                journal["tasks"][name].update(stdout=result.stdout, stderr=result.stderr)
                atomic_json(journal_path, journal)
                raise RuntimeError(f"submission uncertain for {name}; inspect journal")
            ids[name] = int(match.group(1))
            journal["tasks"][name].update(state="submitted", job_id=ids[name], stdout=result.stdout)
            atomic_json(journal_path, journal)
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
