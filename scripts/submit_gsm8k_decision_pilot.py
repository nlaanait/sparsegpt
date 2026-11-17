"""Freeze shared seeded screens and submit a bounded block-15 six-arm pilot."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_decision import prepare_manifest
from sparsifier.eval.gsm8k_batching import policy
from sparsifier.data.gsm8k_calibration import CALIBRATION, DECISION, load_calibration, validate_calibration, regime

MANIFEST = "data/decision64-light128-seed0-v1.json"
PLAN = "decision64-light128-pilot-repcal-batch32-plan-v1.json"


def build_plan(manifest):
    """Historical controlled-diagnostic plan; retained for legacy migration tools."""
    dense = "qwen-9b-gsm8k-decision-dense-seed0-batch8-v1"
    tasks = [{"name":dense,"phase":"dense","depends_on":[]}]
    for structure in ("entrywise","whole_neuron"):
        for metric in ("ambient","local","downstream_weight_diagonal"):
            name = f"qwen-9b-gsm8k-pilot-{structure.replace('_','-')}-{metric.replace('_','-')}-b15-s30-seed0-batch8-v1"
            task = {"name":name,"phase":"prune","structure":structure,"metric":metric,
                    "block":15,"sparsity":.3,"depends_on":[dense]}
            tasks += [task,{"name":name+"-light128","phase":"evaluate","source":name,"depends_on":[name,dense]}]
    return {"purpose":"single_block_engineering_and_metric_pilot_not_retention_frontier",
        "manifest":MANIFEST,"manifest_sha256":manifest["manifest_sha256"],"dense":dense,
        "shuffle_seed":0,"calibration_seed":0,"torch_seed":0,"screen":64,"light":128,
        "max_new_tokens":2048,"max_drop_pp":15,"block":15,"eligible_sparsity":.3,
        "batching":policy(size=8),"tasks":tasks,"next_decision":"Review 64/128 results and solver/backend checks before widening blocks or sparsity"}


def build_representative_plan(manifest, calibration):
    validate_calibration(calibration)
    if manifest["calibration_manifest_sha256"] != calibration["manifest_sha256"]:
        raise ValueError("representative pilot calibration binding mismatch")
    plan = build_plan(manifest)
    names = {t["name"]: t["name"].replace("batch8-v1", "batch32-repcal-v1") for t in plan["tasks"]}
    for task in plan["tasks"]:
        task["name"] = names[task["name"]]
        task["depends_on"] = [names[n] for n in task["depends_on"]]
        if "source" in task:
            task["source"] = names[task["source"]]
    plan.update(manifest=DECISION, calibration_manifest=CALIBRATION,
                batching=policy(),
                calibration_manifest_sha256=calibration["manifest_sha256"], calibration_regime=regime(calibration),
                dense=names[plan["dense"]], probe_count=calibration["backward_problem_count"], num_probes=8,
                purpose="representative_original_calibration_block_pilot_not_controlled_template_transfer")
    return plan


def command(repo, task, dependencies, plan_hash, *, plan_name=PLAN):
    root = repo/"results/tangent/gsm8k"
    argv = ["gbatch","--gpus","1","--gpu-memory","64G","--memory","80G",
        "--name",task["name"],"--project","qwen-gsm8k","--time","24:00:00","--auto-close"]
    if task["phase"] == "dense":
        argv += ["--priority","110"]
    if dependencies:
        argv += ["--depends-on-all", ",".join(map(str,dependencies))]
    return argv+["--","env","GSM8K_HOST_STOP_GIB=80","bash",str(repo/"scripts/run_gsm8k_guarded.sh"),task["name"],
        str(root/task["name"]/"memory.jsonl"),str(repo/".pixi/envs/gb10/bin/python"),"-u","-m",
        "sparsifier.eval.gsm8k_decision_pilot","--root",str(root),"--plan",str(root/plan_name),
        "--plan-sha256",plan_hash,"--task",task["name"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit",action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo/"results/tangent/gsm8k"
    calibration = load_calibration(root/CALIBRATION)
    manifest = prepare_manifest(load_manifest(root/"data/original-validation-v1.json"), calibration, seed=0)
    write_immutable(root/DECISION,manifest)
    plan = build_representative_plan(manifest, calibration)
    write_immutable(root/PLAN,plan)
    path = root/"decision64-light128-repcal-batch32-submissions-v1.json"
    with (root/"decision64-light128-repcal-batch32-submissions-v1.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        journal = json.loads(path.read_text()) if path.exists() else {"plan_sha256":digest(plan),"tasks":{}}
        if journal["plan_sha256"] != digest(plan):
            raise ValueError("decision pilot submission drift")
        ids = {}
        for task in plan["tasks"]:
            name = task["name"]
            if name in journal["tasks"]:
                old = journal["tasks"][name]
                if old["state"] != "submitted":
                    raise RuntimeError("ambiguous submission; reconcile "+name)
                ids[name] = old["job_id"]
                continue
            dependencies = [ids[parent] for parent in task["depends_on"]]
            argv = command(repo,task,dependencies,digest(plan))
            if not args.submit:
                print(json.dumps({"name":name,"phase":task["phase"],"dependencies":dependencies}))
                ids[name] = "JOB_FOR_"+name
                continue
            journal["tasks"][name] = {"state":"submitting","argv":argv}
            atomic_json(path,journal)
            result = subprocess.run(argv,cwd=repo,text=True,capture_output=True)
            match = re.search(r"Submitted batch job (\d+)",result.stdout)
            if result.returncode or not match:
                journal["tasks"][name].update(stdout=result.stdout,stderr=result.stderr)
                atomic_json(path,journal)
                raise RuntimeError("ambiguous submission "+name)
            ids[name] = int(match.group(1))
            journal["tasks"][name].update(state="submitted",job_id=ids[name],stdout=result.stdout)
            atomic_json(path,journal)
            print(result.stdout.strip(),flush=True)


if __name__ == "__main__":
    main()
