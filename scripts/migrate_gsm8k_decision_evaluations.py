"""Replace only the specified queued full evaluations with shared 64/128 jobs.

Submit and journal the replacement before cancelling its old queued job.
Refuse to cancel a job that has started. Historical result files are untouched.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from scripts.submit_gsm8k_decision_pilot import MANIFEST

REPLACEMENTS = [(121,120,"qwen-9b-gsm8k-mlp-local-s50-seed0-v2"),
    (123,122,"qwen-9b-gsm8k-mlp-local-s70-seed0-v2"),
    (125,124,"qwen-9b-gsm8k-mlp-local-s80-seed0-v2"),
    (127,126,"qwen-9b-gsm8k-mlp-local-s90-seed0-v2"),
    (129,128,"qwen-9b-gsm8k-mlp-local-s99-seed0-v2"),
    (138,137,"qwen-9b-gsm8k-mlp-downstream-s30-seed0-weightdiag-v1"),
    (140,139,"qwen-9b-gsm8k-mlp-downstream-s50-seed0-weightdiag-v1"),
    (142,141,"qwen-9b-gsm8k-mlp-downstream-s70-seed0-weightdiag-v1")]


def queued_check(job, source):
    result = subprocess.run(["gjob","show",str(job)],text=True,capture_output=True,check=True)
    if "State=Queued (PD)" not in result.stdout or source+"-original512" not in result.stdout:
        raise RuntimeError(f"job {job} no longer matches the intended queued evaluation; do not cancel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-job",type=int,required=True)
    parser.add_argument("--submit",action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo/"results/tangent/gsm8k"
    manifest = load_manifest(root/MANIFEST)
    policy = {"manifest_sha256":manifest["manifest_sha256"],"dense_job":args.dense_job,
              "replacements":REPLACEMENTS,"screen":64,"light":128,"generation_cap":2048}
    write_immutable(root/"decision64-light128-migration-plan-v1.json",policy)
    path = root/"decision64-light128-migration-submissions-v1.json"
    with (root/"decision64-light128-migration-v1.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        journal = json.loads(path.read_text()) if path.exists() else {"policy_sha256":digest(policy),"tasks":{}}
        if journal["policy_sha256"] != digest(policy):
            raise ValueError("migration policy drift")
        for old, parent, source in REPLACEMENTS:
            name = source+"-decision128-seed0-v1"
            state = journal["tasks"].get(name)
            if state and state["state"] == "replaced":
                continue
            if state and state["state"] != "submitted":
                raise RuntimeError("ambiguous migration submission; reconcile "+name)
            queued_check(old,source)
            if state is None:
                argv = ["gbatch","--gpus","1","--gpu-memory","64G","--memory","80G",
                    "--name",name,"--project","qwen-gsm8k","--time","24:00:00","--auto-close",
                    "--priority","100" if "-local-" in source else "10",
                    "--depends-on-all",f"{parent},{args.dense_job}","--","bash",str(repo/"scripts/run_gsm8k_guarded.sh"),
                    name,str(root/name/"memory.jsonl"),str(repo/".pixi/envs/gb10/bin/python"),"-u","-m",
                    "sparsifier.eval.gsm8k_decision","--manifest",str(root/MANIFEST),
                    "--calibration-manifest",str(root/"data/controlled-v1.json"),"--source",str(root/source),
                    "--reference",str(root/"qwen-9b-gsm8k-decision-dense-seed0-v1"),"--output",str(root/name)]
                if not args.submit:
                    print(json.dumps({"old":old,"replacement":name,"depends_on":[parent,args.dense_job]}))
                    continue
                journal["tasks"][name] = {"state":"submitting","old_job":old,"argv":argv}
                atomic_json(path,journal)
                result = subprocess.run(argv,cwd=repo,text=True,capture_output=True)
                match = re.search(r"Submitted batch job (\d+)",result.stdout)
                if result.returncode or not match:
                    journal["tasks"][name].update(stdout=result.stdout,stderr=result.stderr)
                    atomic_json(path,journal)
                    raise RuntimeError("ambiguous migration submission "+name)
                journal["tasks"][name].update(state="submitted",job_id=int(match.group(1)))
                atomic_json(path,journal)
            if args.submit:
                queued_check(old,source)
                subprocess.run(["gcancel",str(old)],check=True)
                journal["tasks"][name]["state"] = "replaced"
                atomic_json(path,journal)
                print(json.dumps({"old":old,"new":journal["tasks"][name]["job_id"]}),flush=True)


if __name__ == "__main__":
    main()
