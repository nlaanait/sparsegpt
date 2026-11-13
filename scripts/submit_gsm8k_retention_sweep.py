"""Submit artifact-gated sweep tasks once; journal ambiguous submissions for review."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import atomic_json


def command(repo,task,dependencies):
    root = repo/'results/tangent/gsm8k'
    argv = ['gbatch','--gpus','1','--gpu-memory','64G','--memory','80G',
            '--name',task['name'],'--project','qwen-gsm8k','--time','24:00:00','--auto-close',
            '--depends-on-all',','.join(map(str,dependencies)),'--',
            'bash',str(repo/'scripts/run_gsm8k_guarded.sh'),task['name'],
            str(root/task['name']/'memory.jsonl'),str(repo/'.pixi/envs/default/bin/python'),
            '-u','-m','sparsifier.eval.gsm8k_sweep_worker','--root',str(root),'--task',task['name']]
    if task.get('phase') == 'original_validation': argv.append('--screen-128')
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-smoke-job',type=int,required=True)
    p.add_argument('--local-smoke-job',type=int,required=True)
    p.add_argument('--dense-job',type=int,required=True)
    p.add_argument('--submit',action='store_true')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]; root = repo/'results/tangent/gsm8k'
    plan = json.loads((root/'mlp-retention-sweep-plan-v1.json').read_text())
    smoke = {'sparsegpt_task':args.task_smoke_job,'local':args.local_smoke_job}
    journal_path = root/'mlp-retention-submissions-v1.json'
    with (root/'mlp-retention-submissions-v1.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            'plan_sha256':digest(plan),'smoke_jobs':smoke,'dense_job':args.dense_job,'tasks':{}}
        if journal['plan_sha256'] != digest(plan) or journal['smoke_jobs'] != smoke or journal['dense_job'] != args.dense_job:
            raise ValueError('submission plan/dependency drift')
        ids = {}
        for task in plan['tasks']:
            name = task['name']
            old = journal['tasks'].get(name)
            if old:
                if old['state'] != 'submitted':
                    raise RuntimeError(f'ambiguous submission {name}; reconcile with gflow before retrying')
                ids[name] = old['job_id']; continue
            deps = [smoke[task['arm']]] if task['phase']=='prune' else [ids[task['depends_on'][0]],args.dense_job]
            cmd = command(repo,task,deps)
            if not args.submit:
                print(json.dumps({'task':name,'phase':task['phase'],'dependencies':deps}))
                ids[name] = 'JOB_ID_FOR_'+name
                continue
            journal['tasks'][name] = {'state':'submitting','argv':cmd}
            atomic_json(journal_path,journal)
            result = subprocess.run(cmd,cwd=repo,text=True,capture_output=True)
            match = re.search(r'Submitted batch job (\d+)',result.stdout)
            if result.returncode or not match:
                journal['tasks'][name].update(stdout=result.stdout,stderr=result.stderr)
                atomic_json(journal_path,journal)
                raise RuntimeError(f'submission uncertain for {name}; inspect journal')
            job_id = int(match.group(1)); ids[name] = job_id
            journal['tasks'][name].update(state='submitted',job_id=job_id,stdout=result.stdout)
            atomic_json(journal_path,journal)
            print(result.stdout.strip(),flush=True)


if __name__ == '__main__': main()
