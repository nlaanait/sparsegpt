"""Queue matched representative-calibration 55/60/65% GSM8K experiments.

Reuse the successful local/downstream commands and existing 128-to-512
evaluation. Exclusive gflow jobs serialize each prune/evaluate pair. Existing
artifact controls have higher pruning priority. Journal every submission.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from sparsifier.data.gsm8k import digest, write_immutable
from sparsifier.eval.gsm8k import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/tangent/gsm8k'
DEST = RESULTS / 'sparsity-bracket-repcal-v1'
SPARSITIES = (.55, .60, .65)
METHODS = ('sparsegpt-task', 'local', 'downstream-weightdiag')


def prior_command(job):
    text = subprocess.check_output(['gjob','show',str(job)],text=True)
    if 'State=Finished (CD)' not in text:
        raise ValueError(f'reference job {job} is not completed')
    line = next(s.strip() for s in text.splitlines() if s.strip().startswith('Command='))
    return shlex.split(line.removeprefix('Command='))


def rename_worker(worker, old_source, source, old_output=None, output=None):
    old_output = old_output or old_source
    output = output or source
    values = {old_output.name:output.name, str(old_source):str(source),
        str(old_output):str(output), str(old_output/'memory.jsonl'):str(output/'memory.jsonl')}
    return [values.get(v,v) for v in worker]


def build_plan():
    local, downstream, evaluation = map(prior_command, (222,227,225))
    old_local = Path(local[local.index('--output')+1])
    old_down = Path(downstream[downstream.index('--output')+1])
    old_eval = Path(evaluation[evaluation.index('--output')+1])
    tasks=[]
    previous=221  # Successful matched dense reference; no dependency on an unrelated control.
    for sparsity in SPARSITIES:
        for method in METHODS:
            source=RESULTS/f'qwen-9b-gsm8k-mlp-{method}-s{round(100*sparsity)}-seed0-batch32-repcal-v1'
            is_downstream=method=='downstream-weightdiag'
            worker=rename_worker(downstream if is_downstream else local,
                                 old_down if is_downstream else old_local,source)
            worker[worker.index('--sparsity')+1]=str(sparsity)
            if not is_downstream:
                worker[worker.index('--arm')+1]='sparsegpt_task' if method=='sparsegpt-task' else 'local'
            tasks.append({'name':source.name,'phase':'pruning','method':method,'sparsity':sparsity,
                          'depends_on':[previous],'output':str(source),'worker':worker,'priority':8})
            output=source.with_name(source.name+'-decision512')
            worker=rename_worker(evaluation,old_down,source,old_eval,output)
            tasks.append({'name':output.name,'phase':'evaluation','method':method,'sparsity':sparsity,
                          'depends_on':[221,source.name],'output':str(output),'worker':worker,'priority':10})
            previous=output.name
    assert len(tasks)==18 and len({t['name'] for t in tasks})==18
    assert all(t['worker'][t['worker'].index('--batch-size')+1]=='32' for t in tasks)
    assert all(t['worker'][t['worker'].index('--output')+1]==t['output'] for t in tasks)
    configs={p.name:json.loads((p/'config.json').read_text()) for p in (old_local,old_down)}
    return {'purpose':'measure_sample_accuracy_zero_frontier_not_population_zero',
        'model':'Qwen/Qwen3.5-9B','eligible_sparsities':list(SPARSITIES),'methods':list(METHODS),
        'calibration':'original representative-calibration 4096 C4 + 4096 task prompt tokens',
        'precision':'original production FP32; separate artifact controls retain their own namespaces',
        'evaluation':'same first 128 questions at batch32, 2048 new tokens; existing 15-point screen continues passing candidates to 512',
        'dense_reference_job':221,'scheduling':'ascending sparsity, then SparseGPT/local/downstream; each pair depends on the previous evaluation; pruning priority 8 preserves existing controls',
        'baseline_sources':{name:c['sources'] for name,c in configs.items()},
        'source_comparison':'Check actual candidate source hashes against these baselines before combining measurements',
        'endpoint_caveats':['Existing SparseGPT 50/70 endpoints use the earlier calibration/runtime protocol',
                            'Existing representative local 50% evaluation uses batch8; the new nine use matched batch32',
                            'A score of 0/128 is an observed sample zero, not proof of zero population accuracy'],
        'tasks':tasks}


def command(task, dependencies):
    return ['gbatch','--gpus','1','--gpu-memory','64G','--memory','80G',
        '--name',task['name'],'--project','qwen-sparsegpt','--time','06:00:00',
        '--priority',str(task['priority']),'--auto-close','--depends-on-all',
        ','.join(map(str,dependencies)),'--',*task['worker']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'submissions.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=DEST/'plan.json'
        plan=json.loads(path.read_text()) if path.exists() else build_plan()
        write_immutable(path,plan)
        path=DEST/'submissions.json'
        journal=json.loads(path.read_text()) if path.exists() else {'plan_sha256':digest(plan),
            'submitter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tasks':{}}
        if journal['plan_sha256']!=digest(plan):raise ValueError('bracket plan drift')
        ids={}
        for task in plan['tasks']:
            name=task['name'];old=journal['tasks'].get(name)
            if old:
                if old['state']!='submitted':raise RuntimeError(f'ambiguous submission {name}; inspect gflow before retry')
                ids[name]=old['job_id'];continue
            deps=[ids[d] if isinstance(d,str) else d for d in task['depends_on']]
            argv=command(task,deps)
            if not args.submit:
                print(json.dumps({'name':name,'dependencies':deps,'argv':argv}))
                ids[name]='JOB_FOR_'+name;continue
            journal['tasks'][name]={'state':'submitting','argv':argv}
            atomic_json(path,journal)
            result=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
            match=re.search(r'Submitted batch job (\d+)',result.stdout)
            if result.returncode or not match:
                journal['tasks'][name].update(stdout=result.stdout,stderr=result.stderr)
                atomic_json(path,journal)
                raise RuntimeError(f'ambiguous submission {name}; inspect gflow before retry')
            ids[name]=int(match.group(1))
            journal['tasks'][name].update(state='submitted',job_id=ids[name],stdout=result.stdout)
            atomic_json(path,journal)
            print(result.stdout.strip(),flush=True)


if __name__=='__main__':main()
