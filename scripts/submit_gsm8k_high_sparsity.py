"""Create and submit a separate, smoke-gated 80/90/99% MLP sweep."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
from sparsifier.data.gsm8k import digest, write_immutable
from sparsifier.eval.gsm8k import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/tangent/gsm8k'
JOURNAL = RESULTS / 'high-sparsity-submissions-v1.json'


def create_plans():
    outputs = {}
    for kind, source in [('mlp', 'mlp-retention-sweep-plan-v1.json'),
                         ('downstream', 'downstream-retention-sweep-plan-v1.json')]:
        plan = json.loads((RESULTS / source).read_text())
        tasks = []
        arms = ('sparsegpt_task', 'local') if kind == 'mlp' else ('downstream',)
        for sparsity in (.8, .9, .99):
            for arm in arms:
                name = f'qwen-9b-gsm8k-mlp-{arm.replace("_", "-")}-s{round(sparsity*100)}-seed0-v1'
                prune = {'name': name, 'phase': 'prune', 'arm': arm, 'eligible_sparsity': sparsity,
                         'blocks': 'all', 'eval_count': 32,
                         'depends_on': [] if kind == 'mlp' else ['qwen-9b-gsm8k-sequential-downstream-smoke-v1']}
                if kind == 'mlp': prune['requires_smoke_arm'] = arm
                tasks.extend([prune, {'name': name + '-original512', 'phase': 'original_validation',
                                      'arm': arm, 'depends_on': [name]}])
        plan['eligible_sparsities'] = [.8, .9, .99]
        plan['purpose'] = 'development_high_sparsity_MLP_retention_not_confirmatory'
        plan['tasks'] = tasks
        name = f'{kind}-high-sparsity-plan-v1.json'
        write_immutable(RESULTS / name, plan)
        outputs[kind] = (name, plan, digest(plan))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    plans = create_plans()
    with (RESULTS / 'high-sparsity-submissions-v1.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hashes = {kind: row[2] for kind, row in plans.items()}
        journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'plan_sha256': hashes, 'tasks': {}}
        if journal['plan_sha256'] != hashes: raise ValueError('plan drift')
        for kind, (plan_name, plan, plan_hash) in plans.items():
            for task in plan['tasks']:
                name, phase, arm = task['name'], task['phase'], task['arm']
                if name in journal['tasks']:
                    if journal['tasks'][name]['state'] != 'submitted':
                        raise RuntimeError(f'ambiguous submission: {name}; reconcile with gflow')
                    continue
                if phase == 'prune': deps = [86 if arm == 'downstream' else 62 if arm == 'sparsegpt_task' else 63]
                else:
                    parent = task['depends_on'][0]
                    deps = [journal['tasks'][parent]['job_id'], 64] if args.submit else ['PRUNE_JOB', 64]
                gpu_cap = '80G' if arm == 'local' and phase == 'prune' else '64G'
                module = 'sparsifier.eval.gsm8k_downstream_sweep' if kind == 'downstream' else 'sparsifier.eval.gsm8k_sweep_worker'
                argv = ['gbatch', '--gpus','1','--gpu-memory',gpu_cap,'--memory','80G','--name',name,
                        '--project','qwen-gsm8k','--time','24:00:00','--auto-close','--depends-on-all',
                        ','.join(map(str,deps)),'--','bash',str(ROOT/'scripts/run_gsm8k_guarded.sh'),name,
                        str(RESULTS/name/'memory.jsonl'),str(ROOT/'.pixi/envs/gb10/bin/python'),'-u','-m',module,
                        '--root',str(RESULTS),'--task',name,'--plan',plan_name,'--plan-sha256',plan_hash]
                if phase == 'original_validation': argv.append('--screen-128')
                if not args.submit:
                    print(json.dumps({'name':name,'dependencies':deps,'argv':argv})); continue
                journal['tasks'][name] = {'state':'submitting','argv':argv}
                atomic_json(JOURNAL,journal)
                result = subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
                match = re.search(r'Submitted batch job (\d+)',result.stdout)
                if result.returncode or not match:
                    journal['tasks'][name].update(stdout=result.stdout,stderr=result.stderr)
                    atomic_json(JOURNAL,journal)
                    raise RuntimeError(f'ambiguous submission: {name}; reconcile with gflow')
                journal['tasks'][name].update(state='submitted',job_id=int(match.group(1)),stdout=result.stdout)
                atomic_json(JOURNAL,journal)
                print(result.stdout.strip(),flush=True)

if __name__ == '__main__': main()
