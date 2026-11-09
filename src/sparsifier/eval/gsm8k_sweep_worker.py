"""Artifact-gated entry point for one scheduled MLP retention sweep phase."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
from sparsifier.data.gsm8k import digest, load_manifest
from sparsifier.eval.gsm8k_sweep import smoke_gate
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--plan',default='mlp-retention-sweep-plan-v1.json')
    p.add_argument('--plan-sha256')
    args = p.parse_args()
    root = args.root
    plan = json.loads((root/args.plan).read_text())
    if args.plan != 'mlp-retention-sweep-plan-v1.json' and not args.plan_sha256:
        p.error('alternate plan requires --plan-sha256')
    if args.plan_sha256 and digest(plan) != args.plan_sha256:
        raise ValueError('sweep plan drift')
    matches = [t for t in plan['tasks'] if t['name'] == args.task]
    if len(matches) != 1: raise ValueError('task not uniquely identified in plan')
    task = matches[0]
    if task['arm'] == 'local' and plan.get('local_normalization', LOCAL_PRUNING_VERSION) != LOCAL_PRUNING_VERSION:
        raise ValueError('legacy local normalization is no longer supported')
    for filename,key in [('controlled-v1.json','calibration_manifest_sha256'),
                         ('original-validation-v1.json','validation_manifest_sha256')]:
        if load_manifest(root/'data'/filename)['manifest_sha256'] != plan[key]:
            raise ValueError('sweep manifest drift')
    output = root/task['name']
    if task['phase'] == 'prune':
        gate = smoke_gate(root,task['arm'])
        if not gate['ready']: raise RuntimeError(f'smoke gate closed: {gate}')
        module = 'sparsifier.eval.gsm8k_sequential'
        argv = ['--manifest',str(root/'data/controlled-v1.json'),
                '--pilot-report',str(root/'qwen-9b-gsm8k-dense-pilot-v2/report.json'),
                '--c4-capture',str(root/'capture-v1'),'--output',str(output),
                '--arm',task['arm'],'--sparsity',str(task['eligible_sparsity']),
                '--blocks','all','--modules',','.join(plan['modules']),
                '--alpha',str(plan['alpha']),'--seed',str(plan['seed']),
                '--eval-count','32','--max-new-tokens',str(plan['max_new_tokens'])]
    else:
        from sparsifier.eval.gsm8k_original import frozen_stages
        source = root/task['depends_on'][0]
        source_config,_ = frozen_stages(source)
        if source_config['blocks'] != 'all' or source_config['arm'] != task['arm']:
            raise ValueError('source candidate scope mismatch')
        if task['arm'] == 'local' and source_config.get('local_pruning_version') != LOCAL_PRUNING_VERSION:
            raise ValueError('legacy local candidate must be regenerated with corrected pruning')
        module = 'sparsifier.eval.gsm8k_original'
        argv = ['--manifest',str(root/'data/original-validation-v1.json'),
                '--source',str(source),'--output',str(output),
                '--max-new-tokens',str(plan['max_new_tokens'])]
    sys.argv = [module,*argv]
    runpy.run_module(module,run_name='__main__')


if __name__ == '__main__': main()
