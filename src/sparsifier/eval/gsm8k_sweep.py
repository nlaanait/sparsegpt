"""Build the development MLP retention sweep; validate smoke gates separately."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION

ARMS = ('sparsegpt_task','local')
SPARSITIES = (.1,.3,.5,.7)
MODULES = 'mlp.gate_proj,mlp.up_proj,mlp.down_proj'


def smoke_gate(root, arm):
    name = 'task' if arm == 'sparsegpt_task' else arm
    version = 'v2' if arm == 'local' else 'v1'
    folder = root/f'qwen-9b-gsm8k-sequential-{name}-smoke-{version}'
    if not (folder/'report.json').exists(): return {'ready':False,'reason':'smoke report missing'}
    report = json.loads((folder/'report.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    if arm == 'local' and config.get('local_pruning_version') != LOCAL_PRUNING_VERSION:
        return {'ready':False,'reason':'legacy local smoke; rerun corrected pruning'}
    if report.get('config_sha256') != digest(config): raise ValueError('smoke provenance mismatch')
    expected = [[b,m] for b in (15,16) for m in MODULES.split(',')]
    if not report.get('complete'): return {'ready':False,'reason':'smoke incomplete'}
    if config['arm'] != arm or [r['site'] for r in report['stages']] != expected:
        raise ValueError('smoke scope mismatch')
    if report['behavior']['count'] != 4 or not 0 < report['whole_model_removed_fraction'] < 1:
        raise ValueError('smoke behavior or sparsity missing')
    return {'ready':True,'config_sha256':digest(config)}


def build_plan(calibration, validation):
    used = {r['source_index'] for r in calibration['rows']}
    held = {r['source_index'] for r in validation['rows']}
    if used & held: raise ValueError('validation/calibration source leakage')
    tasks = []
    for sparsity in SPARSITIES:
        for arm in ARMS:
            name = f'qwen-9b-gsm8k-mlp-{arm.replace("_","-")}-s{round(100*sparsity)}-seed0-v1'
            evaluation = name+'-original512'
            tasks.append({'name':name,'phase':'prune','arm':arm,'eligible_sparsity':sparsity,
                          'requires_smoke_arm':arm,'depends_on':[],
                          'submit_argv':['bash','scripts/submit_gsm8k_sequential.sh',name,arm,str(sparsity),'all',MODULES,'32']})
            tasks.append({'name':evaluation,'phase':'original_validation','arm':arm,
                          'depends_on':[name], 'submit_argv':['bash','scripts/submit_gsm8k_original.sh',evaluation,name]})
    return {'purpose':'development_MLP_retention_frontier_not_confirmatory',
            'calibration_manifest_sha256':calibration['manifest_sha256'],
            'validation_manifest_sha256':validation['manifest_sha256'],
            'dense_reference':'qwen-9b-gsm8k-original-dense-512-v1',
            'modules':MODULES.split(','),'blocks':'all','eligible_sparsities':list(SPARSITIES),
            'alpha':1.,'seed':0,'max_new_tokens':2048,'allowed_accuracy_drop':None,
            'memory_policy':'exclusive gflow; CUDA cap <=56 GiB; cgroup 80 GiB; host watchdog 96 GiB',
            'tasks':tasks,'limitations':['MLP only; attention and embeddings remain dense',
                'actual whole-model removed fraction must be measured',
                'alpha/seed are initial development settings, not frozen final selections',
                'validate smoke reports before prune submission; validate source completion before evaluation submission',
                'submit each task once and record gflow IDs; this file itself launches no jobs']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/tangent/gsm8k'))
    args = p.parse_args()
    plan = build_plan(load_manifest(args.root/'data/controlled-v1.json'),
                      load_manifest(args.root/'data/original-validation-v1.json'))
    write_immutable(args.root/'mlp-retention-sweep-plan-v1.json',plan)
    print(json.dumps({'tasks':len(plan['tasks']),'gates':{arm:smoke_gate(args.root,arm) for arm in ARMS}},indent=2))


if __name__ == '__main__': main()
