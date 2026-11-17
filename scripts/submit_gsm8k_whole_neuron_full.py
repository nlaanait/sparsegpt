"""Queue whole-model whole-neuron pruning and 128-to-512 GSM8K evaluation."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_decision_full import prepare_manifest, validate_protocol
from sparsifier.data.gsm8k_calibration import CALIBRATION, DECISION, FULL_DECISION, load_calibration, validate_calibration

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/tangent/gsm8k'
MANIFEST = ROOT / FULL_DECISION
REFERENCE = ROOT / 'qwen-9b-gsm8k-decision-dense-seed0-batch32-repcal-v1'
JOURNAL = ROOT / 'whole-neuron-full-repcal-batch32-submissions-v1.json'


def tasks():
    for sparsity in (30, 50, 60, 70, 80, 90, 99):
        for metric in ('ambient', 'downstream_weight_diagonal'):
            name = f"qwen-9b-gsm8k-whole-neuron-{metric.replace('_', '-')}-s{sparsity}-seed0-batch32-repcal-v1"
            yield name, metric, sparsity


def submit(name, argv, dependency=None):
    cmd = ['gbatch', '--gpus', '1', '--gpu-memory', '64G', '--memory', '80G',
           '--name', name, '--project', 'qwen-gsm8k', '--time', '24:00:00', '--auto-close']
    if dependency is not None:
        cmd += ['--depends-on', str(dependency)]
    return cmd + ['--', 'env', 'GSM8K_HOST_STOP_GIB=80', 'bash', str(REPO/'scripts/run_gsm8k_guarded.sh'), name,
                  str(ROOT/name/'memory.jsonl'), str(REPO/'.pixi/envs/gb10/bin/python'), '-u', *argv]


def enqueue(journal, name, argv, dependency, do_submit):
    saved = journal['tasks'].get(name)
    if saved:
        if saved['state'] != 'submitted':
            raise RuntimeError('reconcile ambiguous submission: '+name)
        return saved['job_id']
    cmd = submit(name, argv, dependency)
    if not do_submit:
        print(json.dumps({'name': name, 'command': cmd}), flush=True)
        return 'JOB_FOR_'+name
    journal['tasks'][name] = {'state': 'submitting', 'argv': cmd}
    atomic_json(JOURNAL, journal)
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if result.returncode or not match:
        journal['tasks'][name].update(stdout=result.stdout, stderr=result.stderr)
        atomic_json(JOURNAL, journal)
        raise RuntimeError('submission needs reconciliation: '+name)
    job = int(match.group(1))
    journal['tasks'][name].update(state='submitted', job_id=job, stdout=result.stdout)
    atomic_json(JOURNAL, journal)
    print(result.stdout.strip(), name, flush=True)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    calibration = load_calibration(ROOT/CALIBRATION)
    validate_calibration(calibration)
    manifest = prepare_manifest(load_manifest(ROOT/'data/original-validation-v1.json'), calibration)
    validate_protocol(manifest)
    old = load_manifest(ROOT/DECISION)
    if manifest['reference_manifest_sha256'] != old['manifest_sha256'] or manifest['rows'][:128] != old['rows']:
        raise ValueError('new screen differs from the existing dense reference')
    write_immutable(MANIFEST, manifest)
    with JOURNAL.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'manifest_sha256': manifest['manifest_sha256'], 'tasks': {}}
        if journal['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('submission manifest changed')
        pilot = ROOT/'decision64-light128-repcal-batch32-submissions-v1.json'
        if pilot.exists() and REFERENCE.name not in journal['tasks']:
            external = json.loads(pilot.read_text())['tasks'].get(REFERENCE.name)
            if external and external['state'] == 'submitted':
                journal['tasks'][REFERENCE.name] = {**external, 'reused_from': str(pilot)}
        reference_argv = ['-m', 'sparsifier.eval.gsm8k_decision', '--manifest', str(ROOT/DECISION),
            '--calibration-manifest', str(ROOT/CALIBRATION), '--output', str(REFERENCE)]
        dense_job = enqueue(journal, REFERENCE.name, reference_argv, None, args.submit)
        for name, metric, sparsity in tasks():
            pruning = ['-m', 'sparsifier.eval.gsm8k_whole_neuron', '--manifest', str(ROOT/CALIBRATION),
                       '--pilot-report', str(ROOT/'qwen-9b-gsm8k-dense-pilot-v2/report.json'),
                       '--c4-capture', str(ROOT/'capture-v1'), '--output', str(ROOT/name), '--blocks', 'all',
                       '--metric', metric, '--sparsity', str(sparsity/100), '--seed', '0', '--eval-count', '4', '--skip-behavior']
            evaluate = ['-m', 'sparsifier.eval.gsm8k_decision_full', '--manifest', str(MANIFEST),
                        '--calibration-manifest', str(ROOT/CALIBRATION), '--source', str(ROOT/name),
                        '--reference', str(REFERENCE), '--output', str(ROOT/(name+'-decision512'))]
            parent = dense_job
            for jobname, argv in ((name, pruning), (name+'-decision512', evaluate)):
                parent = enqueue(journal, jobname, argv, parent, args.submit)


if __name__ == '__main__':
    main()
