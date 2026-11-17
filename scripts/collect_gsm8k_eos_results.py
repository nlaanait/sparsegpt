"""Collect completed frozen EOS experiments on CPU; never submit or alter jobs.

Writes independent audits and collector/progress.json, leaving the human status
pages and main progress journal for review. Reuses completed audits only while
all bound config/report/answer hashes and auditor source hashes still match.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from sparsifier.data.gsm8k import digest, load_manifest
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
from sparsifier.prune.whole_neuron_io import file_hash
from scripts.audit_gsm8k_eos_evaluation import audit
from scripts.compare_gsm8k_eos_evaluations import compare


def load(path):
    return json.loads(Path(path).read_text())


def cpu_evidence(base, source, config_hash, report_hash, label, snapshot):
    for directory in (base/'pruning-artifact-audit-v1', base/'sparsity-bracket-repcal-v1/audits',
                      base/'eos-corrected-v1/audits'):
        for path in sorted(directory.glob('*cpu*.json')):
            for record in load(path).get('sources', []):
                if (record['source'] == str(source) and record['config_sha256'] == config_hash
                        and record['report_sha256'] == report_hash):
                    assert len(record['stages']) == 96 and all(s['finite'] for s in record['stages'])
                    return path, record
    from scripts.audit_gsm8k_pruning_artifacts import audit_source
    record = audit_source(source, snapshot)
    assert len(record['stages']) == 96 and all(s['finite'] for s in record['stages'])
    path = base/f'eos-corrected-v1/audits/{label}-cpu.json'
    atomic_json(path, {'schema': 1, 'purpose': 'CPU_artifact_audit_not_behavioral_retention',
        'script_sha256': file_hash(Path(__file__).with_name('audit_gsm8k_pruning_artifacts.py')),
        'sources': [record], 'answers': []})
    return path, record


def collect(base, task, tokenizer, snapshot):
    folder = Path(task['output']); label = task['label']
    config = load(folder/'config.json'); report = load(folder/'report.json')
    assert config['model'] == MODEL and config['revision'] == REVISION
    assert report['config_sha256'] == digest(config)
    assert report.get('complete') or report.get('screen_rejected')
    audits = base/'eos-corrected-v1/audits'; audits.mkdir(parents=True, exist_ok=True)
    raw_path = audits/f'{label}.json'
    auditor_hash = file_hash(Path(__file__).with_name('audit_gsm8k_eos_evaluation.py'))
    previous = load(raw_path) if raw_path.exists() else {}
    reusable = (previous.get('config_sha256') == digest(config)
        and previous.get('report_sha256') == file_hash(folder/'report.json')
        and previous.get('answers_sha256') == file_hash(folder/'answers.jsonl')
        and previous.get('auditor_sha256') == auditor_hash and previous.get('common128_complete'))
    source = Path(task['source']) if task.get('source') else None
    if reusable:
        assert all(file_hash(folder/'raw-generations'/name) == checksum
                   for name, checksum in previous['raw_files'].items())
        result = previous
    else:
        manifest = load_manifest(base/'data'/('decision64-light128-repcal-seed0-v1.json'
                    if source is None else 'decision128-full512-repcal-seed0-batch8-v1.json'))
        result = audit(folder, tokenizer, manifest, source=source)
        result['auditor_sha256'] = auditor_hash
        atomic_json(raw_path, result)
    assert result['common128']['count'] == 128 and result['raw_token_rescore_mismatches'] == 0
    cpu_path = None
    if source is not None:
        assert config['source_config_sha256'] == digest(load(source/'config.json'))
        assert config['source_report_sha256'] == file_hash(source/'report.json')
        cpu_path, cpu = cpu_evidence(base, source, config['source_config_sha256'],
                                    config['source_report_sha256'], label, snapshot)
        assert config['dense_config_sha256'] == digest(load(base/'eos-corrected-v1/dense/config.json'))
        common = result['common128']
        atomic_json(audits/f'{label}-checkpoint-score-link.json', {
            'source': str(source), 'source_config_sha256': cpu['config_sha256'],
            'source_report_sha256': cpu['report_sha256'], 'source_cpu_audit': str(cpu_path),
            'source_cpu_audit_sha256': file_hash(cpu_path), 'verified_stage_count': 96,
            'evaluation_config_sha256': digest(config),
            'evaluation_report_sha256': file_hash(folder/'report.json'),
            'answers_sha256': file_hash(folder/'answers.jsonl'), 'raw_audit_sha256': file_hash(raw_path),
            'dense_config_sha256': config['dense_config_sha256'],
            'common128_ids_sha256': common['ids_sha256'], 'count': 128,
            'correct': common['correct'], 'truncated': common['truncated'],
            'full_evaluation_count': result['count'], 'full512_complete': bool(report['complete'])})
        historical = Path(str(source)+'-decision512')
        if (historical/'report.json').exists():
            comparison = compare(historical, folder, raw_path)
            comparison['comparator_sha256'] = file_hash(Path(__file__).with_name('compare_gsm8k_eos_evaluations.py'))
            assert comparison['decoded_prefix_matches'] == 128
            assert comparison['max_answer_nll_abs_difference'] == 0.
            atomic_json(audits/f'{label}-historical-comparison.json', comparison)
    return {'label': label, 'job_id': task['job_id'], 'state': 'verified',
        'common128': result['common128'], 'full_evaluation_count': result['count'],
        'full512_complete': bool(report['complete'] and source is not None),
        'screen_rejected': bool(report.get('screen_rejected')), 'raw_audit': str(raw_path),
        'raw_audit_sha256': file_hash(raw_path), 'cpu_audit': str(cpu_path) if cpu_path else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('results/tangent/gsm8k'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=45, choices=range(5,61))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("run with CUDA_VISIBLE_DEVICES='' (CPU only)")
    base = args.base.resolve(); journal = base/'eos-corrected-v1/submissions.json'
    state_path = base/'eos-corrected-v1/collector/progress.json'; state_path.parent.mkdir(exist_ok=True)
    lock = (state_path.parent/'collector.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_files = [Path(__file__), Path(__file__).with_name('audit_gsm8k_eos_evaluation.py'),
        Path(__file__).with_name('compare_gsm8k_eos_evaluations.py'),
        Path(__file__).with_name('audit_gsm8k_pruning_artifacts.py'),
        Path(__file__).parents[1]/'src/sparsifier/data/gsm8k.py']
    frozen_hashes = {path: file_hash(path) for path in frozen_files}
    tokenizer = None; verified = {}; emitted = {}
    while True:
        assert all(file_hash(path) == checksum for path, checksum in frozen_hashes.items()), 'collector/auditor/parser source changed; restart the CPU collector explicitly'
        failed_jobs = []
        submission = load(journal); tasks = submission['tasks']+submission['updates']
        assert len({t['label'] for t in tasks}) == len(tasks)
        for task in tasks:
            label = task['label']; folder = Path(task['output'])
            if label not in verified:
                shown = subprocess.run(['gjob','show',str(task['job_id'])], check=True,
                                       capture_output=True,text=True).stdout
                state_match = re.search(r'^\s*State=.*\(([A-Z]+)\)',shown,re.M)
                if state_match is None:
                    raise ValueError('unrecognized scheduler state for '+label)
                job_state = state_match.group(1)
                if job_state not in ('CD','PD','R'):
                    failed_jobs.append({'label':label,'job_id':task['job_id'],'state':job_state})
                if job_state != 'CD':
                    continue
            if not (folder/'report.json').exists():
                continue
            report = load(folder/'report.json')
            if not (report.get('complete') or report.get('screen_rejected')):
                continue
            raw_signature = digest({p.name:file_hash(p) for p in sorted((folder/'raw-generations').glob('*.json'))})
            source_paths = [Path(task['source'])/name for name in ('config.json','report.json')] if task.get('source') else []
            signature = (file_hash(folder/'report.json'), file_hash(folder/'answers.jsonl'),
                         file_hash(folder/'config.json'), raw_signature,
                         *[file_hash(p) for p in source_paths])
            if verified.get(label, {}).get('input_signature') == list(signature):
                continue
            shown = subprocess.run(['gjob','show',str(task['job_id'])], check=True,
                                   capture_output=True,text=True).stdout
            if re.search(r'^\s*State=Finished \(CD\)',shown,re.M) is None:
                continue
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
            row = collect(base,task,tokenizer,args.snapshot)
            row['input_signature'] = list(signature); verified[label] = row
            if emitted.get(label) != row['raw_audit_sha256']:
                print(json.dumps({'event':'verified','label':label,'job_id':task['job_id'],
                    'common128':row['common128']}),flush=True)
                emitted[label] = row['raw_audit_sha256']
        state = {'scope':'independent_collection_not_goal_completion','pid':os.getpid(),
            'updated_at':datetime.now(timezone.utc).isoformat(),
            'collector_sha256':file_hash(Path(__file__)), 'expected_evaluations':len(tasks),
            'verified_count':len(verified),'verified':list(verified.values()),
            'pending_labels':[t['label'] for t in tasks if t['label'] not in verified],
            'all_evaluations_verified':len(verified)==len(tasks), 'failed_jobs':failed_jobs}
        atomic_json(state_path,state)
        if failed_jobs:
            raise RuntimeError('scheduler requires attention: '+json.dumps(failed_jobs))
        if not args.watch or state['all_evaluations_verified']:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
