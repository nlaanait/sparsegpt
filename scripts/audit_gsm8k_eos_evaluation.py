"""Independently regrade corrected GSM8K answers from saved raw token IDs."""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer
from sparsifier.eval.gsm8k_batching import trim_generated
from sparsifier.prune.whole_neuron_io import file_hash


def audit(evaluation, tokenizer, manifest, *, source=None, allow_partial=False):
    folder = Path(evaluation)
    config = json.loads((folder/'config.json').read_text())
    policy = config['stopping_policy']
    if (policy['version'] != 'gsm8k-model-and-tokenizer-eos-v1'
            or policy['eos_token_ids'] != [248044, 248046]
            or tokenizer.eos_token_id != 248046):
        raise ValueError('unexpected stopping policy')
    config_hash = digest(config)
    gold = {r['id']: r for r in manifest['rows']}
    assert config['manifest_sha256'] == manifest['manifest_sha256']
    assert config['evaluation_ids'] == [r['id'] for r in manifest['rows']]
    if source is not None:
        source = Path(source)
        assert config['source_config_sha256'] == digest(json.loads((source/'config.json').read_text()))
        assert config['source_report_sha256'] == file_hash(source/'report.json')
    elif config['arm'] != 'dense':
        raise ValueError('candidate source required for checkpoint provenance')
    report_path = folder/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    if not allow_partial and report is None:
        raise ValueError('completed evaluation report required')
    if report is not None:
        assert report['config_sha256'] == config_hash
    body_bytes = (folder/'answers.jsonl').read_bytes()
    body = body_bytes.decode()
    lines = body.splitlines()
    if not body.endswith('\n'):
        if not allow_partial:
            raise ValueError('incomplete answer line')
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    ids = [row['id'] for row in rows]
    assert ids == config['evaluation_ids'][:len(ids)] and len(set(ids)) == len(ids)
    assert len(rows) > 0
    if not allow_partial:
        assert len(rows) >= 128
    raw_files, terminal_ids = {}, {}
    marker_correct, truncated_marker_correct, truncated_complete_marker = [], [], []
    for index, row in enumerate(rows):
        assert row['config_sha256'] == config_hash
        assert row['expected_answer'] == gold[row['id']]['answer'], row['id']
        size = config['batching']['batch_size']
        start = (index//size)*size
        assert row['batch_ids'] == config['evaluation_ids'][start:start+size]
        path = folder/'raw-generations'/(digest(row['batch_ids'])+'.json')
        if path not in raw_files:
            raw = json.loads(path.read_text())
            assert raw['config_sha256'] == config_hash
            assert raw['eos_token_ids'] == policy['eos_token_ids']
            assert raw['max_new_tokens'] == config['max_new_tokens']
            assert len(raw['batch_ids']) == len(raw['generated_token_ids'])
            raw_files[path] = raw
        raw = raw_files[path]
        assert raw['batch_ids'] == row['batch_ids']
        tokens = raw['generated_token_ids'][raw['batch_ids'].index(row['id'])]
        assert 0 < len(tokens) <= raw['max_new_tokens']
        generated = trim_generated(tokens, set(policy['eos_token_ids']))
        truncated = len(generated) >= config['max_new_tokens'] and generated[-1] not in policy['eos_token_ids']
        text = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_answer(text, truncated=truncated)
        correct = parsed is not None and parsed == Fraction(row['expected_answer'])
        assert row['text'] == text, row['id']
        assert row['generated_tokens'] == len(generated), row['id']
        assert row['truncated'] == truncated, row['id']
        assert row['parsed_answer'] == (str(parsed) if parsed is not None else None), row['id']
        assert row['correct'] == correct, row['id']
        marked = parse_answer(text)
        marked_correct = marked is not None and marked == Fraction(gold[row['id']]['answer'])
        marker_correct.append(marked_correct)
        truncated_marker_correct.append(truncated and marked_correct)
        answer_text = text.rsplit('</think>', 1)[-1]
        complete_line = bool(re.search(r'(?:Final answer\s*:|####)[ \t]*[^\n]+\n', answer_text, re.I))
        truncated_complete_marker.append(truncated and marked_correct and complete_line)
        if not truncated:
            terminal_ids[str(generated[-1])] = terminal_ids.get(str(generated[-1]), 0)+1
    first = rows[:128]
    if report is not None:
        assert report['behavior']['count'] == len(rows)
        assert report['behavior']['accuracy'] == sum(r['correct'] for r in rows)/len(rows)
        assert report['behavior']['truncated'] == sum(r['truncated'] for r in rows)
    return {'evaluation': str(folder.resolve()), 'source': str(source.resolve()) if source else None,
            'config_sha256': config_hash, 'report_sha256': file_hash(report_path) if report else None,
            'answers_sha256': sha256(body_bytes).hexdigest(),
            'audited_records_sha256': digest(rows),
            'common128_complete': len(rows) >= 128,
            'scope': 'complete_evaluation' if report else 'live_prefix_not_final_evaluation',
            'raw_files': {str(p.name): file_hash(p) for p in raw_files},
            'raw_token_rescore_mismatches': 0, 'count': len(rows),
            'correct': sum(r['correct'] for r in rows), 'truncated': sum(r['truncated'] for r in rows),
            'strict_marker_correct_ignoring_truncation': sum(marker_correct),
            'truncated_strict_marker_correct': sum(truncated_marker_correct),
            'truncated_correct_with_complete_marker_line': sum(truncated_complete_marker),
            'common128': {'ids_sha256': digest(ids[:128]), 'count': len(first),
                          'correct': sum(r['correct'] for r in first),
                          'truncated': sum(r['truncated'] for r in first),
                          'strict_marker_correct_ignoring_truncation': sum(marker_correct[:128]),
                          'truncated_correct_with_complete_marker_line': sum(truncated_complete_marker[:128])},
            'terminal_token_counts': terminal_ids, 'stopping_policy': policy}


def main():
    from transformers import AutoTokenizer
    from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true',
                        help='audit a live prefix; label it separately from a completed evaluation')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    result = audit(args.evaluation, tokenizer, load_manifest(args.manifest), source=args.source,
                   allow_partial=args.allow_partial)
    result['auditor_sha256'] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_files', 'stopping_policy')}))


if __name__ == '__main__':
    main()
