"""Pair historical and corrected scores only when EOS is the isolated change."""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path

from sparsifier.data.gsm8k import digest, parse_answer

INPUT_PROVENANCE = ('prompt_sha256', 'input_tokens', 'batch_ids', 'batch_size', 'batching')


def compare(historical, corrected, corrected_audit, *, count=128):
    historical, corrected = Path(historical), Path(corrected)
    old_config = json.loads((historical/'config.json').read_text())
    new_config = json.loads((corrected/'config.json').read_text())
    differences = sorted(k for k in old_config.keys() | new_config.keys()
                         if old_config.get(k) != new_config.get(k))
    unexpected = set(differences)-{'stopping_policy', 'dense_config_sha256'}
    if unexpected:
        raise ValueError('EOS is not isolated; configuration differs: '+', '.join(sorted(unexpected)))
    if new_config.get('stopping_policy', {}).get('version') != 'gsm8k-model-and-tokenizer-eos-v1':
        raise ValueError('corrected EOS policy required')
    audit = json.loads(Path(corrected_audit).read_text())
    assert audit['config_sha256'] == digest(new_config)
    old_bytes, new_bytes = [(p/'answers.jsonl').read_bytes() for p in (historical, corrected)]
    def records(body):
        lines = body.decode().splitlines()
        if not body.endswith(b'\n'):
            lines = lines[:-1]
        return [json.loads(line) for line in lines]
    old_rows, new_rows = records(old_bytes), records(new_bytes)
    assert digest(new_rows[:audit['count']]) == audit['audited_records_sha256']
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError('comparison count must be in 1..128')
    assert min(len(old_rows), len(new_rows), audit['count']) >= count
    old_rows, new_rows = old_rows[:count], new_rows[:count]
    ids = [r['id'] for r in old_rows]
    assert ids == old_config['evaluation_ids'][:count] == new_config['evaluation_ids'][:count]
    assert ids == [r['id'] for r in new_rows] and len(set(ids)) == count
    differences_nll, prefixes, gained, lost = [], [], [], []
    old_marked, new_marked = 0, 0
    for old, new in zip(old_rows, new_rows):
        assert old['config_sha256'] == digest(old_config)
        assert new['config_sha256'] == digest(new_config)
        assert old['expected_answer'] == new['expected_answer']
        for field in INPUT_PROVENANCE:
            if field not in old or field not in new or old[field] != new[field]:
                raise ValueError('recorded generation inputs differ or are missing: '+field)
        expected = Fraction(new['expected_answer'])
        for row in (old, new):
            value = parse_answer(row['text'], truncated=row['truncated'])
            assert row['correct'] == (value is not None and value == expected)
            assert math.isfinite(row['answer_nll'])
        old_marked += parse_answer(old['text']) == expected
        new_marked += parse_answer(new['text']) == expected
        differences_nll.append(abs(old['answer_nll']-new['answer_nll']))
        prefixes.append(old['text'].startswith(new['text']))
        if new['correct'] and not old['correct']:
            gained.append(new['id'])
        if old['correct'] and not new['correct']:
            lost.append(old['id'])
    return {'historical': str(historical.resolve()), 'corrected': str(corrected.resolve()),
            'scope': 'paired_common128' if count == 128 else 'paired_live_prefix_not_final128',
            'count': count, 'ids_sha256': digest(ids), 'configuration_differences': differences,
            'checked_input_provenance_fields': list(INPUT_PROVENANCE),
            'old_config_sha256': digest(old_config), 'corrected_config_sha256': digest(new_config),
            'old_answers_sha256': sha256(old_bytes).hexdigest(),
            'corrected_answers_snapshot_sha256': sha256(new_bytes).hexdigest(),
            'corrected_audited_records_sha256': audit['audited_records_sha256'],
            'old_correct': sum(r['correct'] for r in old_rows),
            'corrected_correct': sum(r['correct'] for r in new_rows),
            'old_truncated': sum(r['truncated'] for r in old_rows),
            'corrected_truncated': sum(r['truncated'] for r in new_rows),
            'old_strict_marker_correct_ignoring_truncation': old_marked,
            'corrected_strict_marker_correct_ignoring_truncation': new_marked,
            'decoded_prefix_matches': sum(prefixes),
            'decoded_prefix_mismatch_ids': [i for i, match in zip(ids, prefixes) if not match],
            'max_answer_nll_abs_difference': max(differences_nll),
            'gained_correct_ids': gained, 'lost_correct_ids': lost}


def main():
    from sparsifier.eval.gsm8k import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('historical', 'corrected', 'corrected-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    args = parser.parse_args()
    result = compare(args.historical, args.corrected, args.corrected_audit, count=args.count)
    result['comparator_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_ids')}))


if __name__ == '__main__':
    main()
