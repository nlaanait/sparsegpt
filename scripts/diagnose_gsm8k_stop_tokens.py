"""Compare model-only and tokenizer-inclusive EOS on a frozen GSM8K batch.

Run through guarded gflow. Preserve raw IDs, strict parsing, the original
batch membership, and both policies; do not overwrite historical evaluations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

import torch

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_batching import pack, pad_id, reference_backend, trim_generated
from sparsifier.eval.gsm8k_whole_neuron import package_versions
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.whole_neuron_io import file_hash
from scripts.diagnose_gsm8k_pruning_frontier import load_source


def summarize(tokens, tokenizer, eos, budget, expected):
    generated = trim_generated(tokens, set(eos))
    truncated = len(generated) >= budget and generated[-1] not in eos
    text = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_answer(text, truncated=truncated)
    return {'generated_tokens': len(generated), 'truncated': truncated, 'text': text,
            'parsed_answer': str(parsed) if parsed is not None else None,
            'correct': parsed is not None and parsed == Fraction(expected)}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'cpu-audit', 'original-evaluation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('scheduler-provided CUDA_VISIBLE_DEVICES required')
    manifest = load_manifest(args.manifest)
    original = json.loads((args.original_evaluation/'config.json').read_text())
    old_records = [json.loads(line) for line in
                   (args.original_evaluation/'answers.jsonl').read_text().splitlines()]
    rows = manifest['rows'][:32]
    ids = [r['id'] for r in rows]
    assert ids == original['evaluation_ids'][:32]
    assert [r['id'] for r in old_records[:32]] == ids
    assert old_records[0]['batch_ids'] == ids
    assert original['batching']['batch_size'] == 32
    assert original['max_new_tokens'] == 2048
    source_config = json.loads((args.source/'config.json').read_text())
    assert original['source_config_sha256'] == digest(source_config)
    assert original['source_report_sha256'] == file_hash(args.source/'report.json')
    audit = json.loads(args.cpu_audit.read_text())
    audited = next(s for s in audit['sources'] if s['source'] == str(args.source.resolve()))
    config = {'purpose': 'frozen_checkpoint_stop_token_diagnostic_not_retention',
              'model': MODEL, 'revision': REVISION, 'ids': ids, 'budget': 2048,
              'source': str(args.source.resolve()), 'source_config_sha256': digest(source_config),
              'source_report_sha256': file_hash(args.source/'report.json'),
              'cpu_audit_sha256': file_hash(args.cpu_audit),
              'manifest_sha256': manifest['manifest_sha256'],
              'original_config_sha256': digest(original),
              'original_answers_sha256': file_hash(args.original_evaluation/'answers.jsonl'),
              'script_sha256': file_hash(Path(__file__)),
              'loader_sha256': file_hash(Path(__file__).with_name('diagnose_gsm8k_pruning_frontier.py')),
              'packages': package_versions(), 'torch_seed': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/'config.json', config)
    path = args.output/'report.json'
    result = json.loads(path.read_text()) if path.exists() else {
        'config_sha256': digest(config), 'complete': False, 'policies': []}
    assert result['config_sha256'] == digest(config)
    if result['complete']:
        return
    torch.set_num_threads(8)
    torch.manual_seed(0)
    result['memory'] = configure_cuda(48.)
    result['backend'] = reference_backend()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, dtype=torch.bfloat16, use_kernels=False,
        attn_implementation='sdpa').eval().to('cuda').requires_grad_(False)
    load_source(model, args.source, audited)
    configured = model.generation_config.eos_token_id
    eos = configured if isinstance(configured, list) else [configured]
    assert eos and None not in eos and tokenizer.eos_token_id is not None
    union = sorted(set(eos+[tokenizer.eos_token_id]))
    result['model_eos_ids'] = eos
    result['tokenizer_eos_id'] = tokenizer.eos_token_id
    result['union_eos_ids'] = union
    inputs = pack([tokenizer.encode(render_prompt(tokenizer, r['question']),
                                   add_special_tokens=False) for r in rows], tokenizer, 'cuda')
    for name, stops in [('model_only', eos), ('model_plus_tokenizer', union)]:
        if any(p['name'] == name for p in result['policies']):
            continue
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=2048,
                use_cache=True, pad_token_id=pad_id(tokenizer), eos_token_id=stops)
        tokens = output[:, inputs['input_ids'].shape[1]:].tolist()
        records = []
        for row, raw, old in zip(rows, tokens, old_records[:32]):
            active = summarize(raw, tokenizer, stops, 2048, row['answer'])
            alternative = summarize(raw, tokenizer, union, 2048, row['answer'])
            records.append({'id': row['id'], 'tokens': raw,
                'stop_token_positions': {str(t): [i for i, x in enumerate(raw) if x == t]
                                        for t in union},
                'active_policy': active, 'trim_same_tokens_with_union': alternative,
                'historical_text_match': active['text'] == old['text'],
                'historical_correct': old['correct'], 'historical_truncated': old['truncated']})
        result['policies'].append({'name': name, 'eos_ids': stops, 'records': records,
            'correct': sum(r['active_policy']['correct'] for r in records),
            'truncated': sum(r['active_policy']['truncated'] for r in records),
            'union_trim_correct': sum(r['trim_same_tokens_with_union']['correct'] for r in records)})
        atomic_json(path, result)
        print(json.dumps({'policy_complete': name, 'correct': result['policies'][-1]['correct'],
                          'truncated': result['policies'][-1]['truncated']}), flush=True)
    result['complete'] = True
    atomic_json(path, result)


if __name__ == '__main__':
    main()
