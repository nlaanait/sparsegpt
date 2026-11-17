"""Bounded frozen-checkpoint diagnostic; launch only through guarded gflow.

Does not prune or rewrite experiment outputs. Compares matched teacher-forced
reasoning, short generation, and cached versus uncached continuation logits.
This is an engineering diagnostic, not a new GSM8K retention evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_batching import pack, pad_id, reference_backend, trim_generated
from sparsifier.eval.gsm8k_whole_neuron import package_versions
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.whole_neuron_io import file_hash


@torch.inference_mode()
def teacher_forcing(model, tokenizer, rows, reference):
    result = []
    device = next(model.parameters()).device
    for row in rows:
        prefix = tokenizer.encode(render_prompt(tokenizer, row['question']), add_special_tokens=False)
        # Score the same dense-generated reasoning continuation in every state.
        target = tokenizer.encode(reference[row['id']]['text'], add_special_tokens=False)[:256]
        assert target
        tokens = torch.tensor([prefix+target], device=device)
        logits = model(input_ids=tokens, use_cache=False, logits_to_keep=len(target)+1).logits
        loss = F.cross_entropy(logits[0, -len(target)-1:-1].float(),
                               torch.tensor(target, device=device))
        result.append({'id':row['id'], 'target_tokens':len(target), 'nll':float(loss),
                       'target_ids_sha256':digest(target)})
    return result


@torch.inference_mode()
def cache_comparison(model, tokenizer, row, reference):
    device = next(model.parameters()).device
    prefix = tokenizer.encode(render_prompt(tokenizer, row['question']), add_special_tokens=False)
    continuation = tokenizer.encode(reference[row['id']]['text'], add_special_tokens=False)[:64]
    tokens = torch.tensor([prefix+continuation], device=device)
    # Compare the same fixed prefix, never different freely generated histories.
    uncached = model(input_ids=tokens, use_cache=False, logits_to_keep=len(continuation)+1).logits.float()
    prefill = model(input_ids=tokens[:, :len(prefix)], use_cache=True, logits_to_keep=1)
    cache = prefill.past_key_values
    steps = [prefill.logits[:, -1].float()]
    for token in continuation:
        out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                    use_cache=True, logits_to_keep=1,
                    attention_mask=torch.ones((1, len(prefix)+len(steps)), device=device, dtype=torch.long))
        cache = out.past_key_values
        steps.append(out.logits[:, -1].float())
    cached = torch.stack(steps, dim=1)
    delta = cached-uncached
    logp, logq = uncached.log_softmax(-1), cached.log_softmax(-1)
    return {'id':row['id'], 'positions':len(steps), 'max_logit_difference':float(delta.abs().max()),
        'rms_logit_difference':float(delta.square().mean().sqrt()),
        'next_token_agreement':float((cached.argmax(-1)==uncached.argmax(-1)).float().mean()),
        'mean_vocabulary_kl':float((logp.exp()*(logp-logq)).sum(-1).mean())}


@torch.inference_mode()
def generations(model, tokenizer, rows, *, cached, batch_size, budget):
    records = []
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for start in range(0, len(rows), batch_size):
        group = rows[start:start+batch_size]
        sequences = [tokenizer.encode(render_prompt(tokenizer, row['question']),
                     add_special_tokens=False) for row in group]
        inputs = pack(sequences, tokenizer, device)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=budget,
                                use_cache=cached, pad_token_id=pad_id(tokenizer))
        for row, tokens in zip(group, output[:, inputs['input_ids'].shape[1]:].tolist()):
            tokens = trim_generated(tokens, eos)
            grams = Counter(tuple(tokens[i:i+8]) for i in range(max(0, len(tokens)-7)))
            records.append({'id':row['id'], 'cached':cached, 'batch_size':batch_size,
                'tokens':tokens, 'text':tokenizer.decode(tokens, skip_special_tokens=True),
                'truncated':len(tokens)>=budget and tokens[-1] not in eos,
                'most_repeated_8gram_count':max(grams.values(), default=0)})
    return records


def load_source(model, source, expected_audit):
    config = json.loads((source/'config.json').read_text())
    report = json.loads((source/'report.json').read_text())
    assert config['revision'] == REVISION and config['model'] == MODEL
    assert report['complete'] and report['config_sha256'] == digest(config)
    assert expected_audit['report_sha256'] == file_hash(source/'report.json')
    assert expected_audit['config_sha256'] == digest(config)
    assert len(report['stages']) == len(expected_audit['stages']) == 96
    upstream = digest(config)
    for n, stage in enumerate(report['stages']):
        path = source/f'site-{n:03d}.pt'
        checksum = file_hash(path)
        assert checksum == stage['candidate_sha256'] == expected_audit['stages'][n]['sha256']
        saved = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        assert saved['upstream'] == upstream and saved['config_sha256'] == digest(config)
        assert saved['site'] == stage['site']
        b, module = saved['site']
        target = model.model.layers[b].get_submodule(module)
        with torch.no_grad():
            target.weight.copy_(saved['weight'])
        upstream = checksum


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dense-answers', type=Path, required=True)
    parser.add_argument('--cpu-audit', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('scheduler-provided CUDA_VISIBLE_DEVICES required')
    audit = json.loads(args.cpu_audit.read_text())
    sources = {s['source']:s for s in audit['sources']}
    assert all(str(p.resolve()) in sources for p in args.source)
    rows = load_manifest(args.manifest)['rows'][:2]
    reference = {r['id']:r for r in map(json.loads, args.dense_answers.read_text().splitlines())}
    assert all(row['id'] in reference and reference[row['id']]['correct'] for row in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'purpose':'frozen_pruning_artifact_diagnostic_not_retention', 'model':MODEL,
        'revision':REVISION, 'ids':[r['id'] for r in rows], 'budget':128,
        'source_paths':[str(p.resolve()) for p in args.source],
        'cpu_audit_sha256':file_hash(args.cpu_audit), 'source_sha256':file_hash(Path(__file__)),
        'dense_answers_sha256':file_hash(args.dense_answers), 'packages':package_versions()}
    write_immutable(args.output/'config.json', config)
    report_path = args.output/'report.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('complete'):
        return
    torch.set_num_threads(8)
    torch.manual_seed(0)
    memory = configure_cuda(48.)
    backend = reference_backend()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation='sdpa').eval().to('cuda').requires_grad_(False)
    result = json.loads(report_path.read_text()) if report_path.exists() else {
        'config_sha256':digest(config), 'complete':False, 'memory':memory, 'backend':backend, 'states':[]}
    assert result['config_sha256'] == digest(config)
    for source in [None, *args.source]:
        label = source.name if source else 'dense'
        if any(s['name'] == label for s in result['states']):
            continue
        if source:
            load_source(model, source, sources[str(source.resolve())])
        state = {'name':label, 'teacher_forcing':teacher_forcing(model, tokenizer, rows, reference),
                 'cache_comparison':cache_comparison(model, tokenizer, rows[0], reference)}
        # Fixed small budget exposes early repetition; it cannot estimate accuracy.
        state['generations'] = []
        for cached, size in [(True, 1), (True, 2), (False, 1)]:
            state['generations'].extend(generations(model, tokenizer, rows,
                cached=cached, batch_size=size, budget=128))
        result['states'].append(state)
        atomic_json(report_path, result)
        print(json.dumps({'state_complete':label, 'teacher_forcing':state['teacher_forcing'],
                          'cache_comparison':state['cache_comparison']}), flush=True)
    result['complete'] = True
    atomic_json(report_path, result)


if __name__ == '__main__':
    main()
