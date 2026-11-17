"""Versioned EOS correction for frozen-checkpoint GSM8K evaluations.

Keep production pruning/resume sources unchanged. Use fresh output folders and
a matching corrected dense reference. Preserve raw generation IDs alongside
the unchanged strict answer parser and decision screens.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

PROTOCOL = 'gsm8k-model-and-tokenizer-eos-v1'
MODULES = ('sparsifier.eval.gsm8k_decision', 'sparsifier.eval.gsm8k_decision_full')


def stopping_policy():
    from sparsifier.prune.whole_neuron_io import file_hash
    import sparsifier.data.gsm8k as parser
    import sparsifier.eval.gsm8k_pruning as evaluator
    return {'version': PROTOCOL, 'eos_token_ids': [248044, 248046],
            'original_model_eos_ids': [248044], 'tokenizer_eos_id': 248046,
            'parser': 'unchanged_strict_numeric_marker_and_budget_truncation_rejection',
            'adapter_sha256': file_hash(Path(__file__)),
            'parser_sha256': file_hash(Path(parser.__file__)),
            'behavior_sha256': file_hash(Path(evaluator.__file__))}


def corrected_behavior(original, model, tokenizer, rows, output, config_hash,
                       *, raw_output, **kwargs):
    from sparsifier.data.gsm8k import digest, write_immutable
    from sparsifier.eval.gsm8k import render_prompt
    expected = stopping_policy()
    previous_config = model.generation_config
    ids = previous_config.eos_token_id
    ids = ids if isinstance(ids, list) else [ids]
    if ids != expected['original_model_eos_ids'] or tokenizer.eos_token_id != expected['tokenizer_eos_id']:
        raise ValueError('pinned model/tokenizer EOS changed; inspect before reusing policy')
    configured = deepcopy(previous_config)
    configured.eos_token_id = expected['eos_token_ids']
    original_generate = model.generate
    lookup = {tuple(tokenizer.encode(render_prompt(tokenizer, row['question']),
                                    add_special_tokens=False)): row['id'] for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate tokenized prompts in evaluation batch')

    def generate_with_raw_ids(**generation_kwargs):
        result = original_generate(**generation_kwargs)
        inputs = generation_kwargs['input_ids'].tolist()
        masks = generation_kwargs['attention_mask'].tolist()
        batch_ids = [lookup[tuple(t for t, m in zip(seq, mask) if m)]
                     for seq, mask in zip(inputs, masks)]
        width = generation_kwargs['input_ids'].shape[1]
        record = {'config_sha256': config_hash, 'batch_ids': batch_ids,
                  'input_ids_sha256': digest(inputs), 'eos_token_ids': configured.eos_token_id,
                  'max_new_tokens': generation_kwargs['max_new_tokens'],
                  'generated_token_ids': result[:, width:].tolist()}
        write_immutable(Path(raw_output)/(digest(batch_ids)+'.json'), record)
        return result

    try:
        model.generation_config = configured
        model.generate = generate_with_raw_ids
        return original(model, tokenizer, rows, output, config_hash, **kwargs)
    finally:
        model.generate = original_generate
        model.generation_config = previous_config


def install(runner, output):
    """Tag config before hashing; reject historical or differently tagged dense scores."""
    original_write, original_behavior = runner.write_immutable, runner.behavior
    original_dense = runner.dense_answers
    policy = stopping_policy()

    def write(path, value):
        if Path(path) == Path(output)/'config.json':
            value['stopping_policy'] = policy
        return original_write(path, value)

    def behavior(model, tokenizer, rows, destination, config_hash, **kwargs):
        return corrected_behavior(original_behavior, model, tokenizer, rows,
            destination, config_hash, raw_output=Path(output)/'raw-generations', **kwargs)

    def dense(reference, *args, **kwargs):
        config = json.loads((Path(reference)/'config.json').read_text())
        if config.get('stopping_policy') != policy:
            raise ValueError('matching corrected EOS dense reference required')
        return original_dense(reference, *args, **kwargs)

    runner.write_immutable, runner.behavior, runner.dense_answers = write, behavior, dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, choices=(8, 16, 32), default=32)
    parser.add_argument('module', choices=MODULES)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from sparsifier.eval.gsm8k_batch_runtime import configure_batch_size
    configure_batch_size(args.batch_size)
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument('--output', type=Path, required=True)
    selected, _ = check.parse_known_args(args.arguments)
    runner = importlib.import_module(args.module)
    install(runner, selected.output)
    sys.argv = [args.module, *args.arguments]
    runner.main()


if __name__ == '__main__':
    main()
