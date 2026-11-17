"""Prepare training-only reasoning tokens for the pruning artifact control.

Only the existing geometry-fit and pruning-calibration parents are admitted.
No model is loaded and no evaluation answers are used for calibration.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import re

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable
from sparsifier.data.gsm8k_calibration import validate_calibration
from sparsifier.eval.gsm8k import MODEL, REVISION, render_prompt
from sparsifier.prune.whole_neuron_io import file_hash


def prepare_bundle(dataset, calibration, evaluation, tokenizer, source_sha256):
    validate_calibration(calibration, [evaluation])
    if source_sha256 != calibration['source_file_sha256']:
        raise ValueError('training Arrow source hash mismatch')
    if tokenizer.eos_token_id is None:
        raise ValueError('explicit answer-ending EOS required')
    rows = []
    for row in calibration['rows']:
        if row['split'] not in ('geometry_fit','pruning_calibration'):
            continue
        source = dataset[row['source_index']]
        if digest(source) != row['source_sha256'] or source['question'] != row['question']:
            raise ValueError('training parent content mismatch')
        # Strip GSM8K calculator annotations while retaining their numeric result.
        response = re.sub(r'<<[^<>]*>>', '', source['answer']).replace('####', 'Final answer:')
        if parse_answer(response) != Fraction(row['answer']):
            raise ValueError('cleaned training solution changed its final answer')
        prefix = tokenizer.encode(render_prompt(tokenizer,row['question']),add_special_tokens=False)
        continuation = tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]
        if not prefix or len(continuation)<2 or len(prefix)+len(continuation)>2048:
            raise ValueError('calibration solution exceeds the context budget; do not silently truncate')
        rows.append({**row,'input_ids':prefix+continuation,'prompt_tokens':len(prefix),
            'prompt_readout_position':len(prefix)-1,'response_start':len(prefix),
            'response_tokens':len(continuation),'response_sha256':digest(response),
            'input_ids_sha256':digest(prefix+continuation)})
    bundle = {'schema':1,'protocol':'gsm8k-reasoning-calibration-control-v1',
        'purpose':'training_reasoning_and_EOS_calibration_control_not_evaluation',
        'source_split':'train','model':MODEL,'revision':REVISION,
        'calibration_manifest_sha256':calibration['manifest_sha256'],
        'reserved_evaluation_manifest_sha256':evaluation['manifest_sha256'],
        'source_file_sha256':source_sha256,'eos_token_id':tokenizer.eos_token_id,
        'chat_template_sha256':digest(tokenizer.chat_template),
        'response_format':'original_training_solution_without_calculator_annotations_Final_answer_marker_then_EOS',
        'rows':rows,'response_token_count':sum(r['response_tokens'] for r in rows),
        'total_token_count':sum(len(r['input_ids']) for r in rows),
        'max_context_tokens':max(len(r['input_ids']) for r in rows)}
    bundle['manifest_sha256'] = digest(bundle)
    return bundle


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-arrow',type=Path,required=True)
    parser.add_argument('--calibration-manifest',type=Path,required=True)
    parser.add_argument('--evaluation-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.train_arrow.name != 'gsm8k-train.arrow':
        parser.error('only the cached GSM8K training Arrow source is accepted')
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    bundle = prepare_bundle(Dataset.from_file(str(args.train_arrow)),
        load_manifest(args.calibration_manifest),load_manifest(args.evaluation_manifest),
        tokenizer,file_hash(args.train_arrow))
    bundle['preparation_source_sha256'] = file_hash(Path(__file__))
    del bundle['manifest_sha256']
    bundle['manifest_sha256'] = digest(bundle)
    write_immutable(args.output,bundle)
    print({key:bundle[key] for key in ('response_token_count','total_token_count','max_context_tokens')},flush=True)


if __name__ == '__main__':
    main()
