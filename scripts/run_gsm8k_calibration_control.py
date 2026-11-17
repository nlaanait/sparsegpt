"""Versioned calibration ablation using the unchanged sequential pruning loop.

Launch only through guarded gflow. Only calibration collection changes; the
existing loop still owns model loading, solver settings, BF16 casting, chained
stage checkpoints, and behavior. Its immutable config includes this adapter
and its data hashes before its config hash or any stage is created.
"""
from __future__ import annotations

import argparse
from functools import wraps
from pathlib import Path
import sys

import numpy as np
import torch

from sparsifier.data.gsm8k import digest, load_manifest
from sparsifier.eval.gsm8k_batching import batches, pack
from sparsifier.prune.whole_neuron_io import file_hash


def validate_bundle(bundle, rows):
    if bundle.get('protocol') != 'gsm8k-reasoning-calibration-control-v1':
        raise ValueError('unexpected reasoning-control protocol')
    if [r['id'] for r in bundle['rows']] != [r['id'] for r in rows]:
        raise ValueError('calibration parents/order differ from the frozen control')
    for actual, expected in zip(bundle['rows'],rows):
        if (actual['source_sha256'] != expected['source_sha256']
                or actual['split'] != expected['split']
                or actual['input_ids_sha256'] != digest(actual['input_ids'])
                or actual['response_start'] != actual['prompt_tokens']
                or actual['prompt_readout_position'] != actual['prompt_tokens']-1
                or actual['input_ids'][-1] != bundle['eos_token_id']):
            raise ValueError('reasoning token/source provenance mismatch')


def capture_control(model, layer, inputs, rows, mode):
    """Extract prompt readouts and unpadded prompt/response input pools."""
    records = []
    def hook(_module,args,output):
        x, y = args[0], output
        if x.shape[:2] != inputs['input_ids'].shape or y.shape[:2] != x.shape[:2]:
            raise ValueError('calibration token axes do not align')
        for n,row in enumerate(rows):
            length = len(row['input_ids']) if mode=='reasoning' else row['prompt_tokens']
            offset = x.shape[1]-length
            readout = row['prompt_tokens']-1
            # Keep every pre-EOS endpoint explicitly. The EOS input itself
            # predicts the next turn and is not an answer-generation state.
            start,stop = ((row['response_start'],length-2) if mode=='reasoning' else (0,readout))
            records.append({'x':x[n,offset+readout].detach().float().cpu().numpy(),
                'y':y[n,offset+readout].detach().float().cpu().numpy(),
                'terminal':x[n,offset+length-2].detach().float().cpu().numpy() if mode=='reasoning' else None,
                'pool':x[n,offset+start:offset+stop].detach().float().cpu().numpy()})
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs,use_cache=False,logits_to_keep=1)
    finally:
        handle.remove()
    if len(records) != len(rows):
        raise ValueError('selected layer was not called exactly once')
    return records


def collect_control(model,tokenizer,layer,rows,c4_tokens,seed,*,bundle,mode,task_tokens,progress=None):
    from sparsifier.eval.gsm8k_capture import capture_batch
    validate_bundle(bundle,rows)
    device = next(model.parameters()).device
    xs,ys,pools,terminals = [],[],[],[]
    for group in batches(bundle['rows']):
        sequences = [r['input_ids'] if mode=='reasoning' else r['input_ids'][:r['prompt_tokens']]
                     for r in group]
        captured = capture_control(model,layer,pack(sequences,tokenizer,device),group,mode)
        for record in captured:
            xs.append(record['x']);ys.append(record['y']);pools.append(record['pool'])
            if record['terminal'] is not None:
                terminals.append(record['terminal'])
        if progress is not None:
            progress({'event':'calibration_batch','phase':f'{mode}_control',
                      'examples_done':len(xs),'examples_total':len(rows)})
    x,y,pool = np.stack(xs),np.stack(ys),np.concatenate(pools)
    forced = np.concatenate((x,np.stack(terminals))) if terminals else x
    needed = task_tokens-len(forced)
    if needed<0 or needed>len(pool):
        raise ValueError('insufficient distinct tokens for the requested task budget')
    chosen = np.random.default_rng(seed).choice(len(pool),needed,replace=False)
    math = np.concatenate((forced,pool[chosen]))
    del pool,pools
    c4_arrays = []
    for group in batches(c4_tokens):
        captured = capture_batch(model,layer,pack(group,tokenizer,device),list(map(len,group)),
                                 [seed]*len(group),token_samples=len(group[0]))
        c4_arrays.extend(r['token_x'] for r in captured)
    c4 = np.concatenate(c4_arrays)
    if progress is not None:
        progress({'event':'calibration_completed','readout_samples':len(x),
            'ambient_tokens':len(c4)+len(math),'c4_tokens':len(c4),'task_tokens':len(math),
            'task_token_population':mode,'input_features':x.shape[1],'output_features':y.shape[1]})
    return x,y,torch.from_numpy(np.concatenate((c4,math))),torch.from_numpy(c4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reasoning-bundle',type=Path,required=True)
    parser.add_argument('--mode',choices=('legacy','prompt','reasoning'),required=True)
    parser.add_argument('--task-tokens',type=int,choices=(4096,32768),required=True)
    parser.add_argument('sequential_arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.sequential_arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    if args.mode=='legacy' and args.task_tokens!=4096:
        parser.error('legacy mode must retain the unchanged 4096 task-token budget')
    from sparsifier.eval import gsm8k_sequential as runner
    from sparsifier.eval.gsm8k_batching import policy
    from sparsifier.eval.gsm8k import MODEL,REVISION
    if policy()['batch_size'] != 32:
        raise ValueError('use the default batch32 process policy for this controlled ablation')
    bundle = load_manifest(args.reasoning_bundle)
    if bundle['model'] != MODEL or bundle['revision'] != REVISION:
        raise ValueError('reasoning-control checkpoint mismatch')
    control = {'version':'gsm8k-calibration-frontier-v1','mode':args.mode,'task_tokens':args.task_tokens,
        'c4':'unchanged_frozen_32_by_128_token_windows','geometry':'unchanged_final_prompt_readout',
        'reasoning_sampling':'all_prompt_readouts_and_pre_EOS_endpoints_then_uniform_other_response_tokens_no_EOS_inputs',
        'bundle_sha256':file_hash(args.reasoning_bundle),'bundle_manifest_sha256':bundle['manifest_sha256'],
        'source_sha256':file_hash(Path(__file__))}
    original_collect,original_write = runner.collect,runner.write_immutable
    @wraps(original_collect)
    def collect(*positional,**keywords):
        rows = positional[3]
        validate_bundle(bundle,rows)
        if args.mode=='legacy':
            return original_collect(*positional,**keywords)
        return collect_control(*positional,**keywords,bundle=bundle,mode=args.mode,task_tokens=args.task_tokens)
    @wraps(original_write)
    def write_config(path,config):
        if Path(path).name!='config.json' or config.get('purpose')!='development_sequential_MLP_retention':
            raise ValueError('unexpected sequential config write')
        if config['manifest_sha256'] != bundle['calibration_manifest_sha256']:
            raise ValueError('reasoning-control parent manifest mismatch')
        config['calibration_control'] = control
        return original_write(path,config)
    runner.collect,runner.write_immutable = collect,write_config
    try:
        sys.argv = ['gsm8k_sequential',*arguments]
        runner.main()
    finally:
        runner.collect,runner.write_immutable = original_collect,original_write


if __name__ == '__main__':
    main()
