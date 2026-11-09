"""Evaluate dense or frozen sequential candidates on original GSM8K validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
from sparsifier.eval.gsm8k_pruning import behavior
from sparsifier.eval.gsm8k_sequential import verify_stage
from sparsifier.eval.memory_guard import configure_cuda, host_memory


def frozen_stages(source):
    config = json.loads((source/'config.json').read_text())
    report = json.loads((source/'report.json').read_text())
    if not report.get('complete') or report['config_sha256'] != digest(config):
        raise ValueError('sequential source incomplete or mismatched')
    if config['model'] != MODEL or config['revision'] != REVISION:
        raise ValueError('source checkpoint mismatch')
    upstream = digest(config)
    for n, row in enumerate(report['stages']):
        path = source/f'site-{n:03d}.pt'
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != row['candidate_sha256']:
            raise ValueError('candidate content hash mismatch')
        stored = torch.load(path,weights_only=True,map_location='cpu')
        verify_stage(stored,digest(config),upstream,tuple(row['site']))
        upstream = checksum
    return config,report


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.data.loaders import get_loaders
    from sparsifier.eval.natural_layer import vocabulary_reference, language_metrics
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source',type=Path,help='completed sequential run; omit for dense')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    args = p.parse_args()
    if args.max_new_tokens < 1: p.error('positive generation budget required')
    manifest = load_manifest(args.manifest)
    rows = manifest['rows']
    if manifest['source_split'] != 'train' or not rows or any(r['split'] != 'validation' for r in rows):
        raise ValueError('original evaluation accepts training validation only')
    if any(not r['id'].startswith('gsm8k/train/') for r in rows):
        raise ValueError('original GSM8K questions required')
    source_config,source_report = frozen_stages(args.source) if args.source else (None,None)
    config = {'model':MODEL,'revision':REVISION,'arm':source_config['arm'] if source_config else 'dense',
              'manifest_sha256':manifest['manifest_sha256'],'evaluation_split':'validation',
              'evaluation_ids':[r['id'] for r in rows],'max_new_tokens':args.max_new_tokens,
              'source_config_sha256':digest(source_config) if source_config else None,
              'candidate_chain':[r['candidate_sha256'] for r in source_report['stages']] if source_report else [],
              'purpose':'original_GSM8K_validation_not_test',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    write_immutable(args.output/'config.json',config)
    config_hash = digest(config); path = args.output/'report.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['config_sha256'] != config_hash: raise ValueError('report mismatch')
        if old.get('complete'): return
    torch.set_num_threads(8); torch.manual_seed(0)
    memory = configure_cuda(56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,use_kernels=False,attn_implementation='sdpa').eval().to('cuda')
    model.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    _,wiki = get_loaders('wikitext2',1,0,128,MODEL)
    windows = [wiki.input_ids[:,i*128:(i+1)*128] for i in range(8)]
    reference = vocabulary_reference(model,windows,'cuda')
    if not np.isfinite([r['nll'] for r in reference]).all() or np.mean([r['nll'] for r in reference]) > np.log(100):
        raise ValueError('implausible dense language loss')
    removed = eligible = 0
    if source_report:
        for n,row in enumerate(source_report['stages']):
            stored = torch.load(args.source/f'site-{n:03d}.pt',weights_only=True,map_location='cpu')
            b,module = stored['site']; layer = model.model.layers[b].get_submodule(module)
            if layer.weight.shape != stored['weight'].shape: raise ValueError('shape mismatch')
            with torch.no_grad(): layer.weight.copy_(stored['weight'])
            removed += stored['removed']; eligible += layer.weight.numel()
            del stored
        if total != source_report['total_parameters']: raise ValueError('parameter count mismatch')
    language = language_metrics(model,windows,reference,'cuda')
    evaluated = behavior(model,tokenizer,rows,args.output,config_hash,max_new_tokens=args.max_new_tokens)
    atomic_json(path,{'complete':True,'config_sha256':config_hash,'behavior':evaluated,
        'language':language,'removed_weights':removed,'total_parameters':total,
        'eligible_parameters':eligible,'whole_model_removed_fraction':removed/total,
        'memory':memory,'host_after':host_memory(),
        'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})


if __name__ == '__main__': main()
