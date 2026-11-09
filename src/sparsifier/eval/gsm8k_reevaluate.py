"""Reevaluate frozen single-layer candidates under a common development budget."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_pruning import behavior
from sparsifier.eval.memory_guard import configure_cuda, host_memory


def validate_source(config, report, manifest, capture):
    if not report.get('complete') or report['config_sha256'] != digest(config):
        raise ValueError('source report incomplete or mismatched')
    if config['manifest_sha256'] != manifest['manifest_sha256'] or config['capture_sha256'] != digest(capture):
        raise ValueError('source capture/manifest mismatch')
    rows = {r['id']: r for r in manifest['rows']}
    selected = [rows[i] for i in config['evaluation_ids']]
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('empty or duplicate evaluation IDs')
    if any(r['split'] != 'validation' for r in selected):
        raise ValueError('budget study restricted to validation')
    return selected


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval.natural_layer import replaced_weight
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error('positive token budget required')
    source = json.loads((args.source/'config.json').read_text())
    source_report = json.loads((args.source/'report.json').read_text())
    capture = json.loads((args.capture/'config.json').read_text())
    manifest = load_manifest(args.manifest)
    rows = validate_source(source, source_report, manifest, capture)
    candidate_path = args.source/'candidate.pt'
    stored = torch.load(candidate_path, weights_only=True, map_location='cpu')
    if stored['config_sha256'] != digest(source):
        raise ValueError('candidate provenance mismatch')
    candidate = stored['weight']
    weight = torch.from_numpy(np.load(args.capture/'weight.npy', allow_pickle=False))
    if candidate.shape != weight.shape or not torch.isfinite(candidate).all():
        raise ValueError('invalid candidate')
    config = {**source, 'purpose': 'development_generation_budget_sensitivity',
              'max_new_tokens': args.max_new_tokens, 'source_config_sha256': digest(source),
              'candidate_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/'config.json', config)
    config_hash = digest(config)
    report_path = args.output/'report.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved['config_sha256'] != config_hash:
            raise ValueError('report provenance mismatch')
        if saved.get('complete'):
            print('Already complete', flush=True)
            return
    torch.set_num_threads(8)
    torch.manual_seed(0)
    memory = configure_cuda(56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(source['model'], revision=source['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(source['model'], revision=source['revision'],
        local_files_only=True, dtype=torch.bfloat16, use_kernels=False,
        attn_implementation='sdpa').eval().to('cuda')
    model.requires_grad_(False)
    layer = model.model.layers[capture['block']].get_submodule(capture['module'])
    if not torch.equal(layer.weight.detach().float().cpu(), weight):
        raise ValueError('dense checkpoint does not match capture')
    total_parameters = sum(t.numel() for t in model.parameters())
    # Count actual removed nonzero entries, not preexisting dense zeros.
    removed = int(((weight != 0) & (candidate == 0)).sum())
    with replaced_weight(layer, candidate):
        evaluated = behavior(model, tokenizer, rows, args.output, config_hash,
                             max_new_tokens=args.max_new_tokens)
    atomic_json(report_path, {'complete': True, 'config_sha256': config_hash,
        'behavior': evaluated, 'removed_weights': removed, 'total_parameters': total_parameters,
        'whole_model_removed_fraction': removed/total_parameters,
        'eligible_parameters': weight.numel(), 'eligible_removed_fraction': removed/weight.numel(),
        'memory': memory, 'host_after': host_memory(),
        'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved()/1024**3})


if __name__ == '__main__':
    main()
