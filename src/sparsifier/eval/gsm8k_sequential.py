"""Sequential MLP pruning with fresh task/C4 activations at every site.

Development-only. A site checkpoint binds its weights to the complete chain
of upstream candidates. All GPU execution must use the guarded gflow worker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_capture import capture_one, check_competence
from sparsifier.eval.gsm8k_pruning import pair_inputs, prune_candidate, select_validation, behavior
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.prune.factor_atlas import FactorAtlas, projector_diagonals
from sparsifier.prune.gram_dictionary import GramBlockDictionary
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION


def sites_for(blocks, modules, layer_count):
    if not blocks or blocks != sorted(set(blocks)) or any(b < 0 or b >= layer_count for b in blocks):
        raise ValueError('blocks must be unique, increasing, and in range')
    allowed = ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    if not modules or len(set(modules)) != len(modules) or any(m not in allowed for m in modules):
        raise ValueError('unsupported or duplicate MLP modules')
    return [(b, m) for b in blocks for m in allowed if m in modules]


def verify_stage(stored, config_hash, upstream, site):
    if stored['config_sha256'] != config_hash or stored['upstream'] != upstream or stored['site'] != list(site):
        raise ValueError('sequential candidate provenance mismatch')
    if not torch.isfinite(stored['weight']).all():
        raise ValueError('nonfinite candidate')
    diagnostics = stored.get('diagnostics', {})
    if (diagnostics.get('solver') == 'row_diagonal_congruence_approximation'
            and diagnostics.get('normalization') != LOCAL_PRUNING_VERSION):
        raise ValueError('legacy local candidate must be regenerated with corrected pruning')


def collect(model, tokenizer, layer, rows, c4_tokens, seed):
    xs, ys, pool = [], [], []
    device = next(model.parameters()).device
    for n, row in enumerate(rows):
        ids = tokenizer.encode(render_prompt(tokenizer, row['question']), add_special_tokens=False)
        if len(ids) > 2048:
            raise ValueError('calibration prompt exceeds budget')
        arrays = capture_one(model, layer, torch.tensor([ids], device=device), seed=seed+n)
        xs.append(arrays['x']); ys.append(arrays['y'])
        pool.append(arrays['token_x'][arrays['positions'] != int(arrays['readout_position'])])
    x, y, pool = np.stack(xs), np.stack(ys), np.concatenate(pool)
    needed = 4096-len(x)
    if needed < 0 or len(pool) < needed:
        raise ValueError('invalid math token budget')
    ix = np.random.default_rng(seed).choice(len(pool), needed, replace=False)
    math = np.concatenate((x, pool[ix]))
    c4 = np.concatenate([capture_one(model, layer, torch.as_tensor(t[None], device=device),
                                    token_samples=len(t), seed=seed)['token_x'] for t in c4_tokens])
    return x, y, torch.from_numpy(np.concatenate((c4, math))), torch.from_numpy(c4)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval import gsm8k_capture, gsm8k_pruning
    from sparsifier.prune import factor_atlas, input_side, sparsegpt, gram_dictionary, geometry_features, rescale
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--pilot-report', type=Path, required=True)
    p.add_argument('--c4-capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', default='all')
    p.add_argument('--modules', default='mlp.gate_proj,mlp.up_proj,mlp.down_proj')
    p.add_argument('--arm', choices=['dense','sparsegpt_task','sparsegpt_c4','input_side','input_rotated','input_repaired','local','factor','rank_matched','shuffled','global'], required=True)
    p.add_argument('--sparsity', type=float, default=.5)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    args = p.parse_args()
    if not 0 <= args.sparsity < 1 or args.alpha < 0 or not np.isfinite(args.alpha) or args.max_new_tokens < 1:
        p.error('invalid sparsity, alpha or generation budget')
    pilot_hash = check_competence(args.pilot_report)
    manifest = load_manifest(args.manifest)
    rows = [r for r in manifest['rows'] if r['split'] in ('geometry_fit','pruning_calibration')]
    evaluation = select_validation(manifest['rows'], args.eval_count, seed=0)
    c4_config = json.loads((args.c4_capture/'config.json').read_text())
    with np.load(args.c4_capture/'c4.npz', allow_pickle=False) as saved:
        if str(saved['config_sha256']) != digest(c4_config):
            raise ValueError('C4 source provenance mismatch')
        c4_tokens = saved['tokens'].copy()
    sources = [Path(__file__)] + [Path(m.__file__) for m in
        (gsm8k_capture, gsm8k_pruning, factor_atlas, input_side, sparsegpt, gram_dictionary, geometry_features, rescale)]
    config = {'model': MODEL, 'revision': REVISION, 'manifest_sha256': manifest['manifest_sha256'],
              'pilot_report_sha256': pilot_hash, 'c4_config_sha256': digest(c4_config),
              'blocks': args.blocks, 'modules': args.modules, 'arm': args.arm, 'sparsity':args.sparsity,
              'alpha':args.alpha, 'seed':args.seed, 'max_new_tokens':args.max_new_tokens,
              'local_pruning_version':LOCAL_PRUNING_VERSION,
              'evaluation_ids':[r['id'] for r in evaluation], 'calibration_ids':[r['id'] for r in rows],
              'purpose':'development_sequential_MLP_retention', 'atlas':{'groups':8,'width':4,'active':2,'charts':4,'rank':1},
              'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/'config.json', config)
    config_hash = digest(config)
    report_path = args.output/'report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['config_sha256'] != config_hash: raise ValueError('report mismatch')
        if old.get('complete'): return
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    # Local geometry needs a larger allocator ceiling than ambient SparseGPT;
    # GB10 unified-memory accounting overlaps host and CUDA reservations.
    memory = configure_cuda(80. if args.arm == 'local' else 56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation='sdpa').eval().to('cuda')
    model.requires_grad_(False)
    blocks = list(range(len(model.model.layers))) if args.blocks == 'all' else [int(b) for b in args.blocks.split(',')]
    sites = sites_for(blocks, args.modules.split(','), len(model.model.layers))
    total = sum(t.numel() for t in model.parameters())
    eligible = sum(model.model.layers[b].get_submodule(m).weight.numel() for b,m in sites)
    upstream = config_hash; stages = []; removed = 0
    by_id = {r['id']:i for i,r in enumerate(rows)}
    fit = np.array([r['split'] == 'geometry_fit' for r in rows])
    for n, site in enumerate(sites):
        b, module = site
        layer = model.model.layers[b].get_submodule(module)
        path = args.output/f'site-{n:03d}.pt'
        if path.exists():
            stored = torch.load(path, weights_only=True, map_location='cpu')
            verify_stage(stored, config_hash, upstream, site)
        else:
            weight = layer.weight.detach().float().cpu()
            if args.arm == 'dense':
                candidate, diagnostics = weight, {'solver':'dense'}
            else:
                x,y,inputs,c4 = collect(model,tokenizer,layer,rows,c4_tokens,args.seed)
                deltas = labels = pdiag = atlas_diagnostics = None
                if args.arm.startswith('input_'):
                    deltas,labels = pair_inputs(manifest,by_id,x,mode=args.arm,seed=args.seed)
                if args.arm in ('local','factor','rank_matched','shuffled','global'):
                    atlas = FactorAtlas(8,4,2,4,1,args.seed)
                    atlas.dictionary = GramBlockDictionary(8,4,2,args.seed)
                    atlas.fit(y[fit])
                    atlas.save(args.output/f'atlas-{n:03d}.npz')
                    frames,_,_ = atlas.frames(y,mode=args.arm,seed=args.seed)
                    pdiag = projector_diagonals(frames)
                    atlas_diagnostics = atlas.diagnostics(y)
                    del frames, atlas
                candidate,diagnostics = prune_candidate(weight,c4 if args.arm == 'sparsegpt_c4' else inputs,
                    arm=args.arm,sparsity=args.sparsity,alpha=args.alpha,deltas=deltas,labels=labels,
                    readout_inputs=torch.from_numpy(x),pdiag=pdiag)
                if atlas_diagnostics is not None:
                    diagnostics['atlas'] = atlas_diagnostics
                del x,y,inputs,c4,deltas,labels,pdiag
            candidate = candidate.to(torch.bfloat16).float()
            stored = {'config_sha256':config_hash,'upstream':upstream,'site':list(site),
                      'weight':candidate,'diagnostics':diagnostics,
                      'removed':int(((weight != 0) & (candidate == 0)).sum())}
            temporary = path.with_suffix('.tmp'); torch.save(stored,temporary); temporary.replace(path)
            del weight,candidate
        if stored['weight'].shape != layer.weight.shape: raise ValueError('candidate shape mismatch')
        with torch.no_grad(): layer.weight.copy_(stored['weight'])
        removed += stored['removed']
        upstream = hashlib.sha256(path.read_bytes()).hexdigest()
        stages.append({'site':list(site),'candidate_sha256':upstream,'removed':stored['removed'],
                       'diagnostics':stored['diagnostics']})
        del stored
        torch.cuda.empty_cache()
        atomic_json(report_path,{'complete':False,'config_sha256':config_hash,'stages':stages,
                                 'removed_weights':removed,'total_parameters':total,'eligible_parameters':eligible})
        print(json.dumps({'completed_site':site,'sites_done':n+1,'sites_total':len(sites),'host':host_memory()}),flush=True)
    evaluated = behavior(model,tokenizer,evaluation,args.output,config_hash,max_new_tokens=args.max_new_tokens)
    atomic_json(report_path,{'complete':True,'config_sha256':config_hash,'stages':stages,'behavior':evaluated,
        'removed_weights':removed,'total_parameters':total,'eligible_parameters':eligible,
        'whole_model_removed_fraction':removed/total,'eligible_removed_fraction':removed/eligible,
        'memory':memory,'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})


if __name__ == '__main__':
    main()
