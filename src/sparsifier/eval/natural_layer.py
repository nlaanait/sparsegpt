"""Natural-text, one-layer Stage 2/3 experiments; GPU launches use gflow.

Concept errors measure held-out paired input differences at the local linear
output. They do not assert local-chart or block-output preservation. Fixed-span
rescaling uses the usual post-scaling damping surrogate, not an exact coupled
output solver. All language metrics come from final vocabulary logits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from sparsifier.data.concept_pairs import DEFAULT_CONCEPT_PAIRS, PRESENCE_REFERENCES, ConceptPair, edit_index
from sparsifier.eval.live_stages import _basis, _cap_cuda_under_brick, _random_basis, assert_below_brick
from sparsifier.prune.input_side import InputSideSparseGPT
from sparsifier.prune.ladder import fisher_sample_weights, row_scales_from_sample_weights
from sparsifier.prune.sparsegpt import SparseGPT


@contextmanager
def replaced_weight(layer, candidate):
    original = layer.weight.detach().clone()
    try:
        with torch.no_grad():
            layer.weight.copy_(candidate.to(device=original.device, dtype=original.dtype))
        yield
    finally:
        with torch.no_grad():
            layer.weight.copy_(original)


def record_aligned(model, layer, windows):
    """Record each input together with d(sum token NLL)/d(its output)."""
    inputs, gradients = [], []
    pending = {}

    def hook(_module, args, output):
        pending['input'] = args[0].detach().float().cpu()
        output.register_hook(lambda grad: pending.update(gradient=grad.detach().float().cpu()))

    handle = layer.register_forward_hook(hook)
    try:
        for tokens in windows:
            pending.clear()
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=tokens.to(layer.weight.device), use_cache=False).logits
            F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            tokens[:, 1:].to(logits.device).reshape(-1), reduction='sum').backward()
            if pending.get('gradient') is None:
                raise RuntimeError('selected layer did not receive an output gradient')
            x, g = pending['input'], pending['gradient']
            if x.shape[:2] != g.shape[:2]:
                raise RuntimeError('activation/gradient token axes differ')
            inputs.append(x[:, :-1].reshape(-1, x.shape[-1]))
            gradients.append(g[:, :-1].reshape(-1, g.shape[-1]))
            del logits
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return torch.cat(inputs), torch.cat(gradients)


@torch.no_grad()
def record_inputs(model, layer, windows):
    rows = []
    handle = layer.register_forward_pre_hook(
        lambda _m, args: rows.append(args[0].detach().float().cpu()[:, :-1].flatten(0, 1))
    )
    try:
        for tokens in windows:
            model(input_ids=tokens.to(layer.weight.device), use_cache=False)
    finally:
        handle.remove()
    return torch.cat(rows)


@torch.no_grad()
def capture_concepts(model, layer, tokenizer):
    """Disjoint prompt contexts per knob for fitting and final evaluation."""
    captured, result = {}, {}
    def hook(_m, args, _out):
        captured['x'] = args[0].detach().float().cpu()[0]
    handle = layer.register_forward_hook(hook)
    def run(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        model(input_ids=torch.tensor([ids], device=layer.weight.device), use_cache=False)
        return captured['x'].clone()
    try:
        for split, prefixes in {
            'fit': ('Example:\n', 'Exercise:\n'),
            'heldout': ('For reference:\n', 'Check this statement:\n'),
        }.items():
            deltas, presence, labels, manifest = [], [], [], []
            for prefix in prefixes:
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                for knob, (pair, ref) in enumerate(zip(DEFAULT_CONCEPT_PAIRS, PRESENCE_REFERENCES)):
                    full = ConceptPair(pair.knob, pair.domain, prefix + pair.left, prefix + pair.right)
                    index = edit_index(tokenizer, full)
                    left, right = run(full.left), run(full.right)
                    deltas.append(right[index] - left[index])
                    # Preserve the reference bank's agreed absent-token positions.
                    texts = [ref.absent, *ref.present]
                    for text in texts:
                        if tokenizer.encode(prefix + text, add_special_tokens=False) != prefix_ids + tokenizer.encode(text, add_special_tokens=False):
                            raise ValueError('prefix changes reference tokenization; presence positions would be invalid')
                    positions = [len(prefix_ids) + p for p in ref.positions]
                    absent = run(prefix + ref.absent)[positions].mean(0)
                    present = torch.stack([run(prefix + text)[positions].mean(0) for text in ref.present]).mean(0)
                    presence.append(present - absent)
                    labels.append(knob)
                    manifest.append({'knob': pair.knob, 'left': full.left, 'right': full.right,
                                     'absent': prefix + ref.absent, 'edit_index': index})
            result[split] = {'deltas': torch.stack(deltas), 'presence': torch.stack(presence),
                             'labels': labels, 'manifest': manifest}
    finally:
        handle.remove()
    return result


@torch.no_grad()
def vocabulary_reference(model, windows, device):
    result = []
    for tokens in windows:
        logits = model(input_ids=tokens.to(device), use_cache=False).logits[:, :-1].float()
        logp = logits.log_softmax(-1)
        nll = F.nll_loss(logp.flatten(0, 1), tokens[:, 1:].to(device).flatten(), reduction='mean')
        result.append({'logp': logp.cpu(), 'nll': float(nll)})
    return result


def interval(values, seed=0):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (1000, x.size), replace=True).mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(means, [.025, .975])]}


@torch.no_grad()
def language_metrics(model, windows, reference, device):
    rows = []
    for tokens, ref in zip(windows, reference):
        logits = model(input_ids=tokens.to(device), use_cache=False).logits[:, :-1].float()
        logp = logits.log_softmax(-1)
        target = tokens[:, 1:].to(device).flatten()
        nll = float(F.nll_loss(logp.flatten(0, 1), target, reduction='mean'))
        dense = ref['logp'].to(device)
        kl = float((dense.exp() * (dense - logp)).sum(-1).mean())
        rows.append({'nll': nll, 'delta_nll': nll - ref['nll'], 'vocabulary_kl': kl})
    return {'windows': rows, **{key: interval([r[key] for r in rows]) for key in rows[0]}}


def correlation(x, y, seed):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {'spearman': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(500):
        index = rng.integers(0, len(x), len(x))
        if np.ptp(x[index]) and np.ptp(y[index]):
            boot.append(float(spearmanr(x[index], y[index]).statistic))
    return {'spearman': float(spearmanr(x, y).statistic),
            'ci95': [float(v) for v in np.quantile(boot, [.025, .975])]}


def local_metrics(weight, candidate, phi, concepts, test_basis):
    delta = weight - candidate
    residual = phi @ delta.T
    result = {'ambient_mse': float(residual.square().sum(1).mean()),
              'heldout_span_mse': float((residual @ test_basis).square().sum(1).mean())}
    for field in ('deltas', 'presence'):
        error = (concepts[field] @ delta.T).square().sum(1)
        base = (concepts[field] @ weight.T).square().sum(1)
        result[field] = {'mse': float(error.mean()),
                         'relative_energy': float(error.sum() / base.sum().clamp_min(1e-12)),
                         'per_knob': {pair.knob: float(error[torch.tensor(concepts['labels']) == i].mean())
                                      for i, pair in enumerate(DEFAULT_CONCEPT_PAIRS)}}
    return result


def prune(weight, phi, *, method, deltas=None, labels=None, alpha=0., scale=None):
    """Use the same block size, damping, and exact top-k mask path for OBS arms."""
    if method == 'wanda':
        score = weight.square() * phi.square().mean(0)
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask.scatter_(1, score.topk(weight.shape[1] // 2, largest=False).indices, True)
        return weight.masked_fill(mask, 0)
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight if scale is None else weight * scale)
    solver = InputSideSparseGPT(layer) if method == 'input_side' else SparseGPT(layer)
    solver.add_batch(phi)
    if method == 'input_side':
        solver.add_pairs(deltas, knob_ids=torch.tensor(labels))
        solver.finalize_pairs(alpha)
    solver.fasterprune(.5, blocksize=128, percdamp=.01, mask_weight=torch.ones_like(weight))
    candidate = layer.weight.detach().clone()
    if scale is not None:
        candidate /= scale
    if int((candidate == 0).sum()) != weight.numel() // 2:
        raise RuntimeError('arm did not achieve exactly 50% elementwise sparsity')
    return candidate


def knockout_metrics(row_delta, output_row, phi, concepts, test_basis, base_pair_energies):
    """Use the rank-one knockout residual instead of materializing dense products."""
    ambient = float((phi @ row_delta).square().mean())
    result = {'ambient_mse': ambient,
              'heldout_span_mse': ambient * float(test_basis[output_row].square().sum())}
    for field in ('deltas', 'presence'):
        error = (concepts[field] @ row_delta).square()
        result[field] = {'mse': float(error.mean()),
                         'relative_energy': float(error.sum() / base_pair_energies[field].clamp_min(1e-12)),
                         'per_knob': {pair.knob: float(error[torch.tensor(concepts['labels']) == i].mean())
                                      for i, pair in enumerate(DEFAULT_CONCEPT_PAIRS)}}
    return result


def atomic_json(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stage2(model, layer, weight, phi, gradients, concepts, test_phi, windows, refs, report, path, count, seed):
    hessian = phi.T @ phi / phi.shape[0]
    hessian.diagonal().add_(.01 * hessian.diagonal().mean())
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    fit_basis = _basis(concepts['fit']['deltas'] @ weight.T)
    test_basis = _basis(concepts['heldout']['deltas'] @ weight.T)
    leverage = fit_basis.square().sum(1)
    energy = phi.square().mean(0)
    scores = {'magnitude': weight.square(), 'wanda': weight.square() * energy,
              'ambient_obs': weight.square() / inverse.diagonal(),
              'fisher': weight.square() * (gradients.square().T @ phi.square()) / len(phi),
              'geometric_wanda': weight.square() * leverage[:, None] * energy,
              'geometric_obs': weight.square() * leverage[:, None] / inverse.diagonal()}
    for control in range(3):
        q = _random_basis(weight.shape[0], fit_basis.shape[1], 100 + control)
        scores[f'random_obs_{control}'] = weight.square() * q.square().sum(1)[:, None] / inverse.diagonal()
    entries = np.random.default_rng(seed + 11).choice(weight.numel(), count, replace=False)
    rows = report.setdefault('knockouts', [])
    base_pair_energies = {field: (concepts['heldout'][field] @ weight.T).square().sum()
                          for field in ('deltas', 'presence')}
    completed = {(r['entry'], r['compensation']) for r in rows}
    for entry in entries:
        p, q = divmod(int(entry), weight.shape[1])
        for mode in ('none', 'obs'):
            if (int(entry), mode) in completed:
                continue
            delta = torch.zeros(weight.shape[1])
            if mode == 'obs':
                delta = weight[p, q] * inverse[q] / inverse[q, q]
            else:
                delta[q] = weight[p, q]
            candidate = weight.clone()
            candidate[p] -= delta
            candidate[p, q] = 0
            candidate[p] = candidate[p].to(layer.weight.dtype).float()
            with replaced_weight(layer, candidate):
                lm = language_metrics(model, windows, refs, layer.weight.device)
            rows.append({'entry': int(entry), 'row': p, 'column': q, 'compensation': mode,
                         'scores': {k: float(v[p, q]) for k, v in scores.items()},
                         'local': knockout_metrics(weight[p] - candidate[p], p, test_phi,
                                                   concepts['heldout'], test_basis, base_pair_energies),
                         'language': lm})
            atomic_json(path, report)
        if len(rows) % 32 == 0:
            print(f'stage2: {len(rows)}/{count * 2} knockout outcomes', flush=True)
    report['correlations'] = {}
    for mode in ('none', 'obs'):
        selected = [r for r in rows if r['compensation'] == mode]
        targets = {'heldout_span_mse': [r['local']['heldout_span_mse'] for r in selected],
                   'pair_difference_mse': [r['local']['deltas']['mse'] for r in selected],
                   'delta_nll': [r['language']['delta_nll']['mean'] for r in selected],
                   'vocabulary_kl': [r['language']['vocabulary_kl']['mean'] for r in selected]}
        report['correlations'][mode] = {target: {name: correlation([r['scores'][name] for r in selected], y, seed)
                                                for name in scores} for target, y in targets.items()}


def stage3(model, layer, weight, phi, gradients, concepts, test_phi, windows, refs, report, path, seed):
    fit_basis = _basis(concepts['fit']['deltas'] @ weight.T)
    test_basis = _basis(concepts['heldout']['deltas'] @ weight.T)
    arms = [('sparsegpt', {}), ('wanda', {})]
    arms += [(f'input_side_alpha_{alpha:g}', {'method': 'input_side', 'alpha': alpha}) for alpha in (.3, 1., 3.)]
    gamma = weight.shape[0] / fit_basis.shape[1]
    fixed_scale = (1 + gamma * fit_basis.square().sum(1)[:, None]).sqrt().expand_as(weight)
    arms.append(('fixed_tangent', {'scale': fixed_scale}))
    fisher_scale = row_scales_from_sample_weights(fisher_sample_weights(gradients, 1.), phi)
    arms.append(('fisher', {'scale': fisher_scale}))
    for control in range(3):
        q = _random_basis(weight.shape[0], fit_basis.shape[1], 100 + control)
        arms.append((f'random_tangent_{control}', {'scale': (1 + gamma * q.square().sum(1)[:, None]).sqrt().expand_as(weight)}))
    results = report.setdefault('arms', {})
    for name, kwargs in arms:
        if name in results:
            continue
        method = kwargs.pop('method', 'wanda' if name == 'wanda' else 'sparsegpt')
        candidate = prune(weight, phi, method=method, deltas=concepts['fit']['deltas'], labels=concepts['fit']['labels'], **kwargs)
        # Evaluate the exact precision used when inserting the weight into the model.
        candidate = candidate.to(layer.weight.dtype).float()
        with replaced_weight(layer, candidate):
            lm = language_metrics(model, windows, refs, layer.weight.device)
        results[name] = {'local': local_metrics(weight, candidate, test_phi, concepts['heldout'], test_basis),
                         'language': lm, 'elementwise_sparsity': float((candidate == 0).float().mean())}
        atomic_json(path, report)
        print(f'stage3 {name}: delta_nll={lm["delta_nll"]["mean"]:.6g}', flush=True)
    baseline = results['sparsegpt']
    for arm in results.values():
        arm['paired_delta_nll_vs_sparsegpt'] = interval([
            a['nll'] - b['nll'] for a, b in zip(arm['language']['windows'], baseline['language']['windows'])], seed)
        arm['within_nll_tolerance'] = arm['paired_delta_nll_vs_sparsegpt']['ci95'][1] <= report['nll_tolerance']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['stage2', 'stage3'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--block', type=int, default=12)
    parser.add_argument('--module', default='mlp.down_proj')
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--seqlen', type=int, default=128)
    parser.add_argument('--calibration-windows', type=int, default=16)
    parser.add_argument('--eval-windows', type=int, default=8)
    parser.add_argument('--knockouts', type=int, default=256)
    parser.add_argument('--max-gib', type=float, default=8.)
    parser.add_argument('--output-dir', type=Path, default=Path('results/tangent/natural-layer'))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'{args.phase}-seed{args.seed}.json'
    report = json.loads(path.read_text()) if path.exists() else {'config': config, 'nll_tolerance': .02}
    if report['config'] != config:
        raise ValueError('output belongs to another config/source; choose a new output directory')
    if report.get('complete'):
        print(f'already complete: {path}', flush=True)
        return
    assert_below_brick('before-natural-layer')
    cap = _cap_cuda_under_brick(int(args.max_gib * 1024**3))
    from sparsifier.models.qwen import force_pytorch_attention, get_qwen
    from sparsifier.data.loaders import get_loaders
    from transformers import AutoTokenizer
    force_pytorch_attention()
    model = get_qwen(args.model, use_kernels=False).eval().to('cuda')
    model.config.use_cache = False
    model.requires_grad_(False)
    layer = model.model.layers[args.block].get_submodule(args.module)
    layer.weight.requires_grad_(True)
    weight = layer.weight.detach().float().cpu().clone()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cal, _ = get_loaders('c4', args.calibration_windows, args.seed, args.seqlen, args.model)
    cal_windows = [tokens for tokens, _ in cal]
    _, wiki = get_loaders('wikitext2', 1, 0, args.seqlen, args.model)
    # Stage 2 and Stage 3 use disjoint test windows; all calibration seeds share
    # each phase's held-out windows to permit paired method/seed comparisons.
    offset = 0 if args.phase == 'stage2' else args.eval_windows * args.seqlen
    windows = [wiki.input_ids[:, offset + i * args.seqlen:offset + (i + 1) * args.seqlen] for i in range(args.eval_windows)]
    if any(t.shape[1] != args.seqlen for t in windows):
        raise ValueError('not enough held-out WikiText2 tokens')
    cache = args.output_dir / f'calibration-seed{args.seed}.pt'
    cache_config = {k: config[k] for k in ('model', 'block', 'module', 'seed', 'seqlen', 'calibration_windows', 'source_sha256')}
    if cache.exists():
        stored = torch.load(cache, weights_only=False)
        if stored['config'] != cache_config:
            raise ValueError('calibration cache provenance mismatch')
        phi, gradients, concepts = stored['phi'], stored['gradients'], stored['concepts']
    else:
        phi, gradients = record_aligned(model, layer, cal_windows)
        concepts = capture_concepts(model, layer, tokenizer)
        torch.save({'config': cache_config, 'phi': phi, 'gradients': gradients, 'concepts': concepts}, cache)
    test_phi = record_inputs(model, layer, windows)
    refs = vocabulary_reference(model, windows, layer.weight.device)
    dense_mean_nll = float(np.mean([r['nll'] for r in refs]))
    if not np.isfinite(dense_mean_nll) or dense_mean_nll > np.log(100):
        raise RuntimeError(f'implausible dense natural-text score: mean NLL={dense_mean_nll}; stop before pruning')
    report.update({'layer_shape': list(weight.shape), 'calibration_tokens': len(phi),
                   'evaluation_tokens': len(test_phi), 'calibration': 'c4/train', 'evaluation': 'wikitext2/test',
                   'dense_nll': interval([r['nll'] for r in refs]), 'cap_gib': cap / 1024**3,
                   'concept_manifest': {k: v['manifest'] for k, v in concepts.items()},
                   'limitations': ['concept holdout changes prompt context; knob values are shared',
                                   'fixed-span and local linear-output metrics; no local-chart claim',
                                   'confidence intervals resample windows or sampled knockout entries']})
    print(f'{args.phase}: layer={args.block}/{args.module}, shape={weight.shape}, aligned_tokens={len(phi)}', flush=True)
    atomic_json(path, report)
    if args.phase == 'stage2':
        stage2(model, layer, weight, phi, gradients, concepts, test_phi, windows, refs, report, path, args.knockouts, args.seed)
    else:
        stage3(model, layer, weight, phi, gradients, concepts, test_phi, windows, refs, report, path, args.seed)
    report.update(complete=True, peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
    atomic_json(path, report)
    print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
