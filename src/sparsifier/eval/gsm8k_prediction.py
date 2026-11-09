"""Exploratory held-out intervention prediction; CPU-only saved-result analysis."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sparsifier.data.gsm8k import digest, load_manifest
from sparsifier.eval.gsm8k import atomic_json


def split_ids(ids):
    """Split by identity, never outcomes; shared across compensation conditions."""
    ordered = sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    cut = int(.7 * len(ordered))
    if cut < 4 or len(ordered) - cut < 4:
        raise ValueError("insufficient interventions for held-out prediction")
    return set(ordered[:cut])


def predict(train_x, train_y, test_x):
    # All transforms and coefficients are estimated only on training interventions.
    train_x, test_x = np.log10(np.maximum(train_x, 1e-30)), np.log10(np.maximum(test_x, 1e-30))
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale > 1e-12, scale, 1)
    a = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    b = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    return b @ np.linalg.lstsq(a, train_y, rcond=None)[0]


def crossed_interval(values, families, seed=0, draws=2000):
    """Resample held-out interventions and problem families independently.

    Conditional on the fitted predictor; does not include training-fit uncertainty.
    """
    rng = np.random.default_rng(seed)
    groups = sorted(set(families))
    columns = {g: np.flatnonzero(np.array(families) == g) for g in groups}
    samples = []
    for _ in range(draws):
        rows = rng.integers(len(values), size=len(values))
        cols = np.concatenate([columns[g] for g in rng.choice(groups, len(groups), replace=True)])
        samples.append(values[np.ix_(rows, cols)].mean())
    return np.quantile(samples, [.025, .975]).tolist()


def analyze(report, config, manifest):
    if report['config_sha256'] != digest(config) or config['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('provenance mismatch')
    by_id = {r['id']: r for r in manifest['rows']}
    families = [by_id[i]['family'] for i in config['evaluation_ids']]
    result = {'scope': 'exploratory_development_prediction', 'source_complete': bool(report.get('complete')),
              'config_sha256': digest(config), 'problem_families': len(set(families)), 'conditions': {},
              'limitations': ['post-hoc development analysis', 'intervals conditional on fitted predictors',
                              'no multiple-comparison correction', 'not an independent final prediction test']}
    for mode in ('none', 'obs'):
        rows = sorted([r for r in report['outcomes'] if r['kind'] == 'representative' and r['compensation'] == mode], key=lambda r:r['id'])
        if len(rows) != config['singles']:
            raise ValueError('representative interventions incomplete')
        train_ids = split_ids([r['id'] for r in rows])
        train = np.array([r['id'] in train_ids for r in rows])
        y = np.array([r['delta_answer_nll'] for r in rows])
        if y.shape != (len(rows), len(families)) or not np.isfinite(y).all():
            raise ValueError('invalid outcome matrix')
        scores = {key: np.array([r['scores_sum_is_descriptive_for_groups'][key] for r in rows])
                  for key in rows[0]['scores_sum_is_descriptive_for_groups']}
        ambient = scores['ambient_obs'][:, None]
        base = predict(ambient[train], y[train].mean(1), ambient[~train])
        base_loss = (y[~train] - base[:, None]) ** 2
        condition = {'train_ids': sorted(train_ids), 'heldout_ids': [r['id'] for r,t in zip(rows,train) if not t],
                     'baseline_mse': float(base_loss.mean()), 'augmented': {}}
        for name, score in scores.items():
            if name == 'ambient_obs': continue
            x = np.column_stack([ambient, score])
            pred = predict(x[train], y[train].mean(1), x[~train])
            loss = (y[~train] - pred[:, None]) ** 2
            improvement = base_loss - loss
            condition['augmented'][name] = {'mse': float(loss.mean()),
                'mse_improvement': float(improvement.mean()),
                'conditional_crossed_ci95': crossed_interval(improvement, families)}
        result['conditions'][mode] = condition
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    result = analyze(json.loads((args.run/'report.json').read_text()),
                     json.loads((args.run/'config.json').read_text()), load_manifest(args.manifest))
    atomic_json(args.run/'prediction-analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
