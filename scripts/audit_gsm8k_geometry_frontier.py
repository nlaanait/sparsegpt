"""Recorded-layer CPU audit: does local approximation error jump at 70%?

Each sparsity uses its own ordered mask. Within a comparison every reference
uses that same mask and the same induced ridge. Coordinate crops are not
evidence about whole-model retention or the full learned geometric metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sparsifier.data.gsm8k import digest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_exactness import compare_fixed_mask
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.gsm8k_pruning import calibration
from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.whole_neuron_io import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, index, _, x, y = load_capture(args.capture)
    ambient, _, use = calibration(args.capture, index, x)
    weight = torch.from_numpy(np.load(args.capture/'weight.npy', mmap_mode='r').copy())
    frames, _, _ = FactorAtlas.load(args.atlas).frames(y[use])
    config = {'purpose':'recorded_coordinate_crop_approximation_frontier_not_retention',
        'capture_sha256':digest(source), 'atlas_sha256':file_hash(args.atlas),
        'script_sha256':file_hash(Path(__file__)), 'crop':[8,16],
        'sparsities':[.5,.7,.9], 'alphas':[.3,1.,3.], 'seeds':[0,1,2],
        'ridge':'same_original_coordinate_delta_times_scale_squared_within_each_mask',
        'sources':{name:file_hash(Path('src/sparsifier')/name) for name in (
            'eval/gsm8k_exactness.py','eval/gsm8k_pruning.py','prune/rescale.py',
            'prune/sparsegpt.py','prune/sample_metric.py')}}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/'config.json', config)
    report = {'config_sha256':digest(config), 'complete':False, 'rows':[]}
    for seed in config['seeds']:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(weight.shape[0],8,replace=False))
        cols = np.sort(rng.choice(weight.shape[1],16,replace=False))
        w, phi, z = weight[rows][:,cols], ambient[:,cols], torch.from_numpy(x[use][:,cols])
        # Preserve principal output metrics, with no crop reorthogonalization.
        metrics = torch.stack([torch.from_numpy(q[rows] @ q[rows].T).float() for q in frames])
        for alpha in config['alphas']:
            for sparsity in config['sparsities']:
                result = compare_fixed_mask(w,phi,z,metrics,alpha=alpha,sparsity=sparsity)
                result.update(seed=seed,sparsity=sparsity,row_ids=rows.tolist(),column_ids=cols.tolist())
                report['rows'].append(result)
                atomic_json(args.output/'report.json',report)
                print({'seed':seed,'alpha':alpha,'sparsity':sparsity,
                       'ordered_to_full_ratio':result['ordered_to_exact_full_ratio']},flush=True)
    report['complete'] = True
    atomic_json(args.output/'report.json',report)


if __name__ == '__main__':
    main()
