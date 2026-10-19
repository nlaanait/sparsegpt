"""Export per-layer tangent probes for geometry-aware SparseGPT calibration.

Writes the directory layout expected by `--geometry-probes`:

    <root>/<block_index>/<module_name>.npy

Each array has shape [input_dim, probe_count] with unit-length columns when
produced by the bundled block-dictionary estimator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .geometry_features import BlockDictionary


def export_layer_probes(
    root,
    block_index,
    module_name,
    activations,
    groups=12,
    width=3,
    active=2,
    seed=0,
    neighbors=16,
    anchors=10,
    rank=1,
):
    """Fit calibration-only charts on activations and save ambient probes.

    `activations` must be shaped [input_dim, samples], matching SparseGPT's
    transposed activation convention after reshaping token batches.
    """
    root = Path(root)
    activations = np.asarray(activations)
    if activations.ndim != 2 or activations.shape[1] < 2 or not np.isfinite(activations).all():
        raise ValueError('activations must be a finite [input_dim, samples] matrix')
    model = BlockDictionary(groups, width, active, seed=seed).fit(activations)
    probes = model.tangents(
        activations,
        neighbors=neighbors,
        anchors=anchors,
        rank=rank,
        seed=seed + 1,
    )
    if probes.shape[1] == 0:
        raise ValueError(
            f'No tangent probes recovered for {block_index}/{module_name}; '
            'increase samples or adjust block dictionary settings'
        )
    out = root / str(block_index)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{module_name}.npy'
    # float32 keeps probe files smaller; add_geometry casts to the Hessian dtype.
    np.save(path, probes.astype(np.float32, copy=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activations', type=Path, required=True,
                        help='NPZ with H keyed by default, shape [input_dim, samples]')
    parser.add_argument('--output', type=Path, required=True, help='Probe directory root')
    parser.add_argument('--block-index', type=int, required=True)
    parser.add_argument('--module-name', required=True,
                        help='SparseGPT module key, e.g. fc1 or self_attn.q_proj')
    parser.add_argument('--key', default='H', help='Array key inside the NPZ')
    parser.add_argument('--groups', type=int, default=12)
    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('--active', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with np.load(args.activations, allow_pickle=False) as data:
        if args.key not in data:
            raise KeyError(f'{args.key} not in {args.activations}; have {list(data.keys())}')
        activations = data[args.key]
    path = export_layer_probes(
        args.output,
        args.block_index,
        args.module_name,
        activations,
        groups=args.groups,
        width=args.width,
        active=args.active,
        seed=args.seed,
    )
    print(path)


if __name__ == '__main__':
    main()
