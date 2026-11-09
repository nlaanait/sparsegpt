"""CPU example: fit local charts, pull them back, then prune with SparseGPT.

Run: python -m examples.downstream_tangent_pruning --num-probes 16
Uses synthetic activations; reported errors are diagnostics, not evidence
of an advantage on language models. No GPU or model download is needed.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from torch import nn

from sparsifier.prune.downstream import (
    ChartTangentBasis,
    DownstreamTangentCalibrator,
    TangentAnchor,
)
from sparsifier.prune.sparsegpt import SparseGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--input-alpha", type=float, default=0.0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 4)).eval()
    model.requires_grad_(False)
    baseline = copy.deepcopy(model)
    # Fit, calibration, and evaluation samples are disjoint.
    fit_x, calibration_x, heldout_x = (torch.randn(128, 8) for _ in range(3))
    anchor_basis = ChartTangentBasis.fit(model(fit_x), charts=4, rank=1, seed=args.seed)
    input_basis = ChartTangentBasis.fit(fit_x, charts=4, rank=1, seed=args.seed)
    dense_local = model[0](heldout_x)
    dense_output = model(heldout_x)
    heldout_frames = anchor_basis(dense_output)

    calibrator = DownstreamTangentCalibrator(
        targets={"first": model[0]},
        anchors={"last": TangentAnchor(model[2], anchor_basis)},
        alpha=args.alpha,
        input_alpha=args.input_alpha,
        input_bases={"first": input_basis} if args.input_alpha else None,
        num_probes=args.num_probes,
        seed=args.seed,
    )
    for batch in calibration_x.split(32):
        calibrator.run(lambda: model(batch))
    solver = calibrator.solvers["first"]
    solver.fasterprune(args.sparsity, blocksize=4)
    ambient = SparseGPT(baseline[0])
    for batch in calibration_x.split(32):
        ambient.add_batch(batch)
    ambient.fasterprune(args.sparsity, blocksize=4, mask_weight=torch.ones_like(baseline[0].weight))

    def evaluate(candidate):
        error = candidate(heldout_x) - dense_output
        projected = torch.einsum("ndr,nd->nr", heldout_frames, error)
        return {
            "actual_sparsity": float((candidate[0].weight == 0).float().mean()),
            "local_ambient_error": float((candidate[0](heldout_x) - dense_local).square().sum(-1).mean()),
            "downstream_ambient_error": float(error.square().sum(-1).mean()),
            "downstream_tangent_error": float(projected.square().sum(-1).mean()),
        }

    print(json.dumps({
        "calibration": solver.tangent_diagnostics,
        "num_probes_per_anchor_per_batch": args.num_probes,
        "ambient_sparsegpt": evaluate(baseline),
        "downstream_tangent_sparsegpt": evaluate(model),
    }, indent=2))


if __name__ == "__main__":
    main()
