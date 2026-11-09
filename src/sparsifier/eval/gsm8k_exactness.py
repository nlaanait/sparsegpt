"""CPU exactness audit of geometric pruning on cropped recorded-layer problems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from sparsifier.data.gsm8k import digest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.gsm8k_pruning import calibration, prune_candidate
from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.sample_metric import quadratic_hessian, refit, objective
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION, local_row_scales


def compare_fixed_mask(weight, ambient, readouts, metrics, *, alpha=1., sparsity=.5):
    """Separate output coupling, congruence, and ordered-solve errors.

    All four candidates use the ordered solver's mask. All three reference
    curvatures include the *same* original-coordinate ridge δ Λ² induced by
    the actual scaled-coordinate solver. Also report losses with δ I and no
    ridge to expose damping effects. This is a small CPU reference only.
    """
    pd = metrics.diagonal(dim1=1, dim2=2)
    scale, diagnostics = local_row_scales(ambient, readouts, pd, alpha)
    if not (ambient.square().sum(0) > 1e-12).all():
        raise ValueError("fixed-mask reference requires supported input coordinates")
    ordered, _ = prune_candidate(weight, ambient, arm="local", alpha=alpha,
        sparsity=sparsity, readout_inputs=readouts, pdiag=pd, device="cpu")
    keep = ordered != 0
    h0 = ambient.double().T @ ambient.double() / len(ambient)
    damping = .01 * float(h0.diag().mean())
    full_unregularized = quadratic_hessian(ambient, readouts, metrics, diagnostics["gamma"])
    ridge = torch.diag(damping * scale.double().square().flatten())
    full = full_unregularized + ridge
    dout, din = weight.shape
    blocks = full_unregularized.reshape(dout, din, dout, din)
    row = torch.block_diag(*[blocks[p, :, p, :] for p in range(dout)]) + ridge
    congruence = torch.block_diag(*[
        scale[p].double()[:, None] * h0 * scale[p].double()[None, :]
        for p in range(dout)]) + ridge
    candidates = {"full_coupled_exact": refit(weight, keep, full),
                  "row_diagonal_exact": refit(weight, keep, row),
                  "congruence_exact": refit(weight, keep, congruence),
                  "ordered_sparsegpt": ordered}
    curvatures = {"full": full, "row_diagonal": row, "congruence": congruence,
                  "full_isotropic_ridge": full_unregularized + damping * torch.eye(weight.numel()),
                  "full_unregularized": full_unregularized}
    losses = {name: {metric: objective(weight, candidate, h) for metric, h in curvatures.items()}
              for name, candidate in candidates.items()}
    residuals = {}
    for name, h in (("full_coupled_exact", full), ("row_diagonal_exact", row),
                    ("congruence_exact", congruence)):
        gradient = h @ (weight.double() - candidates[name]).flatten()
        residuals[name] = float(gradient[keep.flatten()].abs().max()) if keep.any() else 0.
    return {"normalization": LOCAL_PRUNING_VERSION, "alpha": alpha, "diagnostics": diagnostics,
            "damping": damping, "damping_convention": "common_original_coordinate_delta_times_scale_squared",
            "keep_mask": keep.tolist(), "losses": losses, "free_gradient_max": residuals,
            "ordered_to_exact_full_ratio": losses["ordered_sparsegpt"]["full"] / max(losses["full_coupled_exact"]["full"], 1e-30),
            "output_coupling_gap": losses["row_diagonal_exact"]["full"] - losses["full_coupled_exact"]["full"],
            "row_congruence_gap": losses["congruence_exact"]["row_diagonal"] - losses["row_diagonal_exact"]["row_diagonal"],
            "ordered_solver_gap": losses["ordered_sparsegpt"]["congruence"] - losses["congruence_exact"]["congruence"]}


def mask_overlap(left, right):
    a, b = ~torch.as_tensor(left, dtype=torch.bool), ~torch.as_tensor(right, dtype=torch.bool)
    return {"removed_jaccard": float((a & b).sum()) / max(int((a | b).sum()), 1),
            "changed_fraction": float((a != b).double().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sparsifier.prune import rescale, sparsegpt, sample_metric
    from sparsifier.eval import gsm8k_pruning
    torch.set_num_threads(4)
    config, index, _, x, y = load_capture(args.capture)
    ambient, _, use = calibration(args.capture, index, x)
    w = torch.from_numpy(np.load(args.capture / "weight.npy"))
    atlas = FactorAtlas.load(args.atlas)
    frames, _, _ = atlas.frames(y[use])
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"capture_sha256": digest(config), "atlas": str(args.atlas),
                  "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
                  "audit_version": "controlled-local-v3", "local_pruning_version": LOCAL_PRUNING_VERSION,
                  "sources": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (rescale, sparsegpt, sample_metric, gsm8k_pruning)},
                  "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "crop": [8,16], "seeds": [0,1,2], "alphas": [.3,1.,3.],
                  "scope": "principal_output_metric_and_input_coordinate_crop"}
    write_immutable(args.output / "config.json", provenance)
    # The complete layer's trace needs only ranks and input energies, avoiding
    # a full output-by-input geometric matrix on CPU.
    ranks = torch.tensor([q.shape[1] for q in frames], dtype=torch.double)
    ambient_energy = ambient.double().square().sum(0)
    supported = ambient_energy > 1e-12
    readouts = torch.from_numpy(x[use]).double()
    geo_trace = float((ranks * readouts[:, supported].square().sum(1)).sum())
    ambient_trace = w.shape[0] * float(ambient_energy[supported].sum())
    gamma_rank = w.shape[0] * len(ambient) / float(ranks.sum())
    report = {"config": provenance, "rows": [], "complete": False,
              "site": {"block": config["block"], "module": config["module"]},
              "atlas_diagnostics": atlas.diagnostics(y[use]),
              "full_layer_strength_at_alpha_one": {
                  "rank_v1_gamma": gamma_rank,
                  "rank_v1_added_curvature_trace_ratio": gamma_rank * geo_trace / ambient_trace,
                  "curvature_trace_v2_gamma": ambient_trace / geo_trace if geo_trace > 0 else 0.,
                  "curvature_trace_v2_added_curvature_trace_ratio": 1. if geo_trace > 0 else 0.}}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(w.shape[0], 8, replace=False))
        cols = np.sort(rng.choice(w.shape[1], 16, replace=False))
        sub = w[rows][:, cols]
        phi = ambient[:, cols]
        z = torch.from_numpy(x[use][:, cols])
        # Principal submatrices preserve the original metric; reorthogonalizing
        # cropped frames would silently replace the objective being audited.
        metrics = torch.stack([torch.from_numpy(q[rows] @ q[rows].T).float() for q in frames])
        baseline, _ = prune_candidate(sub, phi, arm="sparsegpt_task", device="cpu")
        for alpha in (.3,1.,3.):
            result = compare_fixed_mask(sub, phi, z, metrics, alpha=alpha)
            result.update(seed=seed, row_ids=rows.tolist(), column_ids=cols.tolist(),
                          mask_overlap_ambient=mask_overlap(result["keep_mask"], baseline != 0))
            report["rows"].append(result)
            atomic_json(args.output / "report.json", report)
    report["complete"] = True
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
