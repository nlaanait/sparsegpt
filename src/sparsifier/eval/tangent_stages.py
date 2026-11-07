"""Stage 0–4 harness (revised note §6).

``run_stages`` keeps the original synthetic circle sweep and attaches
``revised_protocol``, which runs the note's diagnostics, the two concept
families, the score comparisons, the Variant A–D order, and the bias arms.
Production CLI prune runs stay separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sparsifier.eval.revised_protocol import run_revised_protocol
from sparsifier.prune.exact_ltan import (
    diag_p_obs_prune,
    diag_p_row_hessians,
    greedy_full_obs,
    kronecker_projected_hessian,
)
from sparsifier.prune.metric import blend_anchor_into_diagonals
from sparsifier.prune.pruners import get_pruner
from sparsifier.prune.wanda import wanda_scores


def make_concept_data(seed: int, n_train: int, n_test: int, dim: int = 8):
    """Circular factor sweep with analytic output tangents (toy note example)."""
    rng = np.random.default_rng(seed)
    factors = dim // 2
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    weight = rng.normal(size=(dim, dim)) / np.sqrt(dim)

    def sample(n: int):
        concept = rng.integers(factors, size=n)
        angle = rng.uniform(-np.pi, np.pi, size=n)
        x = np.zeros((n, dim), dtype=np.float64)
        q = np.zeros((n, dim), dtype=np.float64)
        for k, (g, theta) in enumerate(zip(concept, angle)):
            sl = slice(2 * g, 2 * g + 2)
            local_x = np.array([np.cos(theta), np.sin(theta)])
            local_t = np.array([-np.sin(theta), np.cos(theta)])
            radius = 0.65 + 0.15 * g
            x[k] = radius * (basis[:, sl] @ local_x)
            tangent = radius * (weight @ basis[:, sl] @ local_t)
            q[k] = tangent / max(np.linalg.norm(tangent), 1e-15)
        return x, q, concept

    return weight, basis, sample(n_train), sample(n_test)


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ra = np.empty_like(a)
    rb = np.empty_like(b)
    ra[np.argsort(a, kind="mergesort")] = np.arange(len(a), dtype=np.float64)
    rb[np.argsort(b, kind="mergesort")] = np.arange(len(b), dtype=np.float64)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def stage1_tangent_stability(train, test, weight, basis) -> dict[str, Any]:
    """H1: local vs fixed-global rank-1 finite-difference relative error."""
    x_train, q_train, g_train = train
    x_test, q_test, concepts = test
    delta = 0.15
    local_err = []
    global_err = []
    for i, concept in enumerate(concepts):
        sl = slice(2 * concept, 2 * concept + 2)
        # Finite-difference along analytic angle.
        x = x_test[i]
        # Local tangent from train neighbors of same concept.
        same = np.flatnonzero(g_train == concept)
        if same.size < 2:
            continue
        centered = x_train[same] - x_train[same].mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        local_dir = vt[0]
        global_dir = basis[:, sl] @ np.array([0.0, 1.0])
        global_dir = global_dir / max(np.linalg.norm(global_dir), 1e-15)
        # Projected residual sensitivity proxy.
        local_err.append(
            float(np.linalg.norm(q_test[i] - local_dir * np.sign(np.dot(q_test[i], local_dir))))
        )
        global_err.append(
            float(
                np.linalg.norm(q_test[i] - global_dir * np.sign(np.dot(q_test[i], global_dir)))
            )
        )
    return {
        "local_tangent_mean_rel_error": float(np.mean(local_err)) if local_err else float("nan"),
        "fixed_global_mean_rel_error": float(np.mean(global_err)) if global_err else float("nan"),
        "n_evaluated": len(local_err),
    }


def stage2_score_ranking(weight, x_train, q_train, x_test, q_test) -> dict[str, Any]:
    """H2: predicted scores vs held-out direct tangent impact."""
    n_test = len(x_test)
    heldout = weight**2 * ((q_test**2).T @ (x_test**2)) / n_test
    x2 = np.mean(x_train**2, axis=0)
    wanda = weight**2 * x2[None, :]
    pdiag = q_train**2
    tangent_wanda = weight**2 * (pdiag.T @ (x_train**2) / len(x_train))
    magnitude = weight**2
    return {
        "magnitude_vs_heldout_direct_spearman": _spearman(magnitude, heldout),
        "wanda_vs_heldout_direct_spearman": _spearman(wanda, heldout),
        "tangent_wanda_vs_heldout_direct_spearman": _spearman(tangent_wanda, heldout),
    }


def stage0_diagnostics(q_train) -> dict[str, float]:
    """Stage 0 on the toy concept: leverage null and radial share."""
    from sparsifier.prune.stage0 import leverage_concentration, radial_share

    tangent = torch.tensor(q_train[0], dtype=torch.float32)
    leverages = tangent.square()
    summary = leverage_concentration(leverages, rank=1)
    share = radial_share(torch.tensor(q_train[1], dtype=torch.float32), tangent)
    return {"leverage_ks": summary["ks_statistic"], "radial_share": float(share)}


def stage3_compensated_methods(
    weight, x_train, q_train, x_test, q_test, *, sparsity: float
) -> dict[str, Any]:
    """H3: matched-sparsity arms including Wanda, ambient, exact diag-P."""
    w_t = torch.tensor(weight, dtype=torch.float32)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    q_t = torch.tensor(q_train, dtype=torch.float32)
    diags = torch.tensor(q_train**2, dtype=torch.float32)
    arms: dict[str, Any] = {}

    # Wanda (no compensation).
    scores = wanda_scores(w_t, x_t)
    sparse = w_t.clone()
    k = int(sparsity * sparse.numel())
    chosen = torch.topk(scores.flatten(), k, largest=False).indices
    flat = sparse.reshape(-1)
    flat[chosen] = 0
    arms["wanda"] = _eval_arm(weight, sparse.numpy(), x_test, q_test)

    # Exact diag-P OBS.
    row_h = diag_p_row_hessians(x_t, diags, percdamp=0.01)
    sparse_diag = diag_p_obs_prune(w_t, row_h, sparsity)
    arms["exact_diag_p"] = _eval_arm(weight, sparse_diag.numpy(), x_test, q_test)

    # Full Kronecker when tiny.
    if weight.size <= 64:
        metrics = torch.stack([torch.diag(d) for d in diags], dim=0)
        h = kronecker_projected_hessian(x_t, metrics, ridge=1e-5)
        sparse_full = greedy_full_obs(h, w_t, sparsity)
        arms["exact_full_kronecker"] = _eval_arm(weight, sparse_full.numpy(), x_test, q_test)

    # Option-1-style registry pruner on a tiny Linear.
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(w_t)
    bound = get_pruner(
        "tangent", alpha=1.0, charts=4, rank=1, calibration_samples=min(256, len(x_train)), seed=0
    ).bind(layer, block_index=0, module_name="toy")
    # Fake outputs = W x + noise along q for charting.
    y = (w_t @ x_t.T).T + 0.05 * q_t
    bound.add_batch(x_t.unsqueeze(0), y.unsqueeze(0))
    bound.finalize()
    bound.prune(sparsity=sparsity, blocksize=max(2, weight.shape[1] // 2))
    arms["option1_tangent"] = _eval_arm(
        weight, layer.weight.detach().cpu().numpy(), x_test, q_test
    )
    from sparsifier.prune.revised_methods import sparse_plus_low_rank

    basis = torch.tensor(q_train.mean(axis=0), dtype=torch.float32).reshape(-1, 1)
    corrected = sparse_plus_low_rank(w_t, sparse, basis)
    arms["variant_d"] = _eval_arm(weight, corrected.numpy(), x_test, q_test)
    return arms


def stage4_bias_fold(weight, x_train, q_train, x_test, q_test, *, sparsity: float) -> dict[str, Any]:
    """H4: no bias vs projected bias after a fixed Wanda mask."""
    w_t = torch.tensor(weight, dtype=torch.float32)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    scores = wanda_scores(w_t, x_t)
    sparse = w_t.clone()
    k = int(sparsity * sparse.numel())
    chosen = torch.topk(scores.flatten(), k, largest=False).indices
    flat = sparse.reshape(-1)
    flat[chosen] = 0
    residual_train = ((w_t - sparse) @ x_t.T).T
    # Projected LS bias with diagonal M_n = diag(q_n^2).
    metrics = torch.stack([torch.diag(torch.tensor(q, dtype=torch.float32)**2) for q in q_train])
    from sparsifier.prune.bias import projected_bias_correction

    delta_b = projected_bias_correction(residual_train, metrics).numpy()
    no_bias = _eval_arm(weight, sparse.numpy(), x_test, q_test, correction=np.zeros(weight.shape[0]))
    with_bias = _eval_arm(weight, sparse.numpy(), x_test, q_test, correction=delta_b)
    return {"without_bias": no_bias, "with_projected_bias": with_bias}


def _eval_arm(weight, sparse, x, q, correction=None) -> dict[str, float]:
    residual = ((weight - sparse) @ x.T).T
    if correction is not None:
        residual = residual + correction[None, :]
    projected = np.sum((q * residual) ** 2, axis=1)
    return {
        "tangent_mse": float(np.mean(projected)),
        "ambient_mse": float(np.mean(np.sum(residual**2, axis=1))),
        "actual_sparsity": float(1.0 - np.count_nonzero(sparse) / sparse.size),
    }


def run_stages(
    *,
    seed: int = 0,
    n_train: int = 256,
    n_test: int = 128,
    sparsity: float = 0.5,
    anchor_eta: float = 0.0,
) -> dict[str, Any]:
    """Run Stages 1–4 on one synthetic seed; optional η anchor blend in Stage 2."""
    weight, basis, train, test = make_concept_data(seed, n_train, n_test)
    x_train, q_train, _ = train
    x_test, q_test, _ = test
    if anchor_eta > 0.0:
        centers = torch.stack(
            [
                torch.tensor(q_train[::2].mean(axis=0), dtype=torch.float32),
                torch.tensor(q_train[1::2].mean(axis=0), dtype=torch.float32),
            ]
        )
        assignments = torch.arange(len(q_train)) % 2
        diags = blend_anchor_into_diagonals(
            torch.tensor(q_train**2, dtype=torch.float32),
            centers,
            assignments,
            float(anchor_eta),
        )
        q_train = np.sqrt(np.clip(diags.numpy(), 0.0, None))
    return {
        "seed": seed,
        "train_samples": n_train,
        "test_samples": n_test,
        "sparsity": sparsity,
        "anchor_eta": anchor_eta,
        "stage0_diagnostics": stage0_diagnostics(q_train),
        "stage1_tangent_stability": stage1_tangent_stability(train, test, weight, basis),
        "stage2_score_ranking": stage2_score_ranking(weight, x_train, q_train, x_test, q_test),
        "stage3_compensated_methods": stage3_compensated_methods(
            weight, x_train, q_train, x_test, q_test, sparsity=sparsity
        ),
        "stage4_bias_fold": stage4_bias_fold(
            weight, x_train, q_train, x_test, q_test, sparsity=sparsity
        ),
        "revised_protocol": run_revised_protocol(
            seed=seed,
            n_train=n_train,
            n_test=n_test,
            sparsity=sparsity,
            dim=8,
        ),
    }


def summarize_stages(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std across seeds for key Stage-2 Spearman metrics and Stage-3 MSEs."""
    summary: dict[str, Any] = {"seed_count": len(reports), "stage2_score_ranking": {}, "stage3": {}}
    if "revised_protocol" in reports[0]:
        summary["revised_protocol"] = {
            "stage0_decisions": {
                key: [run["revised_protocol"]["stage0"][key]["decision"] for run in reports]
                for key in ("D1", "D2", "D3", "D4", "D5", "D6")
            }
        }
    keys = reports[0]["stage2_score_ranking"]
    for key in keys:
        values = np.array([r["stage2_score_ranking"][key] for r in reports], dtype=np.float64)
        summary["stage2_score_ranking"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    arms = sorted(reports[0]["stage3_compensated_methods"])
    for arm in arms:
        summary["stage3"][arm] = {}
        for metric in ("tangent_mse", "ambient_mse"):
            values = np.array(
                [r["stage3_compensated_methods"][arm][metric] for r in reports],
                dtype=np.float64,
            )
            summary["stage3"][arm][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--anchor-eta", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/tangent-stages.json"),
    )
    args = parser.parse_args(argv)
    reports = [
        run_stages(
            seed=seed,
            n_train=args.train_samples,
            n_test=args.test_samples,
            sparsity=args.sparsity,
            anchor_eta=args.anchor_eta,
        )
        for seed in args.seeds
    ]
    payload = {
        "experiment": "tangent-stages-0-4",
        "summary": summarize_stages(reports),
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
