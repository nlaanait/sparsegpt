"""Small CPU reference experiment for output-tangent-aware pruning.

The toy layer maps sparse, curved input factors to an output activation. The
script compares direct scores, a reweighted ambient OBS heuristic, and a
full-matrix projected-OBS reference under sample-conditioned output tangents.
It does not claim to reproduce a large-model SparseGPT run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_data(seed: int, n_train: int, n_test: int, dim: int = 8):
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
        return x, q, concept, angle

    train = sample(n_train)
    test = sample(n_test)
    return weight, basis, train, test


def hessian(x: np.ndarray, q: np.ndarray | None, *, bias: bool, ridge: float):
    """Build vec_F(W) curvature for ambient or sample-projected squared error."""
    n, input_dim = x.shape
    output_dim = q.shape[1] if q is not None else input_dim
    xa = np.column_stack((x, np.ones(n))) if bias else x
    width = xa.shape[1]
    h = np.zeros((output_dim * width, output_dim * width), dtype=np.float64)
    eye = np.eye(output_dim)
    for i in range(n):
        metric = eye if q is None else np.outer(q[i], q[i])
        h += np.kron(np.outer(xa[i], xa[i]), metric)
    h /= n
    scale = max(float(np.trace(h) / len(h)), 1.0)
    h += ridge * scale * np.eye(len(h))
    return h


def greedy_obs(
    h: np.ndarray,
    weight: np.ndarray,
    sparsity: float,
    *,
    score_weight: np.ndarray | None = None,
    bias: bool,
):
    """Greedy full-matrix OBS with exact quadratic compensation.

    Weight coordinates are eligible for pruning; an optional bias is always
    free. The Hessian may be ambient or sample-projected.
    """
    out_dim, in_dim = weight.shape
    width = in_dim + int(bias)
    total = out_dim * width
    target = np.zeros((out_dim, width), dtype=np.float64)
    target[:, :in_dim] = weight
    target = target.reshape(-1, order="F")
    delta = np.zeros(total, dtype=np.float64)
    free = np.arange(total)
    inv = np.linalg.inv(h)
    eligible_count = int(np.floor(sparsity * weight.size))
    pruned = np.zeros(weight.size, dtype=bool)
    for _ in range(eligible_count):
        is_eligible = free < weight.size
        local_ids = np.flatnonzero(is_eligible)
        denom = np.diag(inv)[local_ids]
        residual = target[free[local_ids]] - delta[free[local_ids]]
        costs = residual**2 / np.maximum(denom, 1e-30)
        if score_weight is not None:
            flat = free[local_ids]
            p, q = flat % out_dim, flat // out_dim
            costs *= score_weight[p, q]
        selected_local = local_ids[int(np.argmin(costs))]
        selected_flat = free[selected_local]
        pruned[selected_flat] = True
        error = target[selected_flat] - delta[selected_flat]
        delta[free] += error * inv[:, selected_local] / inv[selected_local, selected_local]
        keep = np.arange(len(free)) != selected_local
        inv = inv[np.ix_(keep, keep)] - np.outer(
            inv[keep, selected_local], inv[selected_local, keep]
        ) / inv[selected_local, selected_local]
        free = free[keep]

    delta_matrix = delta.reshape((out_dim, width), order="F")
    sparse_weight = weight - delta_matrix[:, :in_dim]
    sparse_weight[pruned[: weight.size].reshape((out_dim, in_dim), order="F")] = 0.0
    # ``evaluate`` adds this to (W - sparse) x, so it is the negative of the
    # fitted model bias ``-delta_bias``.
    bias_correction = delta_matrix[:, in_dim] if bias else np.zeros(out_dim)
    return sparse_weight, bias_correction


def direct_mask(weight: np.ndarray, score: np.ndarray, sparsity: float):
    mask = mask_from_scores(score, sparsity)
    sparse = weight.copy()
    sparse.reshape(-1)[mask] = 0.0
    return sparse


def mask_from_scores(score: np.ndarray, sparsity: float) -> np.ndarray:
    k = int(np.floor(sparsity * score.size))
    mask = np.zeros(score.size, dtype=bool)
    if k:
        mask[np.argpartition(score.reshape(-1), k - 1)[:k]] = True
    return mask


def compensate_fixed_mask(h, weight, prune_mask, *, bias=True):
    """Apply a fixed deletion mask with exact compensation for the quadratic."""
    out_dim, in_dim = weight.shape
    width = in_dim + int(bias)
    target = np.zeros((out_dim, width), dtype=np.float64)
    target[:, :in_dim] = weight
    target = target.reshape(-1, order="F")
    inv = np.linalg.inv(h)
    delta = np.zeros_like(target)
    free = np.arange(len(target))
    selected = np.flatnonzero(prune_mask.reshape(-1, order="F"))
    for selected_flat in selected:
        selected_local = int(np.flatnonzero(free == selected_flat)[0])
        error = target[selected_flat] - delta[selected_flat]
        denominator = inv[selected_local, selected_local]
        delta[free] += error * inv[:, selected_local] / denominator
        keep = np.arange(len(free)) != selected_local
        inv = inv[np.ix_(keep, keep)] - np.outer(
            inv[keep, selected_local], inv[selected_local, keep]
        ) / denominator
        free = free[keep]
    delta_matrix = delta.reshape((out_dim, width), order="F")
    sparse = weight - delta_matrix[:, :in_dim]
    sparse[prune_mask] = 0.0
    residual_offset = delta_matrix[:, in_dim] if bias else np.zeros(out_dim)
    return sparse, residual_offset


def fit_bias(weight: np.ndarray, sparse: np.ndarray, x: np.ndarray, q: np.ndarray | None):
    """Minimize mean (r_n + c)^T M_n (r_n + c) over a constant offset c."""
    residual = ((weight - sparse) @ x.T).T
    out_dim = weight.shape[0]
    lhs = np.zeros((out_dim, out_dim))
    rhs = np.zeros(out_dim)
    for i in range(len(x)):
        metric = np.eye(out_dim) if q is None else np.outer(q[i], q[i])
        lhs += metric
        rhs += metric @ residual[i]
    # Projected bias curvature can be rank deficient. Drop directions whose
    # eigenvalues are negligible relative to the largest observed one instead
    # of amplifying an unobserved bias direction.
    return -(np.linalg.pinv(lhs, rcond=1e-6) @ rhs)


def evaluate(weight, sparse, correction, x, q, readout):
    residual = ((weight - sparse) @ x.T).T + correction[None, :]
    projected = np.sum((q * residual) ** 2, axis=1)
    return {
        "tangent_mse": float(np.mean(projected)),
        "ambient_mse": float(np.mean(np.sum(residual**2, axis=1))),
        "downstream_readout_mse": float(np.mean((residual @ readout) ** 2)),
        "actual_sparsity": float(1 - np.count_nonzero(sparse) / sparse.size),
    }


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def local_vs_global(train, test, weight, basis, delta=0.15):
    x_train, _q_train, g_train, angles_train = train
    x_test, q_test, concepts, angles = test
    # A fixed rank-1 direction per factor is estimated from training tangents.
    global_dirs = []
    for g in range(basis.shape[1] // 2):
        local = weight @ basis[:, 2 * g : 2 * g + 2]
        ix = np.flatnonzero(g_train == g)
        derivs = np.stack(
            [local @ np.array([-np.sin(angles_train[i]), np.cos(angles_train[i])]) for i in ix],
            axis=1,
        )
        u, _, _ = np.linalg.svd(derivs, full_matrices=False)
        global_dirs.append(u[:, 0])
    local_errors, global_errors = [], []
    for g, theta in zip(concepts, angles):
        sl = slice(2 * g, 2 * g + 2)
        local_map = weight @ basis[:, sl]
        radius = 0.65 + 0.15 * g
        x0 = radius * (basis[:, sl] @ np.array([np.cos(theta), np.sin(theta)]))
        x1 = radius * (basis[:, sl] @ np.array([np.cos(theta + delta), np.sin(theta + delta)]))
        displacement = weight @ (x1 - x0)
        q = q_test[len(local_errors)]
        local_capture = (q @ displacement) ** 2
        global_capture = (global_dirs[g] @ displacement) ** 2
        denom = max(float(displacement @ displacement), 1e-30)
        local_errors.append(max(0.0, 1.0 - local_capture / denom))
        global_errors.append(max(0.0, 1.0 - global_capture / denom))
    return float(np.mean(local_errors)), float(np.mean(global_errors))


def bias_and_off_tangent_control(seed=0, n_train=512, n_test=256):
    """Show what a fixed tangent objective can and cannot certify.

    The active tangent is e1. A constant tangent residual can be folded into a
    bias, while an input-dependent e2 residual is invisible to tangent loss but
    remains visible to an e2 downstream readout.
    """
    rng = np.random.default_rng(seed)
    x_train = 1.5 + rng.normal(size=(n_train, 1))
    x_test = 1.5 + rng.normal(size=(n_test, 1))
    q_train = np.tile(np.array([[1.0, 0.0]]), (n_train, 1))
    q_test = np.tile(np.array([[1.0, 0.0]]), (n_test, 1))
    delta_weight = np.array([[0.3], [0.8]])
    sparse = np.zeros_like(delta_weight)
    correction = fit_bias(delta_weight, sparse, x_train, q_train)
    readout = np.array([0.0, 1.0])
    no_bias_result = evaluate(
        delta_weight, sparse, np.zeros(2), x_test, q_test, readout
    )
    fitted_bias_result = evaluate(
        delta_weight, sparse, correction, x_test, q_test, readout
    )
    no_bias_result.pop("actual_sparsity")
    fitted_bias_result.pop("actual_sparsity")
    return {
        "without_bias": no_bias_result,
        "with_projected_bias_fit": fitted_bias_result,
        "bias_correction": correction.tolist(),
        "interpretation": (
            "The fitted bias can remove the constant tangent component. The "
            "input-dependent normal component remains absent from tangent "
            "loss but is visible to the downstream e2 readout."
        ),
    }


def run(seed=7, n_train=512, n_test=256, sparsity=0.5, ridge=1e-7):
    weight, basis, train, test = make_data(seed, n_train, n_test)
    x_train, q_train, _concept_train, _angle_train = train
    x_test, q_test, _concept_test, _angle_test = test
    out_dim, in_dim = weight.shape
    pdiag = q_train**2
    g = (pdiag.T @ (x_train**2)) / np.maximum(np.sum(x_train**2, axis=0)[None, :], 1e-30)
    h_tan_bias = hessian(x_train, q_train, bias=True, ridge=ridge)
    h_tan_no_bias = hessian(x_train, q_train, bias=False, ridge=ridge)
    h_amb_bias = hessian(x_train, None, bias=True, ridge=ridge)
    h_amb_no_bias = hessian(x_train, None, bias=False, ridge=ridge)
    readout = np.random.default_rng(seed + 1).normal(size=out_dim)
    readout /= np.linalg.norm(readout)

    arms = {}
    for name, h, metric, has_bias in [
        ("ambient_obs_bias", h_amb_bias, None, True),
        ("tangent_obs_bias", h_tan_bias, None, True),
        ("option1_reweighted_ambient_obs", h_amb_bias, g, True),
        ("ambient_obs_no_bias", h_amb_no_bias, None, False),
        ("tangent_obs_no_bias", h_tan_no_bias, None, False),
    ]:
        sparse, correction = greedy_obs(
            h, weight, sparsity, score_weight=metric, bias=has_bias
        )
        arms[name] = evaluate(weight, sparse, correction, x_test, q_test, readout)

    x2 = np.mean(x_train**2, axis=0)
    ambient_wanda = weight**2 * x2[None, :]
    tangent_wanda = weight**2 * (pdiag.T @ (x_train**2) / n_train)
    for name, score in [
        ("magnitude", weight**2),
        ("wanda_squared", ambient_wanda),
        ("tangent_wanda_direct", tangent_wanda),
    ]:
        sparse = direct_mask(weight, score, sparsity)
        correction = fit_bias(weight, sparse, x_train, q_train)
        arms[name] = evaluate(weight, sparse, correction, x_test, q_test, readout)

    # Compare score rankings against the exact projected first-knockout OBS loss.
    inv_tan_train = np.linalg.inv(h_tan_bias)
    inv_tan_test = np.linalg.inv(hessian(x_test, q_test, bias=True, ridge=ridge))
    inv_amb_train = np.linalg.inv(h_amb_bias)
    input_covariance = x_train.T @ x_train / n_train
    input_covariance += 0.01 * np.mean(np.diag(input_covariance)) * np.eye(in_dim)
    input_inverse = np.linalg.inv(input_covariance)
    sparsegpt_factor = np.linalg.cholesky(input_inverse).T
    sparsegpt_factor_score = weight**2 / np.diag(sparsegpt_factor)[None, :] ** 2
    tangent_factor_mask = mask_from_scores(
        sparsegpt_factor_score * g, sparsity
    ).reshape(weight.shape)
    tangent_factor_sparse, tangent_factor_correction = compensate_fixed_mask(
        h_amb_bias, weight, tangent_factor_mask, bias=True
    )
    arms["option1_sparsegpt_factor_mask"] = evaluate(
        weight,
        tangent_factor_sparse,
        tangent_factor_correction,
        x_test,
        q_test,
        readout,
    )
    wvec = weight.reshape(-1, order="F")
    tan_train_obs = wvec**2 / np.diag(inv_tan_train)[: weight.size]
    tan_test_obs = wvec**2 / np.diag(inv_tan_test)[: weight.size]
    amb_train_obs = wvec**2 / np.diag(inv_amb_train)[: weight.size]
    heldout_direct_tangent = weight**2 * (
        (q_test**2).T @ (x_test**2)
    ) / n_test
    gvec = g.T.reshape(-1)
    wanda_vec = ambient_wanda.reshape(-1, order="F")
    tangent_wanda_vec = tangent_wanda.reshape(-1, order="F")
    direct_tangent_test_vec = heldout_direct_tangent.reshape(-1, order="F")
    sparsegpt_factor_vec = sparsegpt_factor_score.reshape(-1, order="F")
    score_comparison = {
        "magnitude_vs_heldout_direct_tangent_impact_spearman": spearman(
            wvec**2, direct_tangent_test_vec
        ),
        "wanda_vs_heldout_direct_tangent_impact_spearman": spearman(
            wanda_vec, direct_tangent_test_vec
        ),
        "tangent_wanda_vs_heldout_direct_tangent_impact_spearman": spearman(
            tangent_wanda_vec, direct_tangent_test_vec
        ),
        "magnitude_vs_heldout_tangent_obs_spearman": spearman(wvec**2, tan_test_obs),
        "wanda_vs_heldout_tangent_obs_spearman": spearman(wanda_vec, tan_test_obs),
        "ambient_obs_vs_heldout_tangent_obs_spearman": spearman(amb_train_obs, tan_test_obs),
        "sparsegpt_cholesky_factor_vs_heldout_tangent_obs_spearman": spearman(
            sparsegpt_factor_vec, tan_test_obs
        ),
        "option1_reweighted_sparsegpt_factor_vs_heldout_tangent_obs_spearman": spearman(
            sparsegpt_factor_vec * gvec, tan_test_obs
        ),
        "option1_reweighted_vs_heldout_tangent_obs_spearman": spearman(amb_train_obs * gvec, tan_test_obs),
        "tangent_wanda_vs_heldout_tangent_obs_spearman": spearman(tangent_wanda_vec, tan_test_obs),
        "train_tangent_obs_vs_heldout_tangent_obs_spearman": spearman(tan_train_obs, tan_test_obs),
    }
    local_error, global_error = local_vs_global(train, test, weight, basis)
    return {
        "seed": seed,
        "train_samples": n_train,
        "test_samples": n_test,
        "shape": list(weight.shape),
        "requested_sparsity": sparsity,
        "projection": "one analytic output tangent per active circular factor",
        "obs_ridge": ridge,
        "local_vs_global_rank1_finite_difference_relative_error": {
            "local_tangent": local_error,
            "fixed_global_direction_per_factor": global_error,
        },
        "score_ranking_spearman": score_comparison,
        "arms": arms,
        "bias_and_off_tangent_control": bias_and_off_tangent_control(
            seed + 1000, n_train, n_test
        ),
        "interpretation": (
            "Synthetic reference only. Greedy full-matrix OBS uses the stated "
            "quadratic surrogate and exact quadratic compensation for each selected "
            "coordinate; the resulting mask is not globally optimal. Option 1 "
            "reweights ambient OBS selection but keeps ambient compensation. The "
            "readout is a fixed linear downstream sensitivity probe."
        ),
    }


def summarize(reports):
    """Mean and sample standard deviation across independent synthetic seeds."""
    arms = sorted(reports[0]["arms"])
    arm_metrics = ("tangent_mse", "ambient_mse", "downstream_readout_mse")
    summary = {"seed_count": len(reports), "arms": {}, "score_ranking_spearman": {}}
    for arm in arms:
        summary["arms"][arm] = {}
        for metric in arm_metrics:
            values = np.array([report["arms"][arm][metric] for report in reports])
            summary["arms"][arm][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    for metric in sorted(reports[0]["score_ranking_spearman"]):
        values = np.array(
            [report["score_ranking_spearman"][metric] for report in reports]
        )
        summary["score_ranking_spearman"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    for metric in ("local_tangent", "fixed_global_direction_per_factor"):
        values = np.array(
            [
                report["local_vs_global_rank1_finite_difference_relative_error"][metric]
                for report in reports
            ]
        )
        summary.setdefault("local_vs_global_rank1_finite_difference_relative_error", {})[
            metric
        ] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Independent synthetic seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tangent-pruning-results.json"))
    args = parser.parse_args()
    if args.train_samples < 2 or args.test_samples < 2:
        parser.error("train and test sample counts must each be at least 2")
    if not 0 < args.sparsity < 1 or not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("sparsity must lie in (0, 1) and ridge must be finite/positive")
    seeds = [args.seed] if args.seed is not None else args.seeds
    reports = [
        run(seed, args.train_samples, args.test_samples, args.sparsity, args.ridge)
        for seed in seeds
    ]
    report = {
        "experiment": "synthetic-output-tangent-pruning-reference",
        "summary": summarize(reports),
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
