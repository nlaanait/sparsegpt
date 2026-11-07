"""Revised note §6, Stages 0–4, on one synthetic layer.

CPU only. Stage 1 compares a planar circle (no rank reduction once presence
is counted) with a multi-harmonic curve. Stages 0, 2, 3, and 4 prune one
small linear layer on two orthogonal circles, so the global-BSF rung is the
active plane and the static union is both planes.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sparsifier.prune.inpass_bias import InPassBiasSparseGPT
from sparsifier.prune.input_side import InputSideSparseGPT
from sparsifier.prune.ladder import fisher_sample_weights, row_scales_from_sample_weights, static_union_row_scales
from sparsifier.prune.rescale import prune_with_rescaling, row_scales
from sparsifier.prune.revised_methods import (
    projected_cg_refit,
    sparse_plus_low_rank,
    upstream_sample_weight,
    write_alignment,
)
from sparsifier.prune.sparsegpt import SparseGPT
from sparsifier.prune.stage0 import (
    congruence_relative_error,
    gradient_enrichment,
    leverage_concentration,
    mask_flip_rate,
    mean_principal_angle,
    radial_share,
)

ALPHA_IN_SWEEP = (0.0, 0.3, 1.0, 3.0)
STAGE4_TOLERANCE = 0.10
_D3_NEAR_ONE = 1.5
_D5_ERROR_LIMIT = 0.25
_D6_SHARE_LIMIT = 0.30


def d1_decision(ks: float, n: int) -> str:
    """Skip entrywise output-side scores when the KS statistic is inside the 5% band."""
    if n < 1:
        raise ValueError("n must be positive")
    critical = 1.36 / math.sqrt(n)
    if ks < critical:
        return "skip_entrywise"
    return "use_entrywise"


def d3_decision(kappa_tangent: float, kappa_presence: float) -> str:
    """H0 holds when either concept family is enriched over the isotropic null."""
    if kappa_tangent < _D3_NEAR_ONE and kappa_presence < _D3_NEAR_ONE:
        return "concept_metrics_only"
    return "h0_holds"


def h1_disconfirmed(*, unstable: bool, local_error: float, global_error: float) -> bool:
    """Disconfirm H1 when charts move or fail to beat a matched global span."""
    return bool(unstable or local_error >= global_error)


def stage4_within_tolerance(arm: dict, baseline: dict, tolerance: float = STAGE4_TOLERANCE) -> bool:
    """Held-out ambient and presence error stay within a relative tolerance of no-bias."""
    limit = 1.0 + float(tolerance)
    return bool(
        arm["ambient"] <= baseline["ambient"] * limit and arm["presence"] <= baseline["presence"] * limit
    )


def run_revised_protocol(
    *,
    seed: int = 0,
    n_train: int = 32,
    n_test: int = 16,
    sparsity: float = 0.5,
    dim: int = 16,
) -> dict:
    """Run Stages 0–4 and return the note's measurements and decision rules."""
    if dim < 8 or dim % 4 != 0:
        raise ValueError("dim must be a multiple of 4 and at least 8")
    data = _pruning_batch(n_train + n_test, dim, seed)
    split = _split(data, n_train, seed)
    weight = _weight(dim, seed)
    frames = _output_frames(weight, split)
    bundle = {
        "dim": dim,
        "sparsity": float(sparsity),
        "weight": weight,
        "split": split,
        "frames": frames,
        "seed": seed,
    }
    started = time.perf_counter()
    tangent_started = time.perf_counter()
    stage1 = _stage1(n_train, n_test, dim, seed)
    tangent_s = time.perf_counter() - tangent_started
    stage3 = _stage3(bundle)
    stage0 = _stage0(bundle, stage1, stage3)
    stage2 = _stage2(bundle)
    stage4 = _stage4(bundle, stage3)
    stage3.pop("cache", None)
    return {
        "seed": seed,
        "dim": dim,
        "sparsity": float(sparsity),
        "train_samples": n_train,
        "test_samples": n_test,
        "runtime_s": time.perf_counter() - started,
        "tangent_estimation_s": tangent_s,
        "activation_bytes": int(bundle["split"]["train_phi"].numel() + bundle["weight"].numel()) * 4,
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
    }


def _stage0(bundle: dict, stage1: dict, stage3: dict) -> dict:
    dim = bundle["dim"]
    train = bundle["frames"]["train"]
    tangent = torch.tensor(train["tangent"], dtype=torch.float32)
    presence = torch.tensor(train["presence"], dtype=torch.float32)
    ks_values = []
    for row_t, row_p in zip(tangent, presence):
        leverages = row_t.square() + row_p.square()
        ks_values.append(leverage_concentration(leverages, rank=2)["ks_statistic"])
    ks = float(np.mean(ks_values))
    d1 = {"ks_statistic": ks, "n": dim, "decision": d1_decision(ks, dim)}

    vanilla = stage3["cache"]["vanilla_mask"]
    variant_b = stage3["cache"]["local_mask"]
    null_mask = stage3["cache"]["null_mask"]
    flip = mask_flip_rate(vanilla, variant_b)
    null_flip = mask_flip_rate(vanilla, null_mask)
    d2_decision = "output_side_cannot_matter" if abs(flip - null_flip) < 0.05 else "output_side_moves_masks"
    d2 = {"flip_rate": flip, "null_flip_rate": null_flip, "decision": d2_decision}

    gradients = torch.tensor(train["gradient"], dtype=torch.float32)
    kappa_t = gradient_enrichment(gradients, _outer(tangent), 1.0)
    kappa_p = gradient_enrichment(gradients, _outer(presence), 1.0)
    d3 = {
        "kappa_tangent": kappa_t,
        "kappa_presence": kappa_p,
        "gradient_source": "concept_aligned_probe",
        "decision": d3_decision(kappa_t, kappa_p),
    }

    family = stage1["planar_circle"]
    harmonic = stage1["multi_harmonic"]
    d4 = {
        "mean_principal_angle": family["mean_principal_angle"],
        "null_principal_angle": family["null_principal_angle"],
        "harmonic_mean_principal_angle": harmonic["mean_principal_angle"],
        "R": int(bundle["frames"]["union_rank"]),
        "spectrum_head": bundle["frames"]["spectrum_head"],
        "decision": "stop_unstable" if family["unstable"] else "continue",
    }

    d5_error, d5_spearman = _congruence_check(
        bundle["weight"], bundle["split"]["train_phi"], tangent
    )
    d5 = {
        "congruence_error": d5_error,
        "obs_cost_spearman": d5_spearman,
        "decision": "use_exact_refit" if d5_error > _D5_ERROR_LIMIT else "congruence_ok",
    }

    shares = [
        radial_share(presence[i], tangent[i])
        for i in range(presence.shape[0])
    ]
    knob_shares = [
        radial_share(tangent[i], tangent[i])
        for i in range(min(8, tangent.shape[0]))
    ]
    share = float(np.mean(shares))
    d6 = {
        "radial_share": share,
        "knob_radial_share": float(np.mean(knob_shares)),
        "decision": "presence_required" if share > _D6_SHARE_LIMIT else "tangent_only",
    }
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}


def _stage1(n_train: int, n_test: int, dim: int, seed: int) -> dict:
    return {
        "planar_circle": _evaluate_family("planar_circle", n_train, n_test, dim, seed),
        "multi_harmonic": _evaluate_family("multi_harmonic", n_train, n_test, dim, seed + 1),
    }


def _evaluate_family(kind: str, n_train: int, n_test: int, dim: int, seed: int) -> dict:
    curve = _curve_points(kind, n_train + n_test, dim)
    order = np.random.default_rng(seed).permutation(n_train + n_test)
    train = order[:n_train]
    test = order[n_train:]
    x_train = curve["x"][train]
    x_test = curve["x"][test]
    budget = curve["analytic_r"] + 1
    local_bases = []
    for index in range(x_test.shape[0]):
        nearest = int(np.argmin(np.linalg.norm(x_train - x_test[index], axis=1)))
        basis, _ = _local_basis(x_train, nearest, k=min(8, x_train.shape[0]))
        local_bases.append(basis[:budget])
    global_basis, estimated_b = _svd_basis(x_train)
    mean = x_train.mean(axis=0)
    local_error = _reconstruction_error(mean, x_test, local_bases)
    matched = [global_basis[:budget] for _ in range(x_test.shape[0])]
    global_error = _reconstruction_error(mean, x_test, matched)
    angle, null_angle = _stability(x_train, seed)
    unstable = bool(angle > 0.5 * null_angle)
    local_ranks = [
        _local_basis(x_train, i, k=min(8, x_train.shape[0]))[1]
        for i in range(0, x_train.shape[0], max(1, x_train.shape[0] // 8))
    ]
    return {
        "analytic_b": curve["analytic_b"],
        "analytic_r": curve["analytic_r"],
        "rank_reduction_possible": curve["rank_reduction_possible"],
        "estimated_b": estimated_b,
        "estimated_r": int(np.median(local_ranks)),
        "matched_budget": budget,
        "heldout_local_error": local_error,
        "heldout_global_error": global_error,
        "mean_principal_angle": angle,
        "null_principal_angle": null_angle,
        "unstable": unstable,
        "h1_disconfirmed": h1_disconfirmed(
            unstable=unstable, local_error=local_error, global_error=global_error
        ),
    }


def _stage2(bundle: dict) -> dict:
    dense = bundle["weight"]
    train = bundle["split"]
    test = bundle["frames"]["test"]
    phi = train["train_phi"]
    hessian, hinv = _damped_inverse(phi)
    diag = torch.diag(hinv).clamp_min(1e-12)
    x2 = phi.square().mean(dim=0)
    coupling = _coupling(torch.tensor(bundle["frames"]["train"]["tangent"]).square(), phi)
    gradients = torch.tensor(bundle["frames"]["train"]["gradient"], dtype=torch.float32)
    fisher = dense.square() * (gradients.square().T @ phi.square())
    scores = {
        "magnitude": dense.square(),
        "wanda": dense.square() * x2.reshape(1, -1),
        "ambient_obs": dense.square() / diag.reshape(1, -1),
        "fisher": fisher,
        "geometric": dense.square() * coupling * x2.reshape(1, -1),
    }
    generator = np.random.default_rng(bundle["seed"] + 11)
    entries = [
        (int(p), int(q))
        for p, q in zip(
            generator.integers(0, dense.shape[0], size=16),
            generator.integers(0, dense.shape[1], size=16),
        )
    ]
    observed = []
    downstream = []
    predicted = {name: [] for name in scores}
    for row, col in entries:
        compensated = _obs_knockout(dense, hinv, row, col)
        observed.append(
            _errors(
                dense,
                compensated,
                bundle["split"]["test_phi"],
                test["tangent"],
                test["presence"],
            )["tangent"]
        )
        downstream.append(
            _downstream(dense, compensated, bundle["split"]["test_phi"], bundle["frames"]["probe"])
        )
        for name, matrix in scores.items():
            predicted[name].append(float(matrix[row, col]))
    spearman = {name: _spearman(predicted[name], observed) for name in scores}
    edit = _editability(bundle)
    selective = _column_knockout(bundle, by="alignment")
    broad = _column_knockout(bundle, by="magnitude")
    return {
        "spearman_vs_compensated_tangent": spearman,
        "editability_spearman": edit,
        "downstream": {
            "probe_mse": float(np.mean([row["probe_mse"] for row in downstream])),
            "kl_to_dense": float(np.mean([row["kl_to_dense"] for row in downstream])),
        },
        "selective_vs_broad": {"selective": selective, "broad": broad},
    }


def _stage3(bundle: dict) -> dict:
    dense = bundle["weight"]
    phi = bundle["split"]["train_phi"]
    deltas = bundle["split"]["deltas"]
    knobs = bundle["split"]["knobs"]
    sparsity = bundle["sparsity"]
    dim = bundle["dim"]
    test_phi = bundle["split"]["test_phi"]
    test = bundle["frames"]["test"]

    def evaluate(hat: torch.Tensor) -> dict:
        stats = _errors(dense, hat, test_phi, test["tangent"], test["presence"])
        stats["actual_sparsity"] = float((hat == 0).to(dtype=torch.float32).mean())
        stats["per_concept"] = _per_concept(
            dense,
            hat,
            test_phi,
            test["tangent"],
            test["presence"],
            bundle["split"]["test_concept"],
        )
        stats["downstream"] = _downstream(dense, hat, test_phi, bundle["frames"]["probe"])
        return stats

    vanilla = _ambient_prune(dense, phi, sparsity)
    local_hat, local_mask = _scaled_prune(dense, phi, bundle["frames"]["local_scale"], sparsity)
    null_hat, null_mask = _scaled_prune(dense, phi, bundle["frames"]["null_scale"], sparsity)
    cache = {
        "vanilla": vanilla,
        "vanilla_mask": vanilla != 0,
        "local_mask": local_mask,
        "null_mask": null_mask,
    }

    variant_a = {}
    for pattern, zeros, group in (("unstructured", 0, 0), ("nm_2_4", 2, 4)):
        points = []
        for alpha in ALPHA_IN_SWEEP:
            hat = _variant_a_prune(dense, phi, deltas, knobs, alpha, sparsity, zeros, group)
            point = evaluate(hat)
            point["alpha_in"] = alpha
            points.append(point)
        variant_a[pattern] = points

    ladder = {
        "ambient": evaluate(vanilla),
        "static_union": evaluate(_scaled_prune(dense, phi, bundle["frames"]["static_scale"], sparsity)[0]),
        "global_bsf": evaluate(_scaled_prune(dense, phi, bundle["frames"]["bsf_scale"], sparsity)[0]),
        "local_tangents": evaluate(local_hat),
        "tangents_and_presence": evaluate(
            _scaled_prune(dense, phi, bundle["frames"]["geo_scale"], sparsity)[0]
        ),
        "fisher": evaluate(_scaled_prune(dense, phi, bundle["frames"]["fisher_scale"], sparsity)[0]),
    }
    static_hat = _scaled_prune(dense, phi, bundle["frames"]["static_scale"], sparsity)[0]
    exactness = _exactness(dense, static_hat, phi, bundle["frames"]["metric"], test_phi, test)
    upstream = _upstream(bundle)
    variant_d = _variant_d(bundle, evaluate)
    return {
        "variant_a": variant_a,
        "ladder": ladder,
        "exactness": exactness,
        "upstream": upstream,
        "variant_d": variant_d,
        "cache": cache,
    }


def _stage4(bundle: dict, stage3: dict) -> dict:
    dense = bundle["weight"]
    phi = bundle["split"]["train_phi"]
    test_phi = bundle["split"]["test_phi"]
    test = bundle["frames"]["test"]
    sparsity = bundle["sparsity"]
    vanilla = stage3["cache"]["vanilla"]

    def pack(hat: torch.Tensor, bias: torch.Tensor | None) -> dict:
        stats = _errors(dense, hat, test_phi, test["tangent"], test["presence"], bias=bias)
        stats["presence_residual"] = stats["presence"]
        stats["normal_residual"] = stats["normal"]
        return stats

    none = pack(vanilla, None)
    centered_hat = _centered_prune(dense, phi, sparsity)
    centered_bias = _mean_residual(dense, centered_hat, phi)
    ones_hat, ones_bias = _ones_prune(dense, phi, sparsity)
    posthoc_bias = _mean_residual(dense, vanilla, phi)
    arms = {
        "no_bias": none,
        "centered": pack(centered_hat, centered_bias),
        "ones_column": pack(ones_hat, ones_bias),
        "posthoc_global_mean": pack(vanilla, posthoc_bias),
    }
    for name, arm in arms.items():
        arm["within_tolerance"] = stage4_within_tolerance(arm, none, STAGE4_TOLERANCE)
        arm["input_dependent"] = arm["normal_cv"] > 0.25
    return {
        "tolerance": STAGE4_TOLERANCE,
        "chart_assignment_agreement": _chart_agreement(bundle),
        "arms": arms,
    }


def _exactness(dense, static_hat, phi, metric, test_phi, test) -> dict:
    hessian = phi.T @ phi / phi.shape[0]
    keep = static_hat != 0
    ambient = projected_cg_refit(dense, keep, hessian)
    cross = projected_cg_refit(dense, keep, hessian, metric=metric)

    def evaluate(hat: torch.Tensor) -> dict:
        return _errors(dense, hat, test_phi, test["tangent"], test["presence"])

    return {
        "row_diagonal": evaluate(static_hat),
        "projected_cg": evaluate(ambient),
        "cross_row": evaluate(cross),
    }


def _upstream(bundle: dict) -> dict:
    dense = bundle["weight"]
    phi = bundle["split"]["train_phi"]
    test_phi = bundle["split"]["test_phi"]
    dim = bundle["dim"]
    generator = torch.Generator().manual_seed(bundle["seed"] + 3)
    w_up = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)
    w_gate = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)
    projector = bundle["frames"]["projector"]
    gamma = bundle["frames"]["gamma_geo"]
    hidden_up = _hidden_factors(phi, w_gate, w_up)
    omega_up = _neuron_weights(hidden_up["silu"], dense, projector, gamma)
    omega_gate = _neuron_weights(hidden_up["gate_factor"], dense, projector, gamma)
    hat_up = _scaled_prune(w_up, phi, row_scales_from_sample_weights(omega_up, phi), bundle["sparsity"])[0]
    hat_gate = _scaled_prune(
        w_gate, phi, row_scales_from_sample_weights(omega_gate, phi), bundle["sparsity"]
    )[0]
    test = bundle["frames"]["test"]
    up_residual = _block_output(test_phi, dense, w_up, w_gate) - _block_output(test_phi, dense, hat_up, w_gate)
    gate_residual = _block_output(test_phi, dense, w_up, w_gate) - _block_output(
        test_phi, dense, w_up, hat_gate
    )
    removed = max(1, dim // 4)
    by_alignment = _drop_neurons(bundle, _neuron_order(bundle, "alignment")[:removed])
    by_magnitude = _drop_neurons(bundle, _neuron_order(bundle, "magnitude")[:removed])
    return {
        "up": _from_residual(up_residual, test["tangent"], test["presence"]),
        "gate": _from_residual(gate_residual, test["tangent"], test["presence"]),
        "whole_neuron": {
            "by_alignment": by_alignment,
            "by_magnitude": by_magnitude,
            "neurons_removed": removed,
        },
    }


def _variant_d(bundle: dict, evaluate) -> dict:
    dim = bundle["dim"]
    budget = 0.5 * dim * dim
    rank = 1
    for candidate in range(1, min(dim, 8) + 1):
        if candidate * 2 * dim <= budget / 2.0:
            rank = candidate
    extra = rank * 2 * dim
    nnz_cap = int(math.floor(budget - extra))
    basis = bundle["frames"]["union_basis"][:, :rank]
    sparse = _match_nnz(
        _ambient_prune(bundle["weight"], bundle["split"]["train_phi"], 1.0 - nnz_cap / (dim * dim)),
        nnz_cap,
    )
    hat = sparse_plus_low_rank(bundle["weight"], sparse, basis)
    stats = evaluate(hat)
    used = int((sparse != 0).sum()) + extra
    stats.update(
        {
            "rank": rank,
            "storage_budget": float(budget),
            "storage_used": float(used),
            "sparse_nonzeros": int((sparse != 0).sum()),
        }
    )
    return stats


def _curve_points(kind: str, n: int, dim: int) -> dict:
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    if kind == "planar_circle":
        raw = np.stack((np.cos(theta), np.sin(theta)), axis=1)
        deriv = np.stack((-np.sin(theta), np.cos(theta)), axis=1)
        width = 2
        analytic_b, analytic_r = 2, 1
    elif kind == "multi_harmonic":
        raw = np.stack(
            (np.cos(theta), np.sin(theta), 0.5 * np.cos(2 * theta), 0.5 * np.sin(2 * theta)),
            axis=1,
        )
        deriv = np.stack(
            (-np.sin(theta), np.cos(theta), -np.sin(2 * theta), np.cos(2 * theta)),
            axis=1,
        )
        width = 4
        analytic_b, analytic_r = 4, 1
    else:
        raise ValueError("kind must be planar_circle or multi_harmonic")
    axes = np.eye(dim)[:, :width]
    points = raw @ axes.T
    tangent = _unit_rows(deriv @ axes.T)
    presence = _unit_rows(points - tangent * np.sum(points * tangent, axis=1, keepdims=True))
    return {
        "x": points,
        "tangent": tangent,
        "presence": presence,
        "analytic_b": analytic_b,
        "analytic_r": analytic_r,
        "rank_reduction_possible": analytic_b > analytic_r + 1,
    }


def _pruning_batch(n: int, dim: int, seed: int) -> dict:
    first = _curve_points("planar_circle", n // 2, dim)
    second = _curve_points("planar_circle", n - n // 2, dim)
    swap = np.arange(dim)
    swap[0], swap[2] = 2, 0
    swap[1], swap[3] = 3, 1
    for key in ("x", "tangent", "presence"):
        second[key] = second[key][:, swap]
    generator = np.random.default_rng(seed)
    points = np.concatenate((first["x"], second["x"]), axis=0)
    points = points + 0.02 * generator.normal(size=points.shape)
    tangent = np.concatenate((first["tangent"], second["tangent"]), axis=0)
    presence = np.concatenate((first["presence"], second["presence"]), axis=0)
    concept = np.concatenate(
        (np.zeros(first["x"].shape[0], dtype=np.int64), np.ones(second["x"].shape[0], dtype=np.int64))
    )
    gradient_noise = _unit_rows(generator.normal(size=tangent.shape))
    return {
        "x": points,
        "tangent_in": tangent,
        "presence_in": presence,
        "concept": concept,
        "gradient_noise": gradient_noise,
    }


def _split(data: dict, n_train: int, seed: int) -> dict:
    order = np.random.default_rng(seed + 5).permutation(data["x"].shape[0])
    train = order[:n_train]
    test = order[n_train:]
    phi_train = torch.tensor(data["x"][train], dtype=torch.float32)
    return {
        "train_index": train,
        "test_index": test,
        "train_phi": phi_train,
        "test_phi": torch.tensor(data["x"][test], dtype=torch.float32),
        "deltas": torch.tensor(0.05 * data["tangent_in"][train], dtype=torch.float32),
        "knobs": torch.tensor(data["concept"][train], dtype=torch.long),
        "train_tangent_in": data["tangent_in"][train],
        "test_tangent_in": data["tangent_in"][test],
        "train_presence_in": data["presence_in"][train],
        "test_presence_in": data["presence_in"][test],
        "train_concept": data["concept"][train],
        "test_concept": data["concept"][test],
        "train_noise": data["gradient_noise"][train],
        "test_noise": data["gradient_noise"][test],
    }


def _output_frames(weight: torch.Tensor, split: dict) -> dict:
    def push(tangent_in, presence_in, noise):
        matrix = weight.detach().cpu().numpy()
        tangent = _unit_rows(tangent_in @ matrix.T)
        raw = presence_in @ matrix.T
        raw = raw - tangent * np.sum(raw * tangent, axis=1, keepdims=True)
        presence = _unit_rows(raw)
        gradient = 0.85 * tangent + 0.15 * noise
        return {"tangent": tangent, "presence": presence, "gradient": gradient}

    train = push(split["train_tangent_in"], split["train_presence_in"], split["train_noise"])
    test = push(split["test_tangent_in"], split["test_presence_in"], split["test_noise"])
    dim = weight.shape[0]
    mean_p = np.zeros((dim, dim), dtype=np.float64)
    for row_t, row_p in zip(train["tangent"], train["presence"]):
        mean_p += np.outer(row_t, row_t) + np.outer(row_p, row_p)
    mean_p /= train["tangent"].shape[0]
    eigvals, evecs = np.linalg.eigh(mean_p)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    evecs = evecs[:, order]
    rank = max(1, int(np.sum(eigvals > 1.0 / dim)))
    basis = torch.tensor(evecs[:, :rank], dtype=torch.float32)
    projector = basis @ basis.T
    gamma_tangent = float(dim)
    gamma_geo = float(dim) / 2.0
    phi = split["train_phi"]
    local_g = _coupling(torch.tensor(train["tangent"]).square(), phi)
    geo_lev = torch.tensor(train["tangent"]).square() + torch.tensor(train["presence"]).square()
    geo_g = _coupling(geo_lev, phi)
    bsf_lev = _bsf_leverages(weight, split["train_concept"], dim)
    bsf_g = _coupling(bsf_lev, phi)
    static_lev = torch.diag(projector).clamp_min(0.0)
    gradients = torch.tensor(train["gradient"], dtype=torch.float32)
    return {
        "train": train,
        "test": test,
        "union_rank": rank,
        "union_basis": basis,
        "projector": projector,
        "metric": torch.eye(dim) + gamma_geo * projector,
        "spectrum_head": [float(value) for value in eigvals[:4]],
        "gamma_geo": gamma_geo,
        "local_scale": row_scales(local_g, gamma_tangent),
        "null_scale": row_scales(_coupling(torch.tensor(_unit_rows(
            np.random.default_rng(0).normal(size=train["tangent"].shape)
        )).square(), phi), gamma_tangent),
        "geo_scale": row_scales(geo_g, gamma_geo),
        "bsf_scale": row_scales(bsf_g, gamma_geo),
        "static_scale": static_union_row_scales(static_lev, gamma_geo, dim),
        "fisher_scale": row_scales_from_sample_weights(fisher_sample_weights(gradients, 1.0), phi),
        "probe": torch.tensor(train["tangent"][0], dtype=torch.float32),
    }


def _bsf_leverages(weight: torch.Tensor, concept: np.ndarray, dim: int) -> torch.Tensor:
    planes = {
        0: weight[:, [0, 1]],
        1: weight[:, [2, 3]],
    }
    leverages = torch.zeros(concept.shape[0], dim, dtype=torch.float32)
    cache: dict[int, torch.Tensor] = {}
    for index, label in enumerate(concept.tolist()):
        if label not in cache:
            basis, _ = torch.linalg.qr(planes[label].float(), mode="reduced")
            cache[label] = basis.square().sum(dim=1)
        leverages[index] = cache[label]
    return leverages


def _congruence_check(weight: torch.Tensor, phi: torch.Tensor, tangent: torch.Tensor) -> tuple[float, float]:
    count = phi.shape[0]
    hessian = phi.T @ phi / count
    gamma = float(phi.shape[1])
    errors = []
    spearmans = []
    damp = 0.01 * torch.diag(hessian).mean()
    eye = torch.eye(hessian.shape[0])
    for row in range(min(3, weight.shape[0])):
        omega = 1.0 + gamma * tangent[:, row].square()
        row_hessian = (phi * omega[:, None]).T @ phi / count
        diagonal_ratio = torch.diag(row_hessian) / torch.diag(hessian).clamp_min(1e-12)
        scale = torch.sqrt(diagonal_ratio.clamp_min(0.0))
        errors.append(congruence_relative_error(row_hessian, hessian, scale))
        congruent = scale[:, None] * hessian * scale[None, :]
        true_cost = _obs_costs(weight[row], row_hessian + damp * eye)
        approx_cost = _obs_costs(weight[row], congruent + damp * eye)
        spearmans.append(_spearman(true_cost, approx_cost))
    return float(np.mean(errors)), float(np.mean(spearmans))


def _obs_costs(row: torch.Tensor, hessian: torch.Tensor) -> list[float]:
    inverse = torch.linalg.inv(0.5 * (hessian + hessian.T))
    costs = row.detach().float().square() / torch.diag(inverse).clamp_min(1e-12)
    return [float(value) for value in costs]


def _stability(points: np.ndarray, seed: int) -> tuple[float, float]:
    left = points[0::2]
    right = points[1::2]
    if left.shape[0] < 4 or right.shape[0] < 4:
        return 0.0, math.pi / 2
    angles = []
    probes = np.linspace(0, left.shape[0] - 1, num=min(8, left.shape[0]), dtype=int)
    for index in probes:
        nearest = int(np.argmin(np.linalg.norm(right - left[int(index)], axis=1)))
        angles.append(
            mean_principal_angle(
                torch.tensor(_chord_tangent(left, int(index)).reshape(-1, 1)),
                torch.tensor(_chord_tangent(right, nearest).reshape(-1, 1)),
            )
        )
    generator = np.random.default_rng(seed + 9)
    nulls = []
    for _ in range(8):
        nulls.append(
            mean_principal_angle(
                torch.tensor(generator.normal(size=(points.shape[1], 1))),
                torch.tensor(generator.normal(size=(points.shape[1], 1))),
            )
        )
    return float(np.mean(angles)), float(np.mean(nulls))


def _editability(bundle: dict) -> float:
    dense = bundle["weight"]
    phi = bundle["split"]["train_phi"]
    test_phi = bundle["split"]["test_phi"]
    generator = torch.Generator().manual_seed(bundle["seed"] + 13)
    w_up = torch.randn(dense.shape[0], dense.shape[1], generator=generator) / math.sqrt(dense.shape[0])
    w_gate = torch.randn_like(w_up)
    factors = _hidden_factors(test_phi, w_gate, w_up)
    predicted = []
    actual = []
    rng = np.random.default_rng(bundle["seed"] + 17)
    for _ in range(8):
        row = int(rng.integers(0, w_up.shape[0]))
        col = int(rng.integers(0, w_up.shape[1]))
        score = abs(float(w_up[row, col])) * float(factors["silu"][:, row].square().mean()) * float(
            test_phi[:, col].square().mean()
        ) * float(dense[:, row].square().sum())
        epsilon = 1e-2
        bumped = w_up.clone()
        bumped[row, col] += epsilon
        before = _block_output(test_phi, dense, w_up, w_gate)
        after = _block_output(test_phi, dense, bumped, w_gate)
        predicted.append(score)
        actual.append(float((after - before).square().sum(dim=1).mean()))
    return _spearman(predicted, actual)


def _column_knockout(bundle: dict, *, by: str) -> dict:
    order = _neuron_order(bundle, "alignment" if by == "alignment" else "magnitude")
    column = int(order[0])
    hat = bundle["weight"].clone()
    hat[:, column] = 0
    stats = _errors(
        bundle["weight"],
        hat,
        bundle["split"]["test_phi"],
        bundle["frames"]["test"]["tangent"],
        bundle["frames"]["test"]["presence"],
    )
    stats["column"] = column
    if by == "alignment":
        stats["chi"] = float(
            write_alignment(bundle["weight"][:, column], bundle["frames"]["projector"])
        )
    return stats


def _neuron_order(bundle: dict, kind: str) -> list[int]:
    dense = bundle["weight"]
    phi = bundle["split"]["train_phi"]
    if kind == "alignment":
        scores = [
            write_alignment(dense[:, column], bundle["frames"]["projector"])
            for column in range(dense.shape[1])
        ]
        return sorted(range(dense.shape[1]), key=lambda column: scores[column])
    ambient = dense.square().sum(dim=0) * phi.square().mean(dim=0)
    return sorted(range(dense.shape[1]), key=lambda column: float(ambient[column]))


def _drop_neurons(bundle: dict, columns: list[int]) -> dict:
    hat = bundle["weight"].clone()
    hat[:, columns] = 0
    stats = _errors(
        bundle["weight"],
        hat,
        bundle["split"]["test_phi"],
        bundle["frames"]["test"]["tangent"],
        bundle["frames"]["test"]["presence"],
    )
    stats["columns"] = columns
    return stats


def _hidden_factors(phi: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> dict:
    gate = phi @ w_gate.T
    up = phi @ w_up.T
    silu = F.silu(gate)
    sigmoid = torch.sigmoid(gate)
    silu_prime = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    return {"silu": silu, "gate_factor": silu_prime * up}


def _neuron_weights(factor: torch.Tensor, w_down: torch.Tensor, projector: torch.Tensor, gamma: float) -> torch.Tensor:
    columns = [
        upstream_sample_weight(factor[:, index], w_down[:, index], projector, gamma)
        for index in range(w_down.shape[1])
    ]
    return torch.stack(columns, dim=1)


def _block_output(phi, w_down, w_up, w_gate) -> torch.Tensor:
    gate = F.silu(phi @ w_gate.T)
    up = phi @ w_up.T
    return (gate * up) @ w_down.T


def _ambient_prune(weight: torch.Tensor, phi: torch.Tensor, sparsity: float) -> torch.Tensor:
    layer = _linear(weight)
    solver = SparseGPT(layer)
    solver.add_batch(phi)
    solver.fasterprune(sparsity, blocksize=weight.shape[1], percdamp=0.01)
    return layer.weight.detach().clone()


def _variant_a_prune(weight, phi, deltas, knobs, alpha, sparsity, zeros, group) -> torch.Tensor:
    layer = _linear(weight)
    solver = InputSideSparseGPT(layer)
    solver.add_batch(phi)
    solver.add_pairs(deltas, knob_ids=knobs)
    solver.finalize_pairs(alpha)
    solver.fasterprune(
        sparsity,
        zeros_per_group=zeros,
        group_size=group,
        blocksize=weight.shape[1],
        percdamp=0.01,
    )
    return layer.weight.detach().clone()


def _scaled_prune(weight, phi, scale, sparsity) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _linear(weight)
    solver = SparseGPT(layer)
    solver.add_batch(phi)
    prune_with_rescaling(solver, scale, sparsity, blocksize=weight.shape[1], percdamp=0.01)
    hat = layer.weight.detach().clone()
    return hat, hat != 0


def _centered_prune(weight, phi, sparsity) -> torch.Tensor:
    layer = _linear(weight, bias=True)
    solver = InPassBiasSparseGPT(layer, mode="centered")
    solver.add_batch(phi)
    solver.fasterprune(sparsity, blocksize=weight.shape[1], percdamp=0.01)
    return layer.weight.detach().clone()


def _ones_prune(weight, phi, sparsity) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _linear(weight, bias=True)
    solver = InPassBiasSparseGPT(layer, mode="ones_column")
    solver.add_batch(phi)
    solver.fasterprune(sparsity, blocksize=weight.shape[1], percdamp=0.01)
    return layer.weight.detach().clone(), layer.bias.detach().clone()


def _linear(weight: torch.Tensor, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias:
            layer.bias.zero_()
    return layer


def _match_nnz(weight: torch.Tensor, nnz_cap: int) -> torch.Tensor:
    hat = weight.clone()
    flat = hat.reshape(-1)
    keep = (flat != 0).nonzero(as_tuple=False).flatten()
    extra = int(keep.numel() - nnz_cap)
    if extra > 0:
        smallest = torch.topk(flat[keep].abs(), extra, largest=False).indices
        flat[keep[smallest]] = 0
    return hat


def _mean_residual(dense: torch.Tensor, hat: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return ((dense - hat) @ phi.T).mean(dim=1)


def _errors(dense, hat, phi, tangent, presence, bias=None) -> dict:
    residual = ((dense - hat) @ phi.T).T
    if bias is not None:
        residual = residual - bias.reshape(1, -1)
    return _from_residual(residual, tangent, presence)


def _downstream(dense, hat, phi, probe: torch.Tensor) -> dict:
    dense_h = phi @ dense.T
    hat_h = phi @ hat.T
    probe_error = ((hat_h - dense_h) @ probe.reshape(-1)).square().mean()
    log_dense = torch.log_softmax(dense_h, dim=1)
    log_hat = torch.log_softmax(hat_h, dim=1)
    kl = torch.nn.functional.kl_div(log_hat, log_dense, log_target=True, reduction="batchmean")
    return {"probe_mse": float(probe_error), "kl_to_dense": float(kl)}


def _per_concept(dense, hat, phi, tangent, presence, concept) -> dict:
    report = {}
    for label in (0, 1):
        mask = torch.tensor(concept == label)
        if int(mask.sum()) == 0:
            continue
        report[str(label)] = _errors(dense, hat, phi[mask], tangent[mask], presence[mask])
    return report


def _chart_agreement(bundle: dict) -> float:
    train = _chart_ids(bundle["split"]["train_phi"].numpy(), bundle["split"]["train_concept"])
    test_phi = bundle["split"]["test_phi"].numpy()
    test = _chart_ids(test_phi, bundle["split"]["test_concept"])
    train_phi = bundle["split"]["train_phi"].numpy()
    matches = []
    for row, chart in zip(test_phi, test):
        nearest = int(np.argmin(np.linalg.norm(train_phi - row, axis=1)))
        matches.append(float(train[nearest] == chart))
    return float(np.mean(matches)) if matches else 0.0


def _chart_ids(points: np.ndarray, concept: np.ndarray) -> np.ndarray:
    ids = []
    for row, label in zip(points, concept.tolist()):
        if int(label) == 0:
            angle = math.atan2(row[1], row[0])
        else:
            angle = math.atan2(row[3], row[2])
        bin_id = int((angle + math.pi) / (math.pi / 2.0)) % 4
        ids.append(int(label) * 4 + bin_id)
    return np.asarray(ids)


def _from_residual(residual: torch.Tensor, tangent, presence) -> dict:
    direction_t = torch.tensor(tangent, dtype=residual.dtype)
    direction_p = torch.tensor(presence, dtype=residual.dtype)
    ambient = residual.square().sum(dim=1).mean()
    tangent_error = (residual * direction_t).sum(dim=1).square().mean()
    presence_error = (residual * direction_p).sum(dim=1).square().mean()
    projected = direction_t * (residual * direction_t).sum(dim=1, keepdim=True)
    projected = projected + direction_p * (residual * direction_p).sum(dim=1, keepdim=True)
    normal_per = (residual - projected).square().sum(dim=1)
    normal = normal_per.mean()
    mean_normal = float(normal)
    cv = 0.0 if mean_normal <= 0.0 else float(normal_per.std(unbiased=False)) / mean_normal
    return {
        "ambient": float(ambient),
        "tangent": float(tangent_error),
        "presence": float(presence_error),
        "normal": mean_normal,
        "normal_cv": cv,
    }


def _damped_inverse(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hessian = phi.T @ phi / phi.shape[0]
    damp = 0.01 * torch.diag(hessian).mean()
    hessian = hessian + damp * torch.eye(hessian.shape[0])
    hessian = 0.5 * (hessian + hessian.T)
    return hessian, torch.linalg.inv(hessian)


def _obs_knockout(dense: torch.Tensor, hinv: torch.Tensor, row: int, col: int) -> torch.Tensor:
    hat = dense.clone()
    hat[row] = dense[row] - (dense[row, col] / hinv[col, col]) * hinv[col]
    return hat


def _coupling(leverages: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    squared = phi.float().square()
    numerator = leverages.float().T @ squared
    denominator = squared.sum(dim=0).clamp_min(1e-12)
    return numerator / denominator.reshape(1, -1)


def _outer(directions: torch.Tensor) -> torch.Tensor:
    return directions[:, :, None] * directions[:, None, :]


def _svd_basis(points: np.ndarray, threshold: float = 0.05) -> tuple[np.ndarray, int]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.eye(points.shape[1])[:1], 1
    rank = max(1, int(np.sum(singular > threshold * singular[0])))
    return vt[:rank], rank


def _chord_tangent(points: np.ndarray, index: int) -> np.ndarray:
    """Unit step from a point toward its nearest neighbor."""
    distances = np.linalg.norm(points - points[index], axis=1)
    distances[index] = np.inf
    neighbor = int(np.argmin(distances))
    step = points[neighbor] - points[index]
    norm = float(np.linalg.norm(step))
    if norm <= 0.0:
        return np.eye(points.shape[1])[0]
    return step / norm


def _local_basis(points: np.ndarray, index: int, k: int) -> tuple[np.ndarray, int]:
    distances = np.linalg.norm(points - points[index], axis=1)
    neighbors = np.argsort(distances)[:k]
    return _svd_basis(points[neighbors])


def _reconstruction_error(mean: np.ndarray, points: np.ndarray, bases: list[np.ndarray]) -> float:
    errors = []
    for point, basis in zip(points, bases):
        delta = point - mean
        coeff = basis @ delta
        recon = coeff @ basis
        errors.append(np.linalg.norm(delta - recon) / max(np.linalg.norm(delta), 1e-15))
    return float(np.mean(errors))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-15)


def _weight(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randn(dim, dim, generator=generator) / math.sqrt(dim)


def _spearman(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    rank_a = np.empty_like(a)
    rank_b = np.empty_like(b)
    rank_a[np.argsort(a, kind="mergesort")] = np.arange(len(a))
    rank_b[np.argsort(b, kind="mergesort")] = np.arange(len(b))
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])
