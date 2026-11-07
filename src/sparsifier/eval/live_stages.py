"""Run revised-note Stages 0–4 on one Qwen 3.5-0.8B linear layer.

The checkpoint is loaded only long enough to record activations and a
language-model gradient. Pruning then runs on CPU. Host memory is checked
against the 110 GiB unified-memory brick before the load and between stages.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from sparsifier.eval.revised_protocol import (
    ALPHA_IN_SWEEP,
    _ambient_prune,
    _centered_prune,
    _damped_inverse,
    _errors,
    _match_nnz,
    _mean_residual,
    _obs_knockout,
    _ones_prune,
    _scaled_prune,
    _spearman,
    _variant_a_prune,
    d1_decision,
    d3_decision,
    h1_disconfirmed,
    stage4_within_tolerance,
)
from sparsifier.prune.ladder import fisher_sample_weights, row_scales_from_sample_weights, static_union_row_scales
from sparsifier.prune.rescale import row_scales
from sparsifier.prune.revised_methods import projected_cg_refit, sparse_plus_low_rank, write_alignment
from sparsifier.prune.stage0 import (
    congruence_relative_error,
    gradient_enrichment,
    leverage_concentration,
    mask_flip_rate,
    mean_principal_angle,
    radial_share,
)

BRICK_GIB = 110.0
CUSHION_GIB = 8.0


def cap_under_brick(
    requested_gib: float,
    *,
    total_gib: float,
    available_gib: float,
    brick_gib: float = BRICK_GIB,
    cushion_gib: float = CUSHION_GIB,
) -> float:
    """Largest cap that keeps used memory at least ``cushion_gib`` under the brick."""
    used = float(total_gib) - float(available_gib)
    room = float(brick_gib) - float(cushion_gib) - used
    if room < 3.0:
        raise RuntimeError(
            f"host is using {used:.1f} GiB; refusing a load because the {brick_gib:.0f} GiB "
            f"brick leaves {room:.1f} GiB"
        )
    return min(float(requested_gib), room)


def read_host_memory_gib() -> tuple[float, float]:
    """Return ``(MemTotal, MemAvailable)`` in GiB."""
    total = available = None
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) / (1024.0**2)
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / (1024.0**2)
    if total is None or available is None:
        raise RuntimeError("MemTotal or MemAvailable is missing from /proc/meminfo")
    return total, available


def assert_below_brick(where: str) -> float:
    """Print host memory and stop inside the cushion under 110 GiB."""
    total, available = read_host_memory_gib()
    used = total - available
    print(
        f"memory [{where}]: used={used:.1f} GiB available={available:.1f} GiB brick={BRICK_GIB:.0f}",
        flush=True,
    )
    if used >= BRICK_GIB - CUSHION_GIB:
        raise RuntimeError(
            f"stopping at {where}: host is using {used:.1f} GiB, inside the cushion under {BRICK_GIB:.0f} GiB"
        )
    return used


def complete_layer_stages(
    weight: torch.Tensor,
    phi: torch.Tensor,
    knobs: torch.Tensor,
    presence: torch.Tensor,
    gradients: torch.Tensor,
    *,
    sparsity: float = 0.5,
    seed: int = 0,
    memory_check=None,
) -> dict:
    """Stages 0–4 for one layer, using knob and presence rows in its input space."""
    dense = weight.detach().float().cpu()
    activations = phi.detach().float().cpu()
    knob_rows = knobs.detach().float().cpu()
    presence_rows = presence.detach().float().cpu()
    grads = gradients.detach().float().cpu()
    if knob_rows.shape[1] != dense.shape[1] or presence_rows.shape[1] != dense.shape[1]:
        raise ValueError("knob and presence rows must match the layer input size")
    if activations.shape[1] != dense.shape[1] or grads.shape != (activations.shape[0], dense.shape[0]):
        raise ValueError("activations and gradients must match the layer and each other")
    if activations.shape[0] < 4 or knob_rows.shape[0] < 4:
        raise ValueError("need at least four calibration rows and four knobs")

    def check(name: str) -> None:
        if memory_check is not None:
            memory_check(name)

    dout, din = dense.shape
    n_train = activations.shape[0] // 2
    train_phi = activations[:n_train]
    test_phi = activations[n_train:]
    train_grad = grads[:n_train]
    knob_out = _push(dense, knob_rows)
    presence_out = _push(dense, presence_rows)
    tangent_basis = _basis(knob_out)
    check("stage0")
    stage0 = _stage0(dense, train_phi, knob_out, presence_out, tangent_basis, train_grad, seed)
    check("stage1")
    stage1 = _stage1(knob_out, stage0["D4"])
    directions = _heldout_directions(knob_out, presence_out, test_phi.shape[0])
    check("stage2")
    stage2 = _stage2(dense, train_phi, test_phi, train_grad, tangent_basis, directions, seed)
    check("stage3")
    stage3 = _stage3(
        dense,
        train_phi,
        test_phi,
        train_grad,
        knob_rows,
        knob_out,
        presence_out,
        tangent_basis,
        directions,
        float(sparsity),
        seed,
    )
    check("stage4")
    stage4 = _stage4(dense, train_phi, test_phi, directions, stage3["cache"], float(sparsity))
    stage3.pop("cache", None)
    return {
        "sparsity": float(sparsity),
        "output_dim": dout,
        "input_dim": din,
        "calibration_rows": int(activations.shape[0]),
        "knobs": int(knob_rows.shape[0]),
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
    }


def run_qwen_layer_stages(
    *,
    concept_dir: Path,
    presence_dir: Path,
    block: int = 0,
    module_name: str | None = None,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_gib: float = 8.0,
    sparsity: float = 0.5,
    out_path: Path = Path("results/tangent/qwen35-0.8b-stages.json"),
) -> dict:
    """Load Qwen under the brick, record one layer, then finish the stages on CPU."""
    assert_below_brick("before-load")
    total, available = read_host_memory_gib()
    allowed = cap_under_brick(max_gib, total_gib=total, available_gib=available)
    from sparsifier.data.export_concept_pairs import decoder_linears
    from sparsifier.models.qwen import force_pytorch_attention, get_qwen

    cap = _cap_cuda_under_brick(int(allowed * 1024**3))
    print(f"CUDA allocator cap: {cap / 1024**3:.2f} GiB", flush=True)
    force_pytorch_attention()
    model = get_qwen(model_id, use_kernels=False)
    model.eval()
    model.to("cuda")
    assert_below_brick("after-load")
    modules = decoder_linears(model)
    name, layer, concept_path, presence_path = _select_layer(
        modules, concept_dir, presence_dir, block, module_name
    )
    recorded = _record_layer(model, layer, short=cap < 6 * 1024**3)
    weight = layer.weight.detach().float().cpu().clone()
    del model
    torch.cuda.empty_cache()
    assert_below_brick("after-free")
    report = complete_layer_stages(
        weight,
        recorded["phi"],
        torch.tensor(np.load(concept_path)),
        torch.tensor(np.load(presence_path)),
        recorded["gradients"],
        sparsity=sparsity,
        memory_check=assert_below_brick,
    )
    report.update(
        {
            "model": model_id,
            "block": block,
            "module": name,
            "cap_gib": cap / 1024**3,
            "peak_gib": recorded["peak_gib"],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert_below_brick("after-stages")
    _print_decisions(report)
    print(f"wrote {out_path}", flush=True)
    return report


def _stage0(weight, phi, knob_out, presence_out, tangent_basis, gradients, seed: int) -> dict:
    dout = weight.shape[0]
    leverages = tangent_basis.square().sum(dim=1)
    summary = leverage_concentration(leverages, rank=int(tangent_basis.shape[1]))
    d1 = {
        "ks_statistic": summary["ks_statistic"],
        "n": dout,
        "decision": d1_decision(summary["ks_statistic"], dout),
    }
    gamma = dout / float(tangent_basis.shape[1])
    vanilla = _ambient_prune(weight, phi, 0.5)
    variant = _scaled_prune(weight, phi, static_union_row_scales(leverages, gamma, weight.shape[1]), 0.5)[0]
    null_basis = _random_basis(dout, int(tangent_basis.shape[1]), seed)
    null = _scaled_prune(
        weight, phi, static_union_row_scales(null_basis.square().sum(dim=1), gamma, weight.shape[1]), 0.5
    )[0]
    flip = mask_flip_rate(vanilla != 0, variant != 0)
    null_flip = mask_flip_rate(vanilla != 0, null != 0)
    d2 = {
        "flip_rate": flip,
        "null_flip_rate": null_flip,
        "decision": "output_side_cannot_matter" if abs(flip - null_flip) < 0.05 else "output_side_moves_masks",
    }
    projector = tangent_basis @ tangent_basis.T
    presence_basis = _basis(_orthogonalize(presence_out, tangent_basis))
    geo = torch.cat((tangent_basis, presence_basis), dim=1)
    kappa_t = gradient_enrichment(gradients, projector, float(tangent_basis.shape[1]))
    kappa_p = gradient_enrichment(
        gradients, presence_basis @ presence_basis.T, float(presence_basis.shape[1])
    )
    d3 = {
        "kappa_tangent": kappa_t,
        "kappa_presence": kappa_p,
        "gradient_source": "token_nll",
        "decision": d3_decision(kappa_t, kappa_p),
    }
    d4_angle, d4_null = _split_angle(knob_out, seed)
    d4 = {
        "mean_principal_angle": d4_angle,
        "null_principal_angle": d4_null,
        "R": int(tangent_basis.shape[1]),
        "decision": "stop_unstable" if d4_angle > 0.5 * d4_null else "continue",
    }
    d5_error, d5_spearman = _fixed_congruence(weight, phi, leverages, gamma)
    d5 = {
        "congruence_error": d5_error,
        "obs_cost_spearman": d5_spearman,
        "decision": "use_exact_refit" if d5_error > 0.25 else "congruence_ok",
    }
    shares = [radial_share(presence_out[index], tangent_basis) for index in range(presence_out.shape[0])]
    share = float(np.mean(shares))
    d6 = {
        "radial_share": share,
        "decision": "presence_required" if share > 0.30 else "tangent_only",
    }
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}


def _stage1(knob_out: torch.Tensor, d4: dict) -> dict:
    local_errors = []
    global_errors = []
    directions = knob_out.numpy()
    for index in range(directions.shape[0]):
        held = directions[index]
        others = np.delete(directions, index, axis=0)
        nearest = others[int(np.argmax(np.abs(others @ held)))]
        local_errors.append(1.0 - abs(float(np.dot(nearest, held))))
        _, _, vt = np.linalg.svd(others, full_matrices=False)
        global_errors.append(1.0 - abs(float(np.dot(vt[0], held))))
    local_error = float(np.mean(local_errors))
    global_error = float(np.mean(global_errors))
    rank = int(d4["R"])
    unstable = d4["decision"] == "stop_unstable"
    return {
        "analytic_rank_reduction_possible": rank > 2,
        "rank_reduction_possible": rank > 2,
        "estimated_r": 1,
        "estimated_b": rank,
        "heldout_local_error": local_error,
        "heldout_global_error": global_error,
        "unstable": unstable,
        "h1_disconfirmed": h1_disconfirmed(
            unstable=unstable, local_error=local_error, global_error=global_error
        ),
    }


def _stage2(weight, train_phi, test_phi, gradients, tangent_basis, directions, seed: int) -> dict:
    hessian, hinv = _damped_inverse(train_phi)
    diagonal = torch.diag(hinv).clamp_min(1e-12)
    energy = train_phi.square().mean(dim=0)
    leverages = tangent_basis.square().sum(dim=1)
    coupling = leverages.reshape(-1, 1) * energy.reshape(1, -1)
    fisher = weight.square() * (gradients.square().T @ train_phi.square())
    scores = {
        "magnitude": weight.square(),
        "wanda": weight.square() * energy.reshape(1, -1),
        "ambient_obs": weight.square() / diagonal.reshape(1, -1),
        "fisher": fisher,
        "geometric": weight.square() * coupling,
    }
    generator = np.random.default_rng(seed + 11)
    count = min(8, weight.numel())
    rows = generator.integers(0, weight.shape[0], size=count)
    cols = generator.integers(0, weight.shape[1], size=count)
    observed = []
    predicted = {name: [] for name in scores}
    downstream = []
    for row, col in zip(rows, cols):
        hat = _obs_knockout(weight, hinv, int(row), int(col))
        observed.append(_errors(weight, hat, test_phi, directions["tangent"], directions["presence"])["tangent"])
        downstream.append(_probe(weight, hat, test_phi, tangent_basis[:, 0]))
        for name, matrix in scores.items():
            predicted[name].append(float(matrix[int(row), int(col)]))
    return {
        "spearman_vs_compensated_tangent": {name: _spearman(predicted[name], observed) for name in scores},
        "downstream": {
            "probe_mse": float(np.mean([row["probe_mse"] for row in downstream])),
            "kl_to_dense": float(np.mean([row["kl_to_dense"] for row in downstream])),
        },
        "selective_vs_broad": {
            "selective": _column_drop(weight, test_phi, directions, tangent_basis, by="alignment"),
            "broad": _column_drop(weight, test_phi, directions, tangent_basis, by="magnitude", phi=train_phi),
        },
    }


def _stage3(
    weight,
    train_phi,
    test_phi,
    gradients,
    knob_rows,
    knob_out,
    presence_out,
    tangent_basis,
    directions,
    sparsity: float,
    seed: int,
) -> dict:
    dout, din = weight.shape
    gamma = dout / float(tangent_basis.shape[1])
    leverages = tangent_basis.square().sum(dim=1)
    local_lev = knob_out.square().mean(dim=0)
    presence_basis = _basis(_orthogonalize(presence_out, tangent_basis))
    geo_basis = torch.cat((tangent_basis, presence_basis), dim=1)
    geo_gamma = dout / float(geo_basis.shape[1])
    scales = {
        "static_union": static_union_row_scales(leverages, gamma, din),
        "global_bsf": static_union_row_scales(leverages, gamma, din),
        "local_tangents": row_scales(local_lev.reshape(-1, 1).expand(-1, din).contiguous(), gamma),
        "tangents_and_presence": static_union_row_scales(geo_basis.square().sum(dim=1), geo_gamma, din),
        "fisher": row_scales_from_sample_weights(fisher_sample_weights(gradients, 1.0), train_phi),
    }

    def evaluate(hat: torch.Tensor) -> dict:
        stats = _errors(weight, hat, test_phi, directions["tangent"], directions["presence"])
        stats["actual_sparsity"] = float((hat == 0).to(dtype=torch.float32).mean())
        stats["downstream"] = _probe(weight, hat, test_phi, tangent_basis[:, 0])
        return stats

    vanilla = _ambient_prune(weight, train_phi, sparsity)
    ladder = {"ambient": evaluate(vanilla)}
    for name, scale in scales.items():
        ladder[name] = evaluate(_scaled_prune(weight, train_phi, scale, sparsity)[0])
    knob_ids = torch.arange(knob_rows.shape[0])
    variant_a = {}
    patterns = [("unstructured", 0, 0)]
    if din % 4 == 0:
        patterns.append(("nm_2_4", 2, 4))
    for pattern, zeros, group in patterns:
        points = []
        for alpha in ALPHA_IN_SWEEP:
            hat = _variant_a_prune(weight, train_phi, knob_rows, knob_ids, alpha, sparsity, zeros, group)
            point = evaluate(hat)
            point["alpha_in"] = alpha
            points.append(point)
        variant_a[pattern] = points
    exactness = _exactness(weight, train_phi, test_phi, tangent_basis, gamma, directions)
    upstream = {
        "whole_neuron": _whole_neuron(weight, test_phi, directions, tangent_basis, train_phi),
    }
    variant_d = _variant_d(weight, train_phi, tangent_basis, evaluate)
    return {
        "variant_a": variant_a,
        "ladder": ladder,
        "exactness": exactness,
        "upstream": upstream,
        "variant_d": variant_d,
        "geometry": "fixed span of the saved knob rows",
        "cache": {"vanilla": vanilla},
    }


def _stage4(weight, train_phi, test_phi, directions, cache, sparsity: float) -> dict:
    vanilla = cache["vanilla"]

    def pack(hat: torch.Tensor, bias: torch.Tensor | None) -> dict:
        stats = _errors(weight, hat, test_phi, directions["tangent"], directions["presence"], bias=bias)
        stats["presence_residual"] = stats["presence"]
        stats["normal_residual"] = stats["normal"]
        return stats

    none = pack(vanilla, None)
    centered_hat = _centered_prune(weight, train_phi, sparsity)
    ones_hat, ones_bias = _ones_prune(weight, train_phi, sparsity)
    arms = {
        "no_bias": none,
        "centered": pack(centered_hat, _mean_residual(weight, centered_hat, train_phi)),
        "ones_column": pack(ones_hat, ones_bias),
        "posthoc_global_mean": pack(vanilla, _mean_residual(weight, vanilla, train_phi)),
    }
    for arm in arms.values():
        arm["within_tolerance"] = stage4_within_tolerance(arm, none)
        arm["input_dependent"] = arm["normal_cv"] > 0.25
    return {"tolerance": 0.10, "arms": arms}


def _exactness(weight, phi, test_phi, tangent_basis, gamma, directions) -> dict:
    width = 32
    row_index = torch.arange(weight.shape[0])
    col_index = torch.arange(weight.shape[1])
    if weight.shape[0] > width:
        row_index = torch.topk(weight.square().sum(dim=1), width).indices
    if weight.shape[1] > width:
        col_index = torch.topk(phi.square().sum(dim=0), width).indices
    sub = weight[row_index][:, col_index]
    sub_phi = phi[:, col_index]
    sub_test = test_phi[:, col_index]
    hat = _ambient_prune(sub, sub_phi, 0.5)
    hessian = sub_phi.T @ sub_phi / sub_phi.shape[0]
    cropped = tangent_basis[row_index]
    projector = cropped @ cropped.T
    metric = torch.eye(sub.shape[0]) + float(gamma) * projector
    keep = hat != 0
    ambient = projected_cg_refit(sub, keep, hessian)
    cross = projected_cg_refit(sub, keep, hessian, metric=metric)
    tangent = directions["tangent"][:, row_index.numpy()]
    presence = directions["presence"][:, row_index.numpy()]

    def evaluate(candidate: torch.Tensor) -> dict:
        return _errors(sub, candidate, sub_test, tangent, presence)

    return {
        "width": int(sub.shape[0]),
        "row_diagonal": evaluate(hat),
        "projected_cg": evaluate(ambient),
        "cross_row": evaluate(cross),
    }


def _variant_d(weight, phi, basis, evaluate) -> dict:
    dout, din = weight.shape
    budget = 0.5 * dout * din
    rank = 1
    for candidate in range(1, min(8, basis.shape[1]) + 1):
        if candidate * (dout + din) <= budget / 2.0:
            rank = candidate
    extra = rank * (dout + din)
    nnz_cap = int(math.floor(budget - extra))
    sparse = _match_nnz(
        _ambient_prune(weight, phi, 1.0 - nnz_cap / float(dout * din)),
        nnz_cap,
    )
    hat = sparse_plus_low_rank(weight, sparse, basis[:, :rank])
    stats = evaluate(hat)
    stats.update(
        {
            "rank": rank,
            "storage_budget": float(budget),
            "storage_used": float(int((sparse != 0).sum()) + extra),
        }
    )
    return stats


def _fixed_congruence(weight, phi, leverages, gamma: float) -> tuple[float, float]:
    count = phi.shape[0]
    hessian = phi.T @ phi / count
    damp = 0.01 * torch.diag(hessian).mean()
    eye = torch.eye(hessian.shape[0])
    errors = []
    spearmans = []
    for row in range(min(3, weight.shape[0])):
        factor = 1.0 + float(gamma) * float(leverages[row])
        row_hessian = factor * hessian
        scale = torch.sqrt((torch.diag(row_hessian) / torch.diag(hessian).clamp_min(1e-12)).clamp_min(0.0))
        errors.append(congruence_relative_error(row_hessian, hessian, scale))
        congruent = scale[:, None] * hessian * scale[None, :]
        true_cost = _diag_costs(weight[row], row_hessian + damp * eye)
        approx_cost = _diag_costs(weight[row], congruent + damp * eye)
        spearmans.append(_spearman(true_cost, approx_cost))
    return float(np.mean(errors)), float(np.mean(spearmans))


def _diag_costs(row: torch.Tensor, hessian: torch.Tensor) -> list[float]:
    inverse = torch.linalg.inv(0.5 * (hessian + hessian.T))
    costs = row.square() / torch.diag(inverse).clamp_min(1e-12)
    return [float(value) for value in costs]


def _split_angle(knob_out: torch.Tensor, seed: int) -> tuple[float, float]:
    midpoint = knob_out.shape[0] // 2
    left = _basis(knob_out[:midpoint])
    right = _basis(knob_out[midpoint:])
    angle = mean_principal_angle(left, right)
    rank = min(left.shape[1], right.shape[1])
    generator = np.random.default_rng(seed + 9)
    nulls = [
        mean_principal_angle(
            torch.tensor(generator.normal(size=(knob_out.shape[1], rank))),
            torch.tensor(generator.normal(size=(knob_out.shape[1], rank))),
        )
        for _ in range(4)
    ]
    return float(angle), float(np.mean(nulls))


def _whole_neuron(weight, test_phi, directions, basis, train_phi) -> dict:
    removed = max(1, weight.shape[1] // 4)
    by_alignment = _drop(weight, test_phi, directions, _neuron_order(weight, basis, "alignment")[:removed])
    by_magnitude = _drop(
        weight,
        test_phi,
        directions,
        _neuron_order(weight, basis, "magnitude", phi=train_phi)[:removed],
    )
    return {"by_alignment": by_alignment, "by_magnitude": by_magnitude, "neurons_removed": removed}


def _column_drop(weight, test_phi, directions, basis, *, by: str, phi=None) -> dict:
    order = _neuron_order(weight, basis, "alignment" if by == "alignment" else "magnitude", phi=phi)
    return _drop(weight, test_phi, directions, [int(order[0])])


def _neuron_order(weight, basis, kind: str, phi=None) -> list[int]:
    projector = basis @ basis.T
    if kind == "alignment":
        scores = [write_alignment(weight[:, column], projector) for column in range(weight.shape[1])]
        return sorted(range(weight.shape[1]), key=lambda column: scores[column])
    ambient = weight.square().sum(dim=0) * phi.square().mean(dim=0)
    return sorted(range(weight.shape[1]), key=lambda column: float(ambient[column]))


def _drop(weight, test_phi, directions, columns: list[int]) -> dict:
    hat = weight.clone()
    hat[:, columns] = 0
    stats = _errors(weight, hat, test_phi, directions["tangent"], directions["presence"])
    stats["columns"] = columns
    return stats


def _heldout_directions(knob_out, presence_out, count: int) -> dict:
    tangent = knob_out.mean(dim=0)
    tangent = tangent / tangent.norm().clamp_min(1e-12)
    presence = presence_out.mean(dim=0)
    presence = presence - tangent * torch.dot(presence, tangent)
    presence = presence / presence.norm().clamp_min(1e-12)
    return {
        "tangent": np.repeat(tangent.numpy().reshape(1, -1), count, axis=0),
        "presence": np.repeat(presence.numpy().reshape(1, -1), count, axis=0),
    }


def _probe(weight, hat, phi, direction: torch.Tensor) -> dict:
    dense_h = phi @ weight.T
    hat_h = phi @ hat.T
    probe = ((hat_h - dense_h) @ direction.reshape(-1)).square().mean()
    log_dense = torch.log_softmax(dense_h, dim=1)
    log_hat = torch.log_softmax(hat_h, dim=1)
    kl = torch.nn.functional.kl_div(log_hat, log_dense, log_target=True, reduction="batchmean")
    return {"probe_mse": float(probe), "kl_to_dense": float(kl)}


def _push(weight: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    projected = rows.float() @ weight.float().T
    return projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _basis(directions: torch.Tensor) -> torch.Tensor:
    matrix = directions.float()
    _, singular, vt = torch.linalg.svd(matrix, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0.0:
        raise ValueError("concept directions have no energy")
    rank = max(1, int((singular > 0.05 * singular[0]).sum()))
    rank = max(1, min(rank, matrix.shape[1] - 1, vt.shape[0]))
    return vt[:rank].T.contiguous()


def _orthogonalize(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    cleaned = rows.float() - (rows.float() @ basis) @ basis.T
    norms = cleaned.norm(dim=1)
    kept = cleaned[norms > 1e-6]
    if kept.shape[0] == 0:
        return basis[:, :1].T
    return kept / kept.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _random_basis(dim: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 3)
    draw = torch.randn(dim, rank, generator=generator)
    basis, _ = torch.linalg.qr(draw, mode="reduced")
    return basis


def _select_layer(modules, concept_dir: Path, presence_dir: Path, block: int, module_name: str | None):
    candidates = []
    for key, module in modules.items():
        prefix, _, name = key.partition("/")
        if prefix != str(block):
            continue
        if module_name is not None and name != module_name:
            continue
        concept = concept_dir / str(block) / f"{name}.npy"
        presence = presence_dir / str(block) / f"{name}.npy"
        if concept.is_file() and presence.is_file():
            candidates.append((int(module.weight.numel()), name, module, concept, presence))
    if not candidates:
        raise FileNotFoundError(f"no saved concept and presence rows for block {block}")
    _, name, module, concept, presence = min(candidates)
    print(f"stage layer: {block}/{name} weights={module.weight.numel()}", flush=True)
    return name, module, concept, presence


def _cap_cuda_under_brick(requested_bytes: int) -> int:
    """Cap this process by the brick and by half of the CUDA-reported free memory."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if total_bytes <= 0 or free_bytes <= 0:
        raise RuntimeError(f"CUDA memory info is unusable (free={free_bytes}, total={total_bytes})")
    cap = min(int(requested_bytes), int(free_bytes * 0.5))
    floor = 3 * 1024**3
    print(
        f"CUDA free={free_bytes / 1024**3:.2f} GiB total={total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    if cap < floor:
        raise RuntimeError(
            f"refusing to load the model: cap would be {cap / 1024**3:.2f} GiB "
            f"with {free_bytes / 1024**3:.2f} GiB free"
        )
    torch.cuda.set_per_process_memory_fraction(cap / total_bytes)
    return cap


def _record_layer(model, layer, *, short: bool = False) -> dict:
    inputs: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def pre_hook(_module, args):
        activation = args[0].detach().float().cpu()
        inputs.append(activation.reshape(-1, activation.shape[-1]))

    def backward_hook(_module, _grad_input, grad_output):
        grad = grad_output[0]
        if grad is None:
            return
        stored = grad.detach().float().cpu()
        gradients.append(stored.reshape(-1, stored.shape[-1]))

    pre = layer.register_forward_pre_hook(pre_hook)
    backward = layer.register_full_backward_hook(backward_hook)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    vocab = int(model.config.vocab_size)
    try:
        plain_count, plain_length = (2, 16) if short else (4, 32)
        grad_count, grad_length = (1, 4) if short else (2, 8)
        with torch.no_grad():
            for _ in range(plain_count):
                tokens = torch.randint(1, vocab, (1, plain_length), device="cuda", generator=generator)
                model(input_ids=tokens)
        assert_below_brick("after-calibration")
        for _ in range(grad_count):
            tokens = torch.randint(1, vocab, (1, grad_length), device="cuda", generator=generator)
            output = model(input_ids=tokens)
            logits = output.logits[:, :-1, :].float()
            target = tokens[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
        assert_below_brick("after-backward")
    finally:
        pre.remove()
        backward.remove()
    if not inputs or not gradients:
        raise RuntimeError("the layer did not record activations and gradients")
    phi = torch.cat(inputs, dim=0)
    grads = torch.cat(gradients, dim=0)
    rows = min(phi.shape[0], 128)
    grad_rows = min(grads.shape[0], rows)
    # Pair the calibration rows with a repeated gradient block of the same length.
    phi = phi[:rows]
    if grads.shape[0] < phi.shape[0]:
        repeats = math.ceil(phi.shape[0] / grads.shape[0])
        grads = grads.repeat(repeats, 1)[: phi.shape[0]]
    else:
        grads = grads[: phi.shape[0]]
    _ = grad_rows
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"recorded rows={phi.shape[0]} peak={peak:.2f} GiB", flush=True)
    return {"phi": phi, "gradients": grads, "peak_gib": peak}


def _print_decisions(report: dict) -> None:
    for key, row in report["stage0"].items():
        if key.startswith("_"):
            continue
        print(f"{key}: {row['decision']}", flush=True)
    print(f"H1 disconfirmed: {report['stage1']['h1_disconfirmed']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-pairs", type=Path, default=Path("results/tangent/qwen35-0.8b-concept-pairs"))
    parser.add_argument("--presence-pairs", type=Path, default=Path("results/tangent/qwen35-0.8b-presence-pairs"))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--module", default=None)
    parser.add_argument("--max-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=Path("results/tangent/qwen35-0.8b-stages.json"))
    args = parser.parse_args(argv)
    run_qwen_layer_stages(
        concept_dir=args.concept_pairs,
        presence_dir=args.presence_pairs,
        block=args.block,
        module_name=args.module,
        max_gib=args.max_gib,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
