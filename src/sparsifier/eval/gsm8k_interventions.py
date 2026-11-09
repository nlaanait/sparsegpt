"""Stage 2: held-out behavioral effects of matched weight removals.

Single entries and ambient-matched discordant groups are measured before and
after simultaneous OBS compensation within each affected output row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.gsm8k_pruning import calibration, pair_inputs, answer_likelihood, behavior
from sparsifier.eval.gsm8k_sensitivity import balanced_split
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.factor_atlas import FactorAtlas, projector_diagonals


def compensated_rows(weight, inverse, entries, compensated=True):
    """Return row updates for an exact jointly constrained damped OBS removal."""
    groups = {}
    for entry in entries:
        row, col = divmod(int(entry), weight.shape[1])
        groups.setdefault(row, set()).add(col)
    updates = {}
    for row, columns in groups.items():
        cols = torch.tensor(sorted(columns), device=inverse.device)
        w = weight[row].to(inverse.device)
        if compensated:
            coefficients = torch.linalg.solve(inverse[cols][:, cols], w[cols])
            delta = inverse[:, cols] @ coefficients
        else:
            delta = torch.zeros_like(w)
            delta[cols] = w[cols]
        candidate = w - delta
        candidate[cols] = 0
        updates[row] = candidate.cpu()
    return updates


def matched_design(entries, ambient, geometric, *, singles=128, group_pairs=8, group_size=8, seed=0):
    """Disjoint high/low geometry groups matched within ambient cost bins.

    Matching uses calibration scores only. Keep actual group costs and ratios
    in the design; this is a development matching rule, not post-hoc filtering.
    """
    rng = np.random.default_rng(seed)
    if singles > len(entries):
        raise ValueError("not enough sampled entries")
    sampled = rng.choice(len(entries), singles, replace=False)
    result = [{"id": f"single-{i}", "kind": "representative", "entries": [int(entries[j])],
               "pool_indices": [int(j)]} for i, j in enumerate(sampled)]
    used = set(sampled.tolist())
    ratios = np.asarray(geometric) / np.maximum(ambient, 1e-30)
    # Tight bins control ambient importance without selecting on outcomes.
    bins = np.array_split(np.argsort(ambient), 64)
    matches = []
    for bucket in bins:
        bucket = np.array([j for j in bucket if j not in used], dtype=int)
        if len(bucket) < 8:
            continue
        order = bucket[np.argsort(ratios[bucket])]
        lows, highs = order[:len(order)//4], order[-len(order)//4:]
        available = set(lows.tolist())
        for high in highs[::-1]:
            if not available:
                break
            low = min(available, key=lambda j: abs(np.log(max(ambient[j],1e-30)/max(ambient[high],1e-30))))
            cost_ratio = max(ambient[low], ambient[high])/max(min(ambient[low], ambient[high]),1e-30)
            if cost_ratio > 1.1 or ratios[high] <= ratios[low]:
                continue
            matches.append((int(high), int(low)))
            available.remove(low)
    rng.shuffle(matches)
    if len(matches) < group_pairs * group_size:
        raise ValueError("insufficient disjoint matches within 10% ambient-cost tolerance")
    for group in range(group_pairs):
        batch = matches[group*group_size:(group+1)*group_size]
        for side, label in ((0,"high_geometry"),(1,"low_geometry")):
            ids = [pair[side] for pair in batch]
            result.append({"id": f"group-{group}-{label}", "kind": label, "match_id": group,
                           "entries": [int(entries[j]) for j in ids], "pool_indices": ids,
                           "ambient_cost_sum": float(np.sum(ambient[ids])),
                           "geometric_cost_sum": float(np.sum(geometric[ids]))})
    return result


def pool_scores(weight, inverse, ambient, readout, pdiag, deltas, labels, gradient_inputs, gradients, entries):
    device = inverse.device
    p = torch.tensor(entries // weight.shape[1], device=device)
    q = torch.tensor(entries % weight.shape[1], device=device)
    unique, lookup = torch.unique(q, return_inverse=True)
    w2 = weight.to(device)[p,q].square()
    diag = inverse.diagonal()[q]
    cols = inverse[:,unique]
    transported = readout.to(device) @ cols
    lev = torch.tensor(pdiag, device=device, dtype=torch.float32)[:,p]
    dx = deltas.to(device).clone()
    ids = labels.to(device)
    for label in ids.unique():
        mask = ids == label
        # Equivalent to the per-knob trace-normalized covariance.
        dx[mask] /= dx[mask].square().sum().sqrt().clamp_min(1e-12) * len(ids.unique())**.5
    difference_transport = dx @ cols
    gx, gg = gradient_inputs.to(device), gradients.to(device)
    result = {"magnitude": w2,
              "wanda": w2 * ambient.to(device).square().mean(0)[q],
              "ambient_obs": w2 / diag,
              "geometric_uncompensated": w2 * (lev * readout.to(device)[:,q].square()).mean(0),
              "geometric_compensated": w2 * (lev * transported[:,lookup].square()).mean(0) / diag.square(),
              "input_difference_compensated": w2 * difference_transport.square().sum(0)[lookup] / diag.square(),
              "empirical_fisher_readout": w2 * (gg[:,p].square() * gx[:,q].square()).mean(0)}
    return {k: v.detach().cpu().numpy() for k,v in result.items()}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval.natural_layer import replaced_weight
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--singles", type=int, default=128)
    parser.add_argument("--group-pairs", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    capture, index, by_id, x, y = load_capture(args.capture)
    manifest = load_manifest(args.manifest)
    sensitivity_config = json.loads((args.sensitivity / "config.json").read_text())
    sensitivity_report = json.loads((args.sensitivity / "report.json").read_text())
    if not sensitivity_report.get("complete") or sensitivity_report["config_sha256"] != digest(sensitivity_config):
        raise ValueError("gradient collection incomplete")
    if manifest["manifest_sha256"] != capture["manifest_sha256"] or manifest["manifest_sha256"] != sensitivity_config["manifest_sha256"]:
        raise ValueError("data provenance mismatch")
    rows = balanced_split(manifest["rows"], "stage2_holdout", args.eval_count)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"capture_sha256": digest(capture), "manifest_sha256": manifest["manifest_sha256"],
              "sensitivity_sha256": digest(sensitivity_config), "revision": REVISION,
              "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
              "seed": args.seed, "singles": args.singles, "group_pairs": args.group_pairs,
              "group_size": args.group_size, "evaluation_ids": [r["id"] for r in rows],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "purpose": "development_stage2_not_confirmatory"}
    write_immutable(args.output / "config.json", config)
    config_hash = digest(config)
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"config_sha256":config_hash,"outcomes":[]}
    if report["config_sha256"] != config_hash:
        raise ValueError("report provenance mismatch")
    if report.get("complete"):
        return
    memory = configure_cuda(56.)
    weight = torch.from_numpy(np.load(args.capture / "weight.npy"))
    ambient, _, use = calibration(args.capture, index, x, seed=args.seed)
    ax = ambient.cuda()
    hessian = ax.T @ ax / len(ax)
    damping = .01 * hessian.diagonal().mean()
    hessian.diagonal().add_(damping)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    del hessian, ax
    atlas = FactorAtlas.load(args.atlas)
    frames, _, _ = atlas.frames(y[use])
    pdiag = projector_diagonals(frames)
    del frames
    deltas, labels = pair_inputs(manifest, by_id, x)
    with np.load(args.sensitivity / "aligned.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != digest(sensitivity_config):
            raise ValueError("gradient array provenance mismatch")
        gx, gg = torch.from_numpy(saved["inputs"].copy()), torch.from_numpy(saved["gradients"].copy())
    entries = np.random.default_rng(args.seed).choice(weight.numel(), 8192, replace=False)
    scores = pool_scores(weight, inverse, ambient, torch.from_numpy(x[use]), pdiag, deltas, labels, gx, gg, entries)
    design = matched_design(entries, scores["ambient_obs"], scores["geometric_compensated"],
        singles=args.singles, group_pairs=args.group_pairs, group_size=args.group_size, seed=args.seed)
    write_immutable(args.output / "design.json", {"config_sha256": config_hash, "interventions": design})
    del ambient, pdiag, deltas, gx, gg
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer = model.model.layers[capture["block"]].get_submodule(capture["module"])
    if not torch.equal(layer.weight.detach().float().cpu(), weight):
        raise ValueError("captured weight differs from model")
    baseline = [answer_likelihood(model, tokenizer, r["question"], r["answer"]) for r in rows]
    repeated = [answer_likelihood(model, tokenizer, r["question"], r["answer"]) for r in rows]
    report["no_op_max_nll_difference"] = float(np.max(np.abs(np.array(baseline)-repeated)))
    report["baseline_nll"] = baseline
    dense_output = args.output / "dense-behavior"
    dense_output.mkdir(exist_ok=True)
    report["dense_behavior"] = behavior(model, tokenizer, rows, dense_output, digest([config_hash,"dense"]))
    test_positions = [by_id[r["id"]] for r in rows]
    test_inputs = torch.from_numpy(x[test_positions])
    test_frames, _, _ = atlas.frames(y[test_positions])
    holdout_pairs = [p for p in manifest["pairs"] if p["split"] == "stage2_holdout"
                     and p["left"] in by_id and p["right"] in by_id]
    held_deltas = torch.from_numpy(np.stack([x[by_id[p["right"]]]-x[by_id[p["left"]]] for p in holdout_pairs]))
    complete = {(r["id"],r["compensation"]) for r in report["outcomes"]}
    for intervention in design:
        for mode in ("none","obs"):
            if (intervention["id"], mode) in complete:
                continue
            updates = compensated_rows(weight, inverse, intervention["entries"], mode == "obs")
            candidate = weight.clone()
            for row, update in updates.items(): candidate[row] = update.to(torch.bfloat16).float()
            with replaced_weight(layer, candidate):
                values = [answer_likelihood(model, tokenizer, r["question"], r["answer"]) for r in rows]
                generated = None
                if intervention["kind"] != "representative":
                    out = args.output / (intervention["id"] + "-" + mode)
                    out.mkdir(exist_ok=True)
                    generated = behavior(model, tokenizer, rows, out, digest([config_hash,intervention["id"],mode]))
            indices = intervention["pool_indices"]
            changed_rows = sorted(updates)
            residual_rows = candidate[changed_rows] - weight[changed_rows]
            error = test_inputs @ residual_rows.T
            projected = [float(np.square(error[i].numpy() @ test_frames[i][changed_rows]).sum())
                         for i in range(len(rows))]
            result = {**intervention, "compensation":mode, "answer_nll":values,
                      "delta_answer_nll":(np.array(values)-baseline).tolist(), "behavior":generated,
                      "local": {"ambient_readout_mse":float(error.square().sum(1).mean()),
                                "heldout_local_span_mse":float(np.mean(projected)),
                                "heldout_pair_difference_mse":float((held_deltas @ residual_rows.T).square().sum(1).mean())},
                      "scores_sum_is_descriptive_for_groups": {name:float(vals[indices].sum()) for name, vals in scores.items()}}
            report["outcomes"].append(result)
            atomic_json(report_path, report)
            print(json.dumps({"intervention":intervention["id"],"compensation":mode,
                              "delta_nll":float(np.mean(result["delta_answer_nll"]))}), flush=True)
            del candidate
    correlations = {}
    for mode in ("none","obs"):
        outcomes = [r for r in report["outcomes"] if r["kind"] == "representative" and r["compensation"] == mode]
        target = [np.mean(r["delta_answer_nll"]) for r in outcomes]
        correlations[mode] = {name:float(spearmanr([r["scores_sum_is_descriptive_for_groups"][name] for r in outcomes],target).statistic)
                              for name in scores} if len(outcomes)>2 else {}
    report.update(complete=True, correlations=correlations, memory=memory,
                  peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    atomic_json(report_path,report)


if __name__ == "__main__":
    main()
