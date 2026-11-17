"""Sequential GSM8K MLP pruning with downstream activation tangent pullbacks.

Run through the guarded gflow worker. This driver keeps the existing 4096
C4 + 4096 math ambient budget, fits factor charts on geometry_fit readouts,
and uses a recorded pruning_calibration subset for extra backward compute.
"""
from __future__ import annotations
from sparsifier.eval.gsm8k_batching import policy as batch_policy

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.data.gsm8k_calibration import DEFAULT_MANIFEST, load_calibration, regime
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_capture import check_competence
from sparsifier.eval.gsm8k_pruning import behavior, select_validation
from sparsifier.eval.gsm8k_sequential import collect, sites_for, verify_stage
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.prune.downstream import DownstreamTangentCalibrator, TangentAnchor, WEIGHT_CONTRACTION
from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.gram_dictionary import GramBlockDictionary
from sparsifier.prune.progress import Progress, compact_diagnostics, scoped


ARM = "downstream"
PULLBACK = {"anchor_offset": 1, "num_probes": 8, "probe_count": 32,
            "weight_contraction": WEIGHT_CONTRACTION,
            "anchor_site": "block_output_residual", "anchor_positions": "last_prompt_token",
            "target_positions": "all_causal_prefix_tokens",
            "geometry": "factor_local_tangents", "input_alpha": 0.0,
            "ambient_tokens": {"c4": 4096, "math": 4096},
            "approximation": "diagonal_output_and_token_then_row_congruence"}


class _AnchorReached(Exception):
    pass


class AnchorPrefix(nn.Module):
    """Stop a real model forward after anchor hooks, before later blocks/logits."""

    def __init__(self, model, anchor):
        super().__init__()
        self.model = model
        self.anchor = anchor

    def forward(self, **kwargs):
        reached = False

        def stop(_module, _args, _output):
            nonlocal reached
            reached = True
            raise _AnchorReached

        # Added after capture/calibration hooks, so those see the anchor first.
        handle = self.anchor.register_forward_hook(stop)
        try:
            try:
                self.model(**kwargs)
            except _AnchorReached:
                pass
        finally:
            handle.remove()
        if not reached:
            raise RuntimeError("forward did not reach downstream anchor")


def anchor_block(block, layer_count, offset=1):
    if not 0 <= block < layer_count or offset < 1:
        raise ValueError("invalid target block or downstream anchor offset")
    # The final block still has its own residual output downstream of the MLP.
    return min(block + offset, layer_count - 1)


def probe_rows(rows, count, seed):
    eligible = [row for row in rows if row["split"] == "pruning_calibration"]
    if count < 1:
        raise ValueError("positive backward calibration problem count required")
    selected = select_validation([dict(row, split="validation") for row in eligible], count, seed)
    by_id = {row["id"]: row for row in eligible}
    return [by_id[row["id"]] for row in selected]


def readout_basis(atlas, output):
    """Fit-free chart assignment; seed only the final prompt activation."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("GSM8K pullbacks require batch-one [batch, tokens, hidden] anchors")
    frames, _, _ = atlas.frames(output[:, -1].detach().float().cpu().numpy(), mode="local")
    frame = torch.as_tensor(frames[0], device=output.device, dtype=torch.float32)
    basis = torch.zeros((*output.shape, frame.shape[-1]), device=output.device, dtype=torch.float32)
    basis[0, -1] = frame
    return basis


def prune_site(model, tokenizer, target, anchor, rows, c4_tokens, output, stage_index,
               *, seed=0, sparsity=.5, alpha=1., num_probes=8, probe_count=32, metric="activation_diagonal",
               input_alpha=0., input_structure="independent_tokens", full_metric_refit=False,
               full_metric_rtol=1e-5, full_metric_max_iterations=200,
               replay_cache_budget_bytes=256*1024**2, replay_disk_budget_bytes=64*1024**3,
               replay_chunk_size=32, progress=None):
    """Refresh ambient/anchor activations and calibrate one current-model site."""
    if any(row["split"] not in ("geometry_fit", "pruning_calibration") for row in rows):
        raise ValueError("geometry and pruning must exclude held-out examples")
    selected = probe_rows(rows, probe_count, seed)
    prefix = AnchorPrefix(model, anchor)
    anchor_readouts = []

    def capture(_module, _args, value):
        hidden = value[0] if isinstance(value, tuple) else value
        anchor_readouts.extend(hidden[:, -1].detach().float().cpu().numpy())

    handle = anchor.register_forward_hook(capture)
    try:
        # Same selected math positions and C4 windows as existing sequential arms.
        x, y, inputs, c4 = collect(prefix, tokenizer, target, rows, c4_tokens, seed, progress=progress)
    finally:
        handle.remove()
    if len(inputs) != 8192 or len(c4) != 4096:
        raise ValueError("downstream arm requires the matched 4096 C4 + 4096 math token budget")
    fit = np.array([row["split"] == "geometry_fit" for row in rows])
    if not fit.any():
        raise ValueError("geometry_fit samples required")
    downstream = np.stack(anchor_readouts[:len(rows)])
    atlas = FactorAtlas(8, 4, 2, 4, 1, seed)
    atlas.dictionary = GramBlockDictionary(8, 4, 2, seed)
    atlas.fit(downstream[fit], **({"progress": scoped(progress, atlas_role="downstream_output")} if progress is not None else {}))
    atlas_diagnostics = atlas.diagnostics(downstream) if progress is not None else None
    atlas_path = output / f"atlas-{stage_index:03d}.npz"
    atlas.save(atlas_path)
    from sparsifier.eval.gsm8k_input_tangents import fit_input_tangents
    from sparsifier.prune.curvature_replay import DiskFactorReplay, ReplayNeuronMetric, ReplayProbeConsumer
    input_basis, input_hash = None, None
    if input_alpha:
        input_path = output/f"input-atlas-{stage_index:03d}.npz"
        input_basis = fit_input_tangents(x, rows, input_path, seed, input_structure,
                                        progress=scoped(progress, atlas_role="input"))
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    replay = None
    if full_metric_refit:
        replay = DiskFactorReplay(output/f"curvature-{stage_index:03d}",
            provenance={"stage": stage_index, "seed": seed, "metric": metric,
                "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(), "input_atlas_sha256": input_hash,
                "input_structure": input_structure, "probe_ids": [r["id"] for r in selected], "num_probes": num_probes},
            cache_budget_bytes=replay_cache_budget_bytes, disk_budget_bytes=replay_disk_budget_bytes)
    base_consumer = ReplayProbeConsumer(target.weight, replay, collect_group_scores=False) if replay is not None else None
    input_consumer = ReplayProbeConsumer(target.weight, replay, term="input", collect_group_scores=False) if replay is not None else None
    del x, y, c4, downstream, anchor_readouts
    calibration = DownstreamTangentCalibrator(
        {"target": target},
        {"downstream": TangentAnchor(anchor, lambda value: readout_basis(atlas, value),
                                      select=lambda value: value[0] if isinstance(value, tuple) else value)},
        alpha=alpha, num_probes=num_probes, seed=seed, metric=metric,
        input_alpha=input_alpha, input_structure=input_structure,
        input_bases={"target": input_basis} if input_basis is not None else None,
        probe_consumer=base_consumer, input_probe_consumer=input_consumer,
    )
    solver = calibration.solvers["target"]
    solver.add_ambient_batch(inputs)
    ambient_count = len(inputs)
    h0 = inputs.to(target.weight.device).float().T@inputs.to(target.weight.device).float()/ambient_count if full_metric_refit else None
    del inputs
    device = target.weight.device
    if progress is not None:
        progress({"event": "pullback_backend", **WEIGHT_CONTRACTION, "device": str(device),
                  "cpu_probe_consumer": base_consumer is not None or input_alpha > 0})
    for index, row in enumerate(selected):
        started = time.monotonic()
        ids = tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False)
        if not 0 < len(ids) <= 2048:
            raise ValueError("backward calibration prompt exceeds budget")
        tokens = torch.tensor([ids], device=device)
        calibration.run(lambda: prefix(input_ids=tokens, use_cache=False, logits_to_keep=1),
                        collect_ambient=False)
        if progress is not None:
            progress({"event": "pullback_calibrated", "problems_done": index+1,
                "problems_total": len(selected), "requested_probes_per_problem": num_probes,
                "prompt_tokens": len(ids), "problem_seconds": time.monotonic()-started,
                "geometry_vectors_accumulated": solver.geometry_vector_count,
                "raw_geometry_energy_accumulated": float(solver.geometry_energy.sum()),
                "raw_input_geometry_energy_accumulated": float(solver.input_geometry_energy.sum())})
    full, replay_hash = None, None
    if replay is not None:
        replay_hash = replay.seal()
        full = ReplayNeuronMetric(ambient_count, replay, chunk_size=replay_chunk_size)
        if metric == "activation_diagonal":
            solver.geometry_energy.copy_(full.diagonal(solver.geometry_energy.shape)*ambient_count)
            solver.metric = "weight_diagonal"
    solver.finalize_tangents()
    if progress is not None:
        progress({"event": "geometry_calibrated", **solver.tangent_diagnostics,
                  "scale_min": float(solver.scale.min()), "scale_max": float(solver.scale.max())})
    if alpha > 0 and solver.tangent_diagnostics["geometry_trace"] <= 0:
        raise ValueError("no supported downstream geometry; refuse geometric pruning arm")
    if input_alpha > 0 and solver.tangent_diagnostics["input_geometry_trace"] <= 0:
        raise ValueError("no supported input tangent geometry; refuse two-sided arm")
    # Match the existing experiment's float32 compensation and bf16 output cast.
    # Do not mutate the real model until its complete stage checkpoint is saved.
    temporary = nn.Linear(target.weight.shape[1], target.weight.shape[0], bias=False,
                          device=device, dtype=torch.float32)
    with torch.no_grad():
        temporary.weight.copy_(target.weight)
    solver.layer = temporary
    solver.fasterprune(sparsity, blocksize=128, percdamp=.01, progress=progress)
    candidate = temporary.weight.detach().cpu().clone()
    full_diagnostics = {}
    if full is not None:
        full = full.with_weights(solver.tangent_diagnostics["gamma"], solver.tangent_diagnostics["eta"]).with_support(solver.input_energy > 0)
        # SparseGPT's accumulated H uses 2/nsamples times X.T@X. Convert
        # the factorization owner's actual ridge (including retries) to the
        # ambient-token mean units, then undo the coordinate scaling.
        delta = solver.damping_diagnostics["damping"]/(2*ambient_count/solver.nsamples)
        ridge = delta*solver.scale.to(h0).square()
        candidate, full_diagnostics = full.refit(target.weight.detach().float(), temporary.weight.detach()!=0,
            h0, ridge, rtol=full_metric_rtol, max_iterations=full_metric_max_iterations, progress=progress)
        if not full_diagnostics["full_metric_converged"]:
            raise RuntimeError("full-metric CG did not converge; candidate was not accepted")
        candidate = candidate.cpu()
        full_diagnostics.update(replay_manifest_sha256=replay_hash, replay_disk_bytes=replay.disk_bytes,
                                full_metric_diagonal="shared_weight_pullback", full_metric_rtol=full_metric_rtol,
                                selection_damping=solver.damping_diagnostics, full_metric_ridge_delta=delta)
    diagnostics = dict(solver.tangent_diagnostics,
                       metric=solver.metric, requested_calibration_metric=metric,
                       solver="downstream_tangent_row_diagonal_congruence",
                       probe_ids=[row["id"] for row in selected],
                       geometry_fit_ids=[row["id"] for row in rows if row["split"] == "geometry_fit"],
                       atlas_sha256=hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
                       num_probes=num_probes, damping=.01, blocksize=128,
                       model_precision="bfloat16", compensation_precision="float32")
    diagnostics.update(full_diagnostics, input_alpha=input_alpha, input_structure=input_structure,
                       input_atlas_sha256=input_hash, selection_damping=solver.damping_diagnostics,
                       **solver.pruning_diagnostics)
    if atlas_diagnostics is not None:
        diagnostics["atlas"] = atlas_diagnostics
    solver.free()
    return candidate, diagnostics


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.eval.gsm8k_batching import reference_backend as force_pytorch_attention
    from sparsifier.eval import gsm8k_capture, gsm8k_sequential, gsm8k_pruning
    from sparsifier.prune import downstream, factor_atlas, gram_dictionary, geometry_features, rescale, sparsegpt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--c4-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--modules", default="mlp.gate_proj,mlp.up_proj,mlp.down_proj")
    parser.add_argument("--sparsity", type=float, default=.5)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--num-probes", type=int, default=PULLBACK["num_probes"])
    parser.add_argument("--probe-count", type=int, help="manifest-defined count; controlled diagnostics default to 32")
    parser.add_argument("--metric", choices=("activation_diagonal", "weight_diagonal"), default="activation_diagonal")
    parser.add_argument("--input-alpha", type=float, default=0.)
    parser.add_argument("--input-structure", choices=("independent_tokens", "shared_directions"), default="independent_tokens")
    parser.add_argument("--full-metric-refit", action="store_true")
    parser.add_argument("--full-metric-rtol", type=float, default=1e-5)
    parser.add_argument("--full-metric-max-iterations", type=int, default=200)
    parser.add_argument("--replay-cache-mib", type=int, default=256)
    parser.add_argument("--replay-disk-budget-gib", type=float, default=64.)
    parser.add_argument("--replay-chunk-size", type=int, default=32)
    args = parser.parse_args()
    manifest = load_calibration(args.manifest)
    if args.probe_count is None:
        args.probe_count = manifest.get("backward_problem_count", PULLBACK["probe_count"])
    if not 0 <= args.sparsity < 1 or not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("invalid sparsity or alpha")
    if min(args.num_probes, args.probe_count, args.eval_count, args.max_new_tokens) < 1:
        parser.error("positive probe and evaluation budgets required")
    if (not np.isfinite(args.input_alpha) or args.input_alpha < 0 or not np.isfinite(args.full_metric_rtol)
        or args.full_metric_rtol <= 0 or args.full_metric_max_iterations < 1
        or args.replay_cache_mib < 0 or not np.isfinite(args.replay_disk_budget_gib)
        or args.replay_disk_budget_gib <= 0 or args.replay_chunk_size < 1):
        parser.error("invalid input coupling or full-metric replay configuration")
    if args.metric == "activation_diagonal" and args.input_structure != "independent_tokens":
        parser.error("activation-diagonal coupling requires independent tokens")
    pilot_hash = check_competence(args.pilot_report)
    rows = [row for row in manifest["rows"] if row["split"] in ("geometry_fit", "pruning_calibration")]
    evaluation = select_validation(manifest["rows"], args.eval_count, seed=0)
    selected = probe_rows(rows, args.probe_count, args.seed)
    c4_config = json.loads((args.c4_capture / "config.json").read_text())
    with np.load(args.c4_capture / "c4.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != digest(c4_config):
            raise ValueError("C4 source provenance mismatch")
        c4_tokens = saved["tokens"].copy()
    from sparsifier.prune import curvature_replay, two_sided, whole_neuron_metric
    from sparsifier.eval import gsm8k_input_tangents
    modules = (gsm8k_capture, gsm8k_sequential, gsm8k_pruning, downstream, factor_atlas,
               gram_dictionary, geometry_features, rescale, sparsegpt, curvature_replay, two_sided,
               whole_neuron_metric, gsm8k_input_tangents)
    sources = [Path(__file__)] + [Path(module.__file__) for module in modules]
    config = {"model": MODEL, "revision": REVISION, "arm": ARM if args.metric == "activation_diagonal" else "downstream_weightdiag",
              "manifest_sha256": manifest["manifest_sha256"], "pilot_report_sha256": pilot_hash,
              "calibration_regime": regime(manifest),
              "c4_config_sha256": digest(c4_config), "blocks": args.blocks, "modules": args.modules,
              "sparsity": args.sparsity, "alpha": args.alpha, "seed": args.seed,
              "max_new_tokens": args.max_new_tokens, "evaluation_ids": [row["id"] for row in evaluation],
              "calibration_ids": [row["id"] for row in rows],
              "pullback": dict(PULLBACK, num_probes=args.num_probes, probe_count=args.probe_count),
              "probe_ids": [row["id"] for row in selected],
              "atlas": {"groups": 8, "width": 4, "active": 2, "charts": 4, "rank": 1},
              "purpose": "development_sequential_MLP_retention",
              "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    config["two_sided"] = {"input_alpha": args.input_alpha, "input_structure": args.input_structure,
                           "version": two_sided.VERSION}
    config["full_metric"] = {"enabled": args.full_metric_refit, "rtol": args.full_metric_rtol,
        "max_iterations": args.full_metric_max_iterations, "cache_budget_bytes": args.replay_cache_mib*1024**2,
        "disk_budget_bytes": int(args.replay_disk_budget_gib*1024**3), "chunk_size": args.replay_chunk_size,
        "version": curvature_replay.VERSION}
    if args.input_alpha:
        config["arm"] += "_two_sided"
    if args.full_metric_refit:
        config["arm"] += "_full_metric_refit"
    if args.metric != "activation_diagonal":
        from importlib.metadata import version, PackageNotFoundError
        from sparsifier.models import qwen
        config["sources"]["qwen.py"] = hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest()
        config["runtime_versions"] = {}
        for package in ("torch", "transformers", "kernels", "flash-linear-attention"):
            try:
                config["runtime_versions"][package] = version(package)
            except PackageNotFoundError:
                config["runtime_versions"][package] = None
        config["backend_policy"] = "verified_pytorch_reference"
        config["pullback"]["metric"] = args.metric
        config["pullback"]["approximation"] = "shared_weight_diagonal_then_row_congruence"
    args.output.mkdir(parents=True, exist_ok=True)
    config["batching"] = batch_policy()
    write_immutable(args.output / "config.json", config)
    config_hash = digest(config)
    report_path = args.output / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["config_sha256"] != config_hash:
            raise ValueError("report provenance mismatch")
        if previous.get("complete"):
            return
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    memory = configure_cuda(56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        use_kernels=False, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer_count = len(model.model.layers)
    blocks = list(range(layer_count)) if args.blocks == "all" else [int(b) for b in args.blocks.split(",")]
    sites = sites_for(blocks, args.modules.split(","), layer_count)
    total = sum(parameter.numel() for parameter in model.parameters())
    eligible = sum(model.model.layers[b].get_submodule(m).weight.numel() for b, m in sites)
    upstream, stages, removed = config_hash, [], 0
    for index, site in enumerate(sites):
        progress = Progress(site=list(site), stage_index=index, sites_total=len(sites), arm=config["arm"])
        progress({"event": "site_started", "resumed": (args.output/f"site-{index:03d}.pt").exists()})
        block, module = site
        target = model.model.layers[block].get_submodule(module)
        anchor_index = anchor_block(block, layer_count)
        anchor = model.model.layers[anchor_index]
        path = args.output / f"site-{index:03d}.pt"
        if path.exists():
            stored = torch.load(path, weights_only=True, map_location="cpu")
            verify_stage(stored, config_hash, upstream, site)
        else:
            original = target.weight.detach().float().cpu()
            with progress.phase("calibration_and_pruning"):
                candidate, diagnostics = prune_site(
                    model, tokenizer, target, anchor, rows, c4_tokens, args.output, index,
                    seed=args.seed, sparsity=args.sparsity, alpha=args.alpha,
                    num_probes=args.num_probes, probe_count=args.probe_count, metric=args.metric,
                    input_alpha=args.input_alpha, input_structure=args.input_structure,
                    full_metric_refit=args.full_metric_refit, full_metric_rtol=args.full_metric_rtol,
                    full_metric_max_iterations=args.full_metric_max_iterations,
                    replay_cache_budget_bytes=args.replay_cache_mib*1024**2,
                    replay_disk_budget_bytes=int(args.replay_disk_budget_gib*1024**3),
                    replay_chunk_size=args.replay_chunk_size, progress=progress)
            candidate = candidate.to(torch.bfloat16).float()
            diagnostics["anchor_block"] = anchor_index
            stored = {"config_sha256": config_hash, "upstream": upstream, "site": list(site),
                      "weight": candidate, "diagnostics": diagnostics,
                      "removed": int(((original != 0) & (candidate == 0)).sum())}
            temporary = path.with_suffix(".tmp")
            torch.save(stored, temporary)
            temporary.replace(path)
            del candidate, original
        if stored["weight"].shape != target.weight.shape:
            raise ValueError("candidate shape mismatch")
        with torch.no_grad():
            target.weight.copy_(stored["weight"])
        removed += stored["removed"]
        upstream = hashlib.sha256(path.read_bytes()).hexdigest()
        stages.append({"site": list(site), "candidate_sha256": upstream,
                       "removed": stored["removed"], "diagnostics": stored["diagnostics"]})
        del stored
        torch.cuda.empty_cache()
        atomic_json(report_path, {"complete": False, "config_sha256": config_hash, "stages": stages,
                                  "removed_weights": removed, "total_parameters": total,
                                  "eligible_parameters": eligible})
        progress({"event": "site_completed", "sites_done": index+1,
                  "removed_weights": stages[-1]["removed"],
                  "metrics": compact_diagnostics(stages[-1]["diagnostics"])})
    evaluated = behavior(model, tokenizer, evaluation, args.output, config_hash,
                         max_new_tokens=args.max_new_tokens)
    atomic_json(report_path, {"complete": True, "config_sha256": config_hash, "stages": stages,
                             "behavior": evaluated, "removed_weights": removed,
                             "total_parameters": total, "eligible_parameters": eligible,
                             "whole_model_removed_fraction": removed / total,
                             "eligible_removed_fraction": removed / eligible,
                             "memory": memory, "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})


if __name__ == "__main__":
    main()
