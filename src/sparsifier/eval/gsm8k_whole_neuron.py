"""One transaction per dense SwiGLU triplet; no entrywise gate/up pruning.

GPU execution belongs in the existing guarded gflow workflow. This module
does not submit jobs. Whole-neuron stages use a separate schema/version.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
import inspect
import json
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import torch
from torch.nn import functional as F
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_capture import check_competence
from sparsifier.eval.gsm8k_pruning import behavior, select_validation
from sparsifier.eval.gsm8k_sequential import collect, sites_for
from sparsifier.eval.gsm8k_downstream import AnchorPrefix, anchor_block, probe_rows, readout_basis
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.prune.whole_neuron import (
    VERSION, METRICS, SELECTORS, WholeNeuronConfig, SwiGLUTriplet, prune_triplet, apply_candidate,
)
from sparsifier.prune.whole_neuron_metric import FrozenNeuronMetric, NeuronProbeConsumer, StreamedGroupMetric
from sparsifier.prune.whole_neuron_io import save_stage, load_stage, file_hash
from sparsifier.prune.downstream import DownstreamTangentCalibrator, TangentAnchor
from sparsifier.prune.factor_atlas import FactorAtlas, projector_diagonals
from sparsifier.prune.gram_dictionary import GramBlockDictionary

MODULES = "mlp.gate_proj,mlp.up_proj,mlp.down_proj"


def triplet_sites(blocks, modules, layer_count):
    if set(modules) != set(MODULES.split(",")) or len(modules) != 3:
        raise ValueError("whole-neuron stages require all three projections, once each")
    sites_for(blocks, modules, layer_count)
    return [(b, "mlp_triplet") for b in blocks]


def memory_estimate(hidden, width, sparsity, *, element_bytes=4):
    kept = width-int(sparsity*width)
    # Shared curvature, inverse/factor work, scaled weights and row-chunk refit.
    return {"shared_solver_buffer_estimate_bytes": element_bytes*(4*width**2+kept**2+6*hidden*width),
            "retained_factor_width": kept, "refit_factorization_order": "O(retained_width^3)",
            "excludes": "model, backward graphs, atlas and framework workspace; not a peak-memory guarantee"}


def reference_backend():
    """Strict opt-in for this driver; do not change historical helper semantics."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    identities = {}
    for name in _PYTORCH_ATTENTION:
        fn = getattr(modeling, name)
        seen = set()
        while True:
            if id(fn) in seen:
                raise RuntimeError("cyclic reference backend wrapper")
            seen.add(id(fn))
            if isinstance(fn, torch.nn.Module):
                closure = inspect.getclosurevars(fn.forward).nonlocals
                if "func" not in closure:
                    raise RuntimeError(f"unsupported backend module wrapper for {name}")
                fn = closure["func"]
            elif hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
            else:
                break
        if not inspect.isfunction(fn) or fn.__name__ != name:
            raise RuntimeError(f"could not verify reference backend for {name}")
        setattr(modeling, name, fn)
        identities[name] = fn.__module__+"."+fn.__qualname__
    return identities


def package_versions():
    result = {}
    for name in ("torch", "transformers", "kernels", "flash-linear-attention"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_backward(prefix, target, anchor, tokens, *, rtol=.2, atol=1e-3):
    """Actual chosen backend and deployment-dtype signed weight perturbation.

    Compare finite differences with the *actual cast perturbation*. A tiny
    response cannot pass solely by falling below the tolerance. Runs before
    geometric pruning in each block; restores weights/hooks on all failures.
    """
    captured = {}
    def at_target(module, args, output):
        output = output if output.requires_grad else output.detach().requires_grad_(True)
        captured["x"], captured["y"] = args[0].detach(), output
        return output
    def at_anchor(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["scalar"] = value[0, -1, 0]
    handles = [target.register_forward_hook(at_target), anchor.register_forward_hook(at_anchor)]
    original = target.weight.detach().clone()
    try:
        with torch.enable_grad():
            prefix(input_ids=tokens, use_cache=False, logits_to_keep=1)
            gradient, = torch.autograd.grad(captured["scalar"], captured["y"])
        z = gradient[0].float().T @ captured["x"][0].float()
        coordinate = int(z.abs().argmax())
        derivative = z.flatten()[coordinate]
        if not torch.isfinite(derivative) or derivative.abs() < 1e-8:
            raise RuntimeError("backward validation readout has no finite measurable response")
        for attempt in range(5):
            step = .01*2**attempt
            values, weights = [], []
            for sign in (1, -1):
                perturbed = original.float().clone()
                perturbed.flatten()[coordinate] += sign*step
                perturbed = perturbed.to(original.dtype)
                with torch.no_grad():
                    target.weight.copy_(perturbed)
                    prefix(input_ids=tokens, use_cache=False, logits_to_keep=1)
                values.append(captured["scalar"].detach().float())
                weights.append(perturbed.flatten()[coordinate].float())
            observed = (values[0]-values[1])/2
            predicted = derivative*(weights[0]-weights[1])/2
            if observed.abs() > atol and torch.isclose(observed, predicted, rtol=rtol, atol=atol):
                return {"passed": True, "observed_half_difference": float(observed),
                        "predicted_half_difference": float(predicted), "step": step,
                        "coordinate": coordinate, "rtol": rtol, "atol": atol,
                        "dtype": str(original.dtype), "target": "down_weight_and_down_output"}
        raise RuntimeError("actual-backend production-precision finite-change validation failed")
    finally:
        with torch.no_grad():
            target.weight.copy_(original)
        for handle in handles:
            handle.remove()


def calibration_positions(tokenizer, rows, c4_tokens, seed):
    pool, readouts = [], []
    for n, row in enumerate(rows):
        length = len(tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False))
        positions = np.unique(np.append(np.random.default_rng(seed+n).choice(length, min(32, length), replace=False), length-1))
        readouts.append([row["id"], length-1])
        pool.extend([row["id"], int(p)] for p in positions if p != length-1)
    needed = 4096-len(readouts)
    if needed < 0 or needed > len(pool):
        raise ValueError("insufficient ambient math token pool")
    selection = np.random.default_rng(seed).choice(len(pool), needed, replace=False)
    return {"math": readouts+[pool[i] for i in selection],
            "c4": [[int(window), int(position)] for window, tokens in enumerate(c4_tokens) for position in range(len(tokens))],
            "geometric_readouts": readouts,
            "convention": "fit and pruning-calibration readouts included in ambient, padding absent"}


def calibrate(model, tokenizer, block, rows, c4_tokens, config, output, *,
              num_probes=8, probe_count=32, cache_budget_bytes=256*1024**2):
    start = time.monotonic()
    if any(row["split"] not in ("geometry_fit", "pruning_calibration") for row in rows):
        raise ValueError("calibration contains held-out data")
    triplet = SwiGLUTriplet(model.model.layers[block].mlp)
    geometry = config.metric != "ambient" and config.alpha > 0 and int(config.sparsity*len(triplet.gate.weight)) > 0
    is_downstream = geometry and config.metric.startswith("downstream_")
    index = anchor_block(block, len(model.model.layers)) if is_downstream else block
    anchor = model.model.layers[index] if is_downstream else triplet.down
    prefix = AnchorPrefix(model, anchor)
    captured = []
    def capture(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value[0, -1].detach().float().cpu().numpy())
    handle = anchor.register_forward_hook(capture) if geometry else None
    try:
        x, y, ambient, c4 = collect(prefix, tokenizer, triplet.down, rows, c4_tokens, config.seed)
    finally:
        if handle is not None:
            handle.remove()
    if len(ambient) != 8192 or len(c4) != 4096:
        raise ValueError("matched ambient budget requires 4096 C4 + 4096 math tokens")
    metadata = {"ambient_positions": calibration_positions(tokenizer, rows, c4_tokens, config.seed),
                "anchor_block": index, "target": "mlp.down_proj", "probe_ids": [],
                "probe_seed": config.seed, "num_probes": num_probes if is_downstream else 0}
    if not geometry:
        metadata["calibration_seconds"] = time.monotonic()-start
        return triplet, ambient, None, None, metadata
    fit = np.array([row["split"] == "geometry_fit" for row in rows])
    atlas = FactorAtlas(8, 4, 2, 4, 1, config.seed)
    atlas.dictionary = GramBlockDictionary(8, 4, 2, config.seed)
    atlas.fit(np.stack(captured[:len(rows)])[fit])
    atlas_path = output/f"atlas-{block:03d}.npz"
    atlas.save(atlas_path)
    metadata["atlas_sha256"] = file_hash(atlas_path)
    if not is_downstream:
        frames, active, assignment = atlas.frames(y, mode="local")
        q = [torch.as_tensor(frame, dtype=torch.float32) for frame in frames]
        metric = FrozenNeuronMetric(len(ambient), local_inputs=torch.from_numpy(x), frames=q,
                                   cache_budget_bytes=cache_budget_bytes)
        k = metric.diagonal(triplet.down.weight.shape, dtype=torch.float32)
        metadata.update(geometry_fit_ids=[row["id"] for row in rows if row["split"] == "geometry_fit"],
                        geometric_readout_ids=[row["id"] for row in rows],
                        mean_union_rank=float(np.mean([frame.shape[1] for frame in frames])))
    else:
        selected = probe_rows(rows, probe_count, config.seed)
        first_ids = tokenizer.encode(render_prompt(tokenizer, selected[0]["question"]), add_special_tokens=False)
        tokens = torch.tensor([first_ids], device=triplet.down.weight.device)
        metadata["backward_validation"] = validate_backward(prefix, triplet.down, anchor, tokens)
        consumer = NeuronProbeConsumer(triplet.down.weight,
            cache_budget_bytes=cache_budget_bytes if config.compensation == "full_metric_refit" else 0)
        mode = config.metric.removeprefix("downstream_")
        calibration = DownstreamTangentCalibrator({"down": triplet.down},
            {"next_block": TangentAnchor(anchor, lambda out: readout_basis(atlas, out),
                select=lambda out: out[0] if isinstance(out, tuple) else out)},
            alpha=config.alpha, metric=mode, num_probes=num_probes, seed=config.seed,
            probe_consumer=consumer)
        for row in selected:
            ids = tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False)
            tokens = torch.tensor([ids], device=triplet.down.weight.device)
            if not 0 < len(ids) <= 2048:
                raise ValueError("backward calibration exceeds prompt budget")
            calibration.run(lambda: prefix(input_ids=tokens, use_cache=False, logits_to_keep=1), collect_ambient=False)
        k = calibration.solvers["down"].geometry_energy/len(ambient)
        metric = (FrozenNeuronMetric(len(ambient), probes=consumer.probes, cache_budget_bytes=cache_budget_bytes)
                  if consumer.probes else StreamedGroupMetric(consumer.group_scores, len(ambient)))
        metadata.update(probe_ids=[row["id"] for row in selected], probe_contractions=consumer.count,
                        cached_probe_bytes=consumer.cache_bytes,
                        full_direction_group_energy=consumer.group_scores.tolist())
        # Do not retain an unnecessary entrywise Hessian at the group solve.
        calibration.solvers["down"].free()
    metadata["calibration_seconds"] = time.monotonic()-start
    return triplet, ambient, k, metric, metadata


def cast_finite_check(candidate, diagnostic_input):
    with torch.no_grad():
        result = F.linear(F.silu(F.linear(diagnostic_input, candidate.gate))*
                          F.linear(diagnostic_input, candidate.up), candidate.down)
    if not torch.isfinite(result).all():
        raise ValueError("deployment-cast triplet produced nonfinite diagnostic outputs")
    return {"finite": True, "positions": diagnostic_input.numel()//diagnostic_input.shape[-1]}


def restore_chain(model, source):
    config = json.loads((source/"config.json").read_text())
    report = json.loads((source/"report.json").read_text())
    if report["config_sha256"] != digest(config) or not report.get("complete") or config["method_version"] != VERSION:
        raise ValueError("whole-neuron source is incomplete or incompatible")
    if [stage["site"] for stage in report["stages"]] != config["sites"]:
        raise ValueError("whole-neuron source stage scope mismatch")
    upstream = digest(config)
    kept = {}
    for n, stage in enumerate(report["stages"]):
        block, kind = stage["site"]
        if kind != "mlp_triplet" or block in kept:
            raise ValueError("duplicate or non-triplet stage")
        triplet = SwiGLUTriplet(model.model.layers[block].mlp)
        candidate = load_stage(source/f"stage-{n:03d}.pt", triplet,
            expected_hash=stage["sha256"], config_hash=digest(config), upstream=upstream, block=block)
        apply_candidate(triplet, candidate)
        kept[block] = candidate.keep.cpu()
        upstream = stage["sha256"]
    return config, report, kept


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from sparsifier.eval import gsm8k_capture, gsm8k_sequential, gsm8k_downstream
    from sparsifier.prune import whole_neuron, whole_neuron_metric, whole_neuron_io, downstream, rescale, factor_atlas, gram_dictionary
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pilot-report", "c4-capture", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--modules", default=MODULES)
    parser.add_argument("--metric", choices=METRICS, default="ambient")
    parser.add_argument("--selector", choices=SELECTORS, default="block_obs")
    parser.add_argument("--compensation", choices=("none", "surrogate_refit", "full_metric_refit"), default="surrogate_refit")
    parser.add_argument("--unsupported-policy", choices=("protect", "allow"), default="protect")
    parser.add_argument("--sparsity", type=float, default=.3)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--num-probes", type=int, default=8)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--cache-budget-mib", type=int, default=256)
    args = parser.parse_args()
    cfg = WholeNeuronConfig(args.sparsity, args.metric,
        alpha=(0. if args.metric == "ambient" else 1.) if args.alpha is None else args.alpha,
        selector=args.selector, compensation=args.compensation, blocksize=args.blocksize,
        percdamp=args.percdamp, unsupported_policy=args.unsupported_policy, seed=args.seed)
    if min(args.eval_count, args.num_probes, args.probe_count, args.max_new_tokens, args.cache_budget_mib) < 1:
        parser.error("positive evaluation/probe/cache budgets required")
    pilot_hash = check_competence(args.pilot_report)
    manifest = load_manifest(args.manifest)
    rows = [row for row in manifest["rows"] if row["split"] in ("geometry_fit", "pruning_calibration")]
    evaluation = select_validation(manifest["rows"], args.eval_count, seed=0)
    c4_config = json.loads((args.c4_capture/"config.json").read_text())
    with np.load(args.c4_capture/"c4.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != digest(c4_config):
            raise ValueError("C4 capture provenance mismatch")
        c4_tokens = saved["tokens"].copy()
    hf_config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    blocks = list(range(text_config.num_hidden_layers)) if args.blocks == "all" else [int(b) for b in args.blocks.split(",")]
    sites = triplet_sites(blocks, args.modules.split(","), text_config.num_hidden_layers)
    if cfg.compensation == "full_metric_refit" and cfg.alpha > 0 and text_config.hidden_size*text_config.intermediate_size > 2048:
        parser.error("production full-metric refit requires deterministic replay; use bounded references (<=2048 coordinates)")
    # Match execution across ambient/local/downstream cells in this new method.
    backend = reference_backend()
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__)] + [Path(module.__file__) for module in (gsm8k_capture, gsm8k_sequential,
        gsm8k_downstream, whole_neuron, whole_neuron_metric, whole_neuron_io, downstream, rescale, factor_atlas, gram_dictionary)]
    config = {"method_version": VERSION, "arm": "whole_neuron_"+cfg.metric,
        "model": MODEL, "revision": REVISION, "manifest_sha256": manifest["manifest_sha256"],
        "pilot_sha256": pilot_hash, "c4_config_sha256": digest(c4_config), "pruning": asdict(cfg),
        "sites": [list(s) for s in sites], "calibration_ids": [r["id"] for r in rows],
        "evaluation_ids": [r["id"] for r in evaluation], "max_new_tokens": args.max_new_tokens,
        "num_probes": args.num_probes, "probe_count": args.probe_count,
        "cache_budget_bytes": args.cache_budget_mib*1024**2, "backend": backend, "packages": package_versions(),
        "thinking": False, "purpose": "whole_neuron_development_not_confirmatory",
        "memory_estimate": memory_estimate(text_config.hidden_size, text_config.intermediate_size, cfg.sparsity),
        "sources": {str(path.relative_to(repo)): file_hash(path) for path in sources}}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/"config.json", config)
    config_hash = digest(config)
    report_path = args.output/"report.json"
    old = json.loads(report_path.read_text()) if report_path.exists() else {"stages": []}
    if old.get("config_sha256", config_hash) != config_hash:
        raise ValueError("whole-neuron report configuration drift")
    if old.get("complete"):
        return
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    memory = configure_cuda(56.)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    eligible = sum(sum(w.numel() for w in SwiGLUTriplet(model.model.layers[b].mlp).weights) for b, _ in sites)
    upstream, stages = config_hash, []
    removed = architectural = 0
    for n, (block, _) in enumerate(sites):
        triplet = SwiGLUTriplet(model.model.layers[block].mlp)
        path = args.output/f"stage-{n:03d}.pt"
        if path.exists():
            if n < len(old["stages"]):
                checksum = old["stages"][n]["sha256"]
            elif path.with_suffix(".sha256").exists():
                checksum = path.with_suffix(".sha256").read_text().strip()
            else:
                raise ValueError("orphan stage lacks a committed hash; reconcile before resuming")
            candidate = load_stage(path, triplet, expected_hash=checksum, config_hash=config_hash,
                                   upstream=upstream, block=block)
        else:
            triplet, ambient, k, full, metadata = calibrate(model, tokenizer, block, rows, c4_tokens,
                cfg, args.output, num_probes=args.num_probes, probe_count=args.probe_count,
                cache_budget_bytes=config["cache_budget_bytes"])
            candidate = prune_triplet(triplet, ambient, cfg, geometric_diagonal=k, full_metric=full)
            # Capture an actual normalized MLP input from a held-out diagnostic prompt.
            diagnostic = {}
            def capture_input(module, inputs):
                diagnostic["x"] = inputs[0][:, :8].detach().clone()
            handle = triplet.mlp.register_forward_pre_hook(capture_input)
            try:
                ids = tokenizer.encode(render_prompt(tokenizer, evaluation[0]["question"]), add_special_tokens=False)
                with torch.inference_mode():
                    AnchorPrefix(model, triplet.down)(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
            finally:
                handle.remove()
            metadata["cast_output_check"] = cast_finite_check(candidate, diagnostic["x"])
            metadata["diagnostic_validation_id"] = evaluation[0]["id"]
            checksum = save_stage(path, triplet, candidate, config_hash=config_hash,
                                  upstream=upstream, block=block, metadata=metadata)
            del ambient, k, full
        apply_candidate(triplet, candidate)
        diagnostic = candidate.diagnostics
        removed += diagnostic.get("original_nonzero_removed", 0)
        architectural += diagnostic.get("architectural_removed_slots", 0)
        stages.append({"site": [block, "mlp_triplet"], "sha256": checksum, "diagnostics": diagnostic})
        upstream = checksum
        atomic_json(report_path, {"complete": False, "config_sha256": config_hash, "stages": stages,
            "removed_weights": removed, "architectural_removed_slots": architectural,
            "eligible_parameters": eligible, "total_parameters": total})
        del candidate
        torch.cuda.empty_cache()
        print(json.dumps({"completed_block": block, "host": host_memory()}), flush=True)
    evaluated = behavior(model, tokenizer, evaluation, args.output, config_hash, max_new_tokens=args.max_new_tokens)
    atomic_json(report_path, {"complete": True, "config_sha256": config_hash, "stages": stages,
        "behavior": evaluated, "removed_weights": removed, "architectural_removed_slots": architectural,
        "eligible_parameters": eligible, "total_parameters": total,
        "whole_model_removed_fraction": removed/total, "eligible_removed_fraction": removed/eligible,
        "architectural_removed_fraction": architectural/eligible, "memory": memory,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved()/1024**3})


if __name__ == "__main__":
    main()
