"""Sequential GSM8K MLP pruning with downstream activation tangent pullbacks.

Run through the guarded gflow worker. This driver keeps the existing 4096
C4 + 4096 math ambient budget, fits factor charts on geometry_fit readouts,
and uses a recorded pruning_calibration subset for extra backward compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_capture import check_competence
from sparsifier.eval.gsm8k_pruning import behavior, select_validation
from sparsifier.eval.gsm8k_sequential import collect, sites_for, verify_stage
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.prune.downstream import DownstreamTangentCalibrator, TangentAnchor
from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.gram_dictionary import GramBlockDictionary


ARM = "downstream"
PULLBACK = {"anchor_offset": 1, "num_probes": 8, "probe_count": 32,
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
               *, seed=0, sparsity=.5, alpha=1., num_probes=8, probe_count=32):
    """Refresh ambient/anchor activations and calibrate one current-model site."""
    if any(row["split"] not in ("geometry_fit", "pruning_calibration") for row in rows):
        raise ValueError("geometry and pruning must exclude held-out examples")
    selected = probe_rows(rows, probe_count, seed)
    prefix = AnchorPrefix(model, anchor)
    anchor_readouts = []

    def capture(_module, _args, value):
        hidden = value[0] if isinstance(value, tuple) else value
        anchor_readouts.append(hidden[0, -1].detach().float().cpu().numpy())

    handle = anchor.register_forward_hook(capture)
    try:
        # Same selected math positions and C4 windows as existing sequential arms.
        x, y, inputs, c4 = collect(prefix, tokenizer, target, rows, c4_tokens, seed)
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
    atlas.fit(downstream[fit])
    atlas_path = output / f"atlas-{stage_index:03d}.npz"
    atlas.save(atlas_path)
    del x, y, c4, downstream, anchor_readouts
    calibration = DownstreamTangentCalibrator(
        {"target": target},
        {"downstream": TangentAnchor(anchor, lambda value: readout_basis(atlas, value),
                                      select=lambda value: value[0] if isinstance(value, tuple) else value)},
        alpha=alpha, num_probes=num_probes, seed=seed,
    )
    solver = calibration.solvers["target"]
    solver.add_ambient_batch(inputs)
    del inputs
    device = target.weight.device
    for index, row in enumerate(selected):
        ids = tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False)
        if not 0 < len(ids) <= 2048:
            raise ValueError("backward calibration prompt exceeds budget")
        tokens = torch.tensor([ids], device=device)
        calibration.run(lambda: prefix(input_ids=tokens, use_cache=False, logits_to_keep=1),
                        collect_ambient=False)
        print(json.dumps({"pullback_problem": index + 1, "pullback_total": len(selected),
                          "stage_index": stage_index}), flush=True)
    if float(solver.geometry_energy.sum()) <= 0:
        raise ValueError("no supported downstream geometry; refuse geometric pruning arm")
    # Match the existing experiment's float32 compensation and bf16 output cast.
    # Do not mutate the real model until its complete stage checkpoint is saved.
    temporary = nn.Linear(target.weight.shape[1], target.weight.shape[0], bias=False,
                          device=device, dtype=torch.float32)
    with torch.no_grad():
        temporary.weight.copy_(target.weight)
    solver.layer = temporary
    solver.fasterprune(sparsity, blocksize=128, percdamp=.01)
    candidate = temporary.weight.detach().cpu().clone()
    diagnostics = dict(solver.tangent_diagnostics,
                       solver="downstream_tangent_row_diagonal_congruence",
                       probe_ids=[row["id"] for row in selected],
                       geometry_fit_ids=[row["id"] for row in rows if row["split"] == "geometry_fit"],
                       atlas_sha256=hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
                       num_probes=num_probes, damping=.01, blocksize=128,
                       model_precision="bfloat16", compensation_precision="float32")
    solver.free()
    return candidate, diagnostics


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval import gsm8k_capture, gsm8k_sequential, gsm8k_pruning
    from sparsifier.prune import downstream, factor_atlas, gram_dictionary, geometry_features, rescale, sparsegpt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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
    parser.add_argument("--probe-count", type=int, default=PULLBACK["probe_count"])
    args = parser.parse_args()
    if not 0 <= args.sparsity < 1 or not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("invalid sparsity or alpha")
    if min(args.num_probes, args.probe_count, args.eval_count, args.max_new_tokens) < 1:
        parser.error("positive probe and evaluation budgets required")
    pilot_hash = check_competence(args.pilot_report)
    manifest = load_manifest(args.manifest)
    rows = [row for row in manifest["rows"] if row["split"] in ("geometry_fit", "pruning_calibration")]
    evaluation = select_validation(manifest["rows"], args.eval_count, seed=0)
    selected = probe_rows(rows, args.probe_count, args.seed)
    c4_config = json.loads((args.c4_capture / "config.json").read_text())
    with np.load(args.c4_capture / "c4.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != digest(c4_config):
            raise ValueError("C4 source provenance mismatch")
        c4_tokens = saved["tokens"].copy()
    modules = (gsm8k_capture, gsm8k_sequential, gsm8k_pruning, downstream, factor_atlas,
               gram_dictionary, geometry_features, rescale, sparsegpt)
    sources = [Path(__file__)] + [Path(module.__file__) for module in modules]
    config = {"model": MODEL, "revision": REVISION, "arm": ARM,
              "manifest_sha256": manifest["manifest_sha256"], "pilot_report_sha256": pilot_hash,
              "c4_config_sha256": digest(c4_config), "blocks": args.blocks, "modules": args.modules,
              "sparsity": args.sparsity, "alpha": args.alpha, "seed": args.seed,
              "max_new_tokens": args.max_new_tokens, "evaluation_ids": [row["id"] for row in evaluation],
              "calibration_ids": [row["id"] for row in rows],
              "pullback": dict(PULLBACK, num_probes=args.num_probes, probe_count=args.probe_count),
              "probe_ids": [row["id"] for row in selected],
              "atlas": {"groups": 8, "width": 4, "active": 2, "charts": 4, "rank": 1},
              "purpose": "development_sequential_MLP_retention",
              "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    args.output.mkdir(parents=True, exist_ok=True)
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
            candidate, diagnostics = prune_site(
                model, tokenizer, target, anchor, rows, c4_tokens, args.output, index,
                seed=args.seed, sparsity=args.sparsity, alpha=args.alpha,
                num_probes=args.num_probes, probe_count=args.probe_count)
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
        print(json.dumps({"completed_site": site, "sites_done": index + 1,
                          "sites_total": len(sites), "host": host_memory()}), flush=True)
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
