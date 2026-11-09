"""Additional matched controls: task gradients and whole-activation charts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.gsm8k_pruning import calibration, prune_candidate, select_validation, behavior
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.tangent_common import fit_output_chart_atlas


def whole_chart_diagonals(fit_outputs, assigned_outputs, *, charts=4, rank=2, seed=0):
    atlas = fit_output_chart_atlas(torch.as_tensor(fit_outputs).float(), charts=charts, rank=rank, seed=seed)
    assignment = torch.cdist(torch.as_tensor(assigned_outputs).float(), atlas.centers).argmin(1)
    diagonal = torch.stack([basis.square().sum(1) for basis in atlas.bases])
    return diagonal[assignment].numpy(), {
        "chart_counts": torch.bincount(assignment, minlength=len(atlas.bases)).tolist(),
        "chart_ranks": atlas.ranks,
        "storage_scalars": atlas.centers.numel()+sum(b.numel() for b in atlas.bases),
        "scope": "whole_activation_clustering_not_factor_decomposition"}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval.natural_layer import replaced_weight, vocabulary_reference, language_metrics
    from sparsifier.data.loaders import get_loaders
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("fisher","whole_chart"), required=True)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("alpha must be finite and nonnegative")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    capture,index,_,x,y = load_capture(args.capture)
    manifest = load_manifest(args.manifest)
    if capture["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("capture/manifest mismatch")
    rows = select_validation(manifest["rows"],32,seed=0)
    config = {"model":MODEL,"revision":REVISION,"arm":args.arm,"alpha":args.alpha,
              "seed":args.seed,"sparsity":.5,"capture_sha256":digest(capture),
              "evaluation_ids":[r["id"] for r in rows],"evaluation_split":"validation",
              "manifest_sha256":manifest["manifest_sha256"],
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "purpose":"development_single_layer_not_confirmatory"}
    args.output.mkdir(parents=True,exist_ok=True)
    inputs,_,use = calibration(args.capture,index,x,seed=args.seed)
    if args.arm == "whole_chart":
        train = np.array([r["split"] == "geometry_fit" for r in index["rows"]])
        pdiag, metric_info = whole_chart_diagonals(y[train],y[use],seed=args.seed)
        readout = torch.from_numpy(x[use])
    else:
        if args.sensitivity is None:
            parser.error("Fisher control requires sensitivity data")
        sensitivity_config = json.loads((args.sensitivity/"config.json").read_text())
        report = json.loads((args.sensitivity/"report.json").read_text())
        if not report.get("complete") or report["config_sha256"] != digest(sensitivity_config):
            raise ValueError("sensitivity data incomplete")
        if sensitivity_config["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("sensitivity manifest mismatch")
        with np.load(args.sensitivity/"aligned.npz",allow_pickle=False) as saved:
            if str(saved["config_sha256"]) != digest(sensitivity_config):
                raise ValueError("sensitivity array provenance mismatch")
            readout = torch.from_numpy(saved["inputs"].copy())
            pdiag = np.square(saved["gradients"])
        metric_info = {"scope":"empirical_fisher_style_answer_gradient_diagonal",
                       "normalization":"trace_relative_to_ambient", "not_model_fisher":True}
        config["sensitivity_sha256"] = digest(sensitivity_config)
    config["metric"] = metric_info
    write_immutable(args.output/"config.json",config)
    config_hash = digest(config)
    if (args.output/"report.json").exists() and json.loads((args.output/"report.json").read_text()).get("complete"):
        return
    memory = configure_cuda(56.)
    weight = torch.from_numpy(np.load(args.capture/"weight.npy"))
    candidate_path = args.output/"candidate.pt"
    if candidate_path.exists():
        saved = torch.load(candidate_path,weights_only=True,map_location="cpu")
        if saved["config_sha256"] != config_hash:
            raise ValueError("candidate provenance mismatch")
        candidate, diagnostics = saved["weight"],saved["diagnostics"]
    else:
        candidate, diagnostics = prune_candidate(weight,inputs,arm="local",alpha=args.alpha,
            readout_inputs=readout,pdiag=pdiag)
        candidate = candidate.to(torch.bfloat16).float()
        # Trace is a metric normalization, not a rank for the Fisher arm.
        diagnostics["mean_metric_trace"] = diagnostics.pop("mean_readout_rank")
        torch.save({"weight":candidate,"diagnostics":diagnostics,"config_sha256":config_hash},
                   candidate_path.with_suffix(".tmp"))
        candidate_path.with_suffix(".tmp").replace(candidate_path)
    del inputs,readout,pdiag
    torch.cuda.empty_cache()
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,use_kernels=False,attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer = model.model.layers[capture["block"]].get_submodule(capture["module"])
    if not torch.equal(layer.weight.detach().float().cpu(),weight):
        raise ValueError("model/capture weight mismatch")
    _,wiki = get_loaders("wikitext2",1,0,128,MODEL)
    windows = [wiki.input_ids[:,i*128:(i+1)*128] for i in range(8)]
    reference = vocabulary_reference(model,windows,"cuda")
    dense_nll = float(np.mean([r["nll"] for r in reference]))
    if not np.isfinite(dense_nll) or dense_nll > np.log(100):
        raise RuntimeError("implausible dense natural-text metric")
    with replaced_weight(layer,candidate):
        language = language_metrics(model,windows,reference,"cuda")
        evaluated = behavior(model,tokenizer,rows,args.output,config_hash)
    atomic_json(args.output/"report.json",{"complete":True,"config_sha256":config_hash,
        "behavior":evaluated,"language":language,"dense_nll":dense_nll,
        "local":{"actual_sparsity":float((candidate==0).float().mean()),"solver":diagnostics},
        "memory":memory,"peak_cuda_reserved_gib":torch.cuda.max_memory_reserved()/1024**3})


if __name__ == "__main__":
    main()
