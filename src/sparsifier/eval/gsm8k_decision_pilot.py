"""Small matched entrywise/whole-neuron candidate phases for decision64/light128."""
from __future__ import annotations
from sparsifier.eval.gsm8k_batching import policy as batch_policy
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import torch
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.data.gsm8k_calibration import CALIBRATION, load_calibration, regime
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.gsm8k_capture import check_competence, capture_batch
from sparsifier.eval.gsm8k_sequential import collect, verify_stage
from sparsifier.eval.gsm8k_pruning import prune_candidate
from sparsifier.eval.gsm8k_downstream import prune_site, AnchorPrefix, anchor_block, probe_rows
from sparsifier.eval.gsm8k_whole_neuron import reference_backend, package_versions, validate_backward
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.factor_atlas import FactorAtlas, projector_diagonals
from sparsifier.prune.gram_dictionary import GramBlockDictionary
from sparsifier.prune.whole_neuron_io import file_hash


def entrywise(root, task, output, *, calibration_path=None, probe_count=None, num_probes=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    manifest = load_calibration(calibration_path or root/CALIBRATION)
    probe_count = probe_count if probe_count is not None else manifest.get("backward_problem_count", 32)
    rows = [r for r in manifest["rows"] if r["split"] in ("geometry_fit","pruning_calibration")]
    capture_config = json.loads((root/"capture-v1/config.json").read_text())
    with np.load(root/"capture-v1/c4.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != digest(capture_config):
            raise ValueError("C4 provenance mismatch")
        tokens = saved["tokens"].copy()
    config = {"model":MODEL,"revision":REVISION,"arm":"entrywise_"+task["metric"],
        "structure":"entrywise","blocks":[task["block"]],"sparsity":task["sparsity"],
        "seed":0,"alpha":0 if task["metric"]=="ambient" else 1,
        "manifest_sha256":manifest["manifest_sha256"],"calibration_ids":[r["id"] for r in rows],
        "calibration_regime":regime(manifest), "probe_count":probe_count, "num_probes":num_probes,
        "c4_config_sha256":digest(capture_config),
        "pilot_sha256":check_competence(root/"qwen-9b-gsm8k-dense-pilot-v2/report.json"),
        "backend":reference_backend(),"packages":package_versions(),"purpose":"candidate_only_matched_block_pilot",
        "source_sha256":file_hash(Path(__file__))}
    import inspect
    repo = Path(__file__).resolve().parents[3]
    config["helper_sources"] = {str(Path(inspect.getfile(obj)).resolve().relative_to(repo)):file_hash(inspect.getfile(obj))
        for obj in (collect, capture_batch, prune_candidate, prune_site, FactorAtlas, GramBlockDictionary, reference_backend)}
    output.mkdir(parents=True, exist_ok=True)
    config["batching"] = batch_policy()
    write_immutable(output/"config.json", config)
    report_path = output/"report.json"
    if report_path.exists() and json.loads(report_path.read_text()).get("complete"):
        return
    torch.manual_seed(0)
    torch.set_num_threads(8)
    memory = configure_cuda(56.)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    stages, upstream, removed, eligible = [], digest(config), 0, 0
    for n, module in enumerate(("mlp.gate_proj","mlp.up_proj","mlp.down_proj")):
        site = [task["block"], module]
        layer = model.model.layers[site[0]].get_submodule(module)
        eligible += layer.weight.numel()
        path = output/f"site-{n:03d}.pt"
        if path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            verify_stage(stored, digest(config), upstream, site)
        else:
            original = layer.weight.detach().float().cpu()
            if task["metric"] == "downstream_weight_diagonal":
                anchor = model.model.layers[anchor_block(site[0], len(model.model.layers))]
                selected = probe_rows(rows, probe_count, 0)
                ids = tokenizer.encode(render_prompt(tokenizer, selected[0]["question"]), add_special_tokens=False)
                check = validate_backward(AnchorPrefix(model, anchor), layer, anchor, torch.tensor([ids], device="cuda"))
                candidate, diagnostics = prune_site(model, tokenizer, layer, anchor, rows, tokens,
                    output, n, sparsity=task["sparsity"], metric="weight_diagonal", probe_count=probe_count, num_probes=num_probes)
                diagnostics["backward_validation"] = check
            else:
                x, y, ambient, _ = collect(model, tokenizer, layer, rows, tokens, 0)
                pdiag = None
                if task["metric"] == "local":
                    atlas = FactorAtlas(8,4,2,4,1,0)
                    atlas.dictionary = GramBlockDictionary(8,4,2,0)
                    atlas.fit(y[np.array([r["split"]=="geometry_fit" for r in rows])])
                    atlas.save(output/f"atlas-{n:03d}.npz")
                    frames, _, _ = atlas.frames(y, mode="local")
                    pdiag = projector_diagonals(frames)
                candidate, diagnostics = prune_candidate(original, ambient, arm="sparsegpt_task" if pdiag is None else "local",
                    sparsity=task["sparsity"], alpha=config["alpha"], readout_inputs=torch.from_numpy(x), pdiag=pdiag)
                del x, y, ambient
            candidate = candidate.to(torch.bfloat16).float()
            stored = {"config_sha256":digest(config),"upstream":upstream,"site":site,
                "weight":candidate,"diagnostics":diagnostics,"removed":int(((original!=0)&(candidate==0)).sum())}
            tmp = path.with_suffix(".tmp")
            torch.save(stored,tmp); tmp.replace(path)
        with torch.no_grad():
            layer.weight.copy_(stored["weight"])
        removed += stored["removed"]
        upstream = file_hash(path)
        stages.append({"site":site,"candidate_sha256":upstream,"diagnostics":stored["diagnostics"]})
        del stored
        torch.cuda.empty_cache()
    atomic_json(report_path, {"complete":True,"config_sha256":digest(config),"stages":stages,
        "removed_weights":removed,"eligible_parameters":eligible,"total_parameters":sum(p.numel() for p in model.parameters()),
        "memory":memory,"phase":"candidate_only"})


def worker(root, plan_path, name, plan_hash):
    plan = json.loads(plan_path.read_text())
    if digest(plan) != plan_hash:
        raise ValueError("decision pilot plan drift")
    if plan.get("batching") != batch_policy():
        raise ValueError("decision pilot batching/runtime drift; use a plan matching the current batch policy")
    if load_manifest(root/plan["manifest"])["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("decision pilot evaluation manifest drift")
    matches = [t for t in plan["tasks"] if t["name"] == name]
    if len(matches) != 1:
        raise ValueError("pilot task not unique")
    task = matches[0]
    calibration_path = root/plan.get("calibration_manifest", "data/controlled-v1.json")
    calibration = load_calibration(calibration_path)
    if plan.get("calibration_manifest_sha256", calibration["manifest_sha256"]) != calibration["manifest_sha256"]:
        raise ValueError("pilot calibration manifest drift")
    probe_count = plan.get("probe_count", calibration.get("backward_problem_count", 32))
    num_probes = plan.get("num_probes", 8)
    if task["phase"] in ("dense","evaluate"):
        from sparsifier.eval.gsm8k_decision import main
        argv = ["--manifest",str(root/plan["manifest"]),"--calibration-manifest",str(calibration_path),
                "--output",str(root/name)]
        if task["phase"] == "evaluate":
            argv += ["--source",str(root/task["source"]),"--reference",str(root/plan["dense"])]
        sys.argv = ["decision",*argv]
        main()
    elif task["structure"] == "whole_neuron":
        from sparsifier.eval.gsm8k_whole_neuron import main
        sys.argv = ["whole_neuron","--manifest",str(calibration_path),
            "--pilot-report",str(root/"qwen-9b-gsm8k-dense-pilot-v2/report.json"),
            "--c4-capture",str(root/"capture-v1"),"--output",str(root/name),"--blocks",str(task["block"]),
            "--metric",task["metric"],"--sparsity",str(task["sparsity"]),"--seed","0","--eval-count","4",
            "--probe-count",str(probe_count),"--num-probes",str(num_probes),"--skip-behavior"]
        main()
    else:
        entrywise(root, task, root/name, calibration_path=calibration_path, probe_count=probe_count, num_probes=num_probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--task",required=True)
    args = parser.parse_args()
    worker(args.root,args.plan,args.task,args.plan_sha256)


if __name__ == "__main__":
    main()
