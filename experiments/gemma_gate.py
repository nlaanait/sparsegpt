"""E2B correctness gate.

Dense layer-wise WikiText2 and PTB perplexity must match a full forward on the
same windows. A 50% unstructured prune must hit the decoder linears that exist
and leave embeddings, per-layer embeddings, norms, and any encoder weights alone.
"""

from __future__ import annotations

import torch

from qwen_ptb import _prune_args
from sparsegpt.datautils import get_loaders
from sparsegpt.gemma import (
    gemma_eval,
    gemma_reference_nlls,
    gemma_sequential,
    get_gemma,
    is_prunable_parameter,
)
from sparsegpt.modelutils import DEV
import sparsegpt.gemma as gemma_mod

MODEL = "google/gemma-4-E2B-it"


def _perplexity(nlls, seqlen: int) -> float:
    total = torch.stack(nlls).sum()
    return torch.exp(total / (len(nlls) * seqlen)).item()


def _checksum(model) -> dict[str, float]:
    sums = {}
    for name, param in model.named_parameters():
        if is_prunable_parameter(name):
            continue
        sums[name] = float(param.detach().float().sum())
    return sums


def main() -> None:
    model = get_gemma(MODEL)
    model.eval()
    for dataset in ("ptb", "wikitext2"):
        _loader, testenc = get_loaders(dataset, seed=0, model=MODEL, seqlen=model.seqlen)
        reference = gemma_reference_nlls(model, testenc, DEV)
        layer_ppl = gemma_eval(model, testenc, DEV, dataset).item()
        ref_ppl = _perplexity(reference, model.seqlen)
        gap = abs(layer_ppl - ref_ppl) / ref_ppl
        print(f"{dataset} full={ref_ppl:.6f} layerwise={layer_ppl:.6f} rel={gap:.3e}", flush=True)
        if gap > 1e-3:
            raise SystemExit(f"{dataset} layer-wise perplexity does not match the full forward")
        from analysis.sweep import add_point, load_results, result_path, save_results
        from experiments.gemma import GEMMA

        path = result_path(GEMMA, MODEL)
        results = load_results(path, GEMMA)
        results["model"] = MODEL
        add_point(results, MODEL, dataset, "dense", 0.0, layer_ppl)
        save_results(path, results)

    before = _checksum(model)
    gemma_mod.args = _prune_args(MODEL, 0.5)
    dataloader, _test = get_loaders("c4", nsamples=128, seed=0, model=MODEL, seqlen=model.seqlen)
    gemma_sequential(model, dataloader, DEV)
    after = _checksum(model)
    changed = [name for name in before if before[name] != after.get(name)]
    if changed:
        raise SystemExit("prune changed weights that are not decoder linears: " + ", ".join(changed[:12]))
    seen = 0
    for name, param in model.named_parameters():
        if not is_prunable_parameter(name):
            continue
        seen += 1
        fraction = float(torch.mean((param == 0).float()))
        print(f"{name} sparsity={fraction:.4f}", flush=True)
        if abs(fraction - 0.5) > 0.02:
            raise SystemExit(f"{name} sparsity {fraction:.4f} is not 50%")
    if seen == 0:
        raise SystemExit("no decoder linears were pruned")
    from sparsegpt.prune_cache import prune_config_from_args, resolve_save_dir, write_prune_config

    config = prune_config_from_args(gemma_mod.args)
    dest = resolve_save_dir("sparse_opt/gemma-4-e2b-it", config)
    print(f"Saving pruned model to {dest}", flush=True)
    model.save_pretrained(dest)
    write_prune_config(dest, config)
    print(f"gate passed: {seen} decoder linears at 50%, protected weights unchanged", flush=True)


if __name__ == "__main__":
    main()
