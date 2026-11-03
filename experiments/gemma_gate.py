"""E2B correctness gate.

Dense layer-wise WikiText2 and PTB perplexity must match a full forward on the
same windows. A 50% unstructured prune must hit the decoder linears that exist
and leave embeddings, per-layer embeddings, norms, and any encoder weights alone.
The agreement is not a score and is not written into the base sweep.
"""

from __future__ import annotations

import torch

from sparsifier.data.cache import resolve_save_dir, write_prune_config
from sparsifier.data.loaders import get_loaders
from sparsifier.models.gemma import (
    GEMMA,
    gemma_eval,
    gemma_reference_nlls,
    get_gemma,
    is_prunable_parameter,
    reference_gap,
)
from sparsifier.models.layers import DEV
from sparsifier.models.loop import pruner_for
from sparsifier.sweep import RunConfig, default_prune_config

MODEL = "google/gemma-4-E2B-it"


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
        gap = reference_gap(layer_ppl, reference, model.seqlen)
        print(f"{dataset} layerwise={layer_ppl:.6f} rel={gap:.3e}", flush=True)
        if gap > 1e-3:
            raise SystemExit(f"{dataset} layer-wise perplexity does not match the full forward")

    before = _checksum(model)
    config = RunConfig(model=MODEL, calibration="c4", sparsity=0.5)
    dataloader, _test = get_loaders(
        "c4",
        nsamples=config.nsamples,
        seed=config.seed,
        model=MODEL,
        seqlen=model.seqlen,
    )
    GEMMA.prune_model(model, dataloader, config, pruner_for(config))
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
    saved = default_prune_config(MODEL, 0.5)
    dest = resolve_save_dir("sparse_opt/gemma-4-e2b-it", saved)
    print(f"Saving pruned model to {dest}", flush=True)
    model.save_pretrained(dest)
    write_prune_config(dest, saved)
    print(f"gate passed: {seen} decoder linears at 50%, protected weights unchanged", flush=True)


if __name__ == "__main__":
    main()
