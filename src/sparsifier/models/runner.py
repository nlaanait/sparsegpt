"""Dispatch a Click request to the family that owns the checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class RunRequest:
    model: str
    calibration: str
    seed: int
    nsamples: int
    percdamp: float
    sparsity: float
    zeros_per_group: int
    group_size: int
    blocksize: int
    wbits: int
    minlayer: int
    maxlayer: int
    prune_only: str
    invert: bool
    save: str
    true_sequential: bool
    log_wandb: bool
    eval_datasets: tuple[str, ...]
    skip_eval: bool
    load_cached: bool
    pytorch_attention: bool
    forward_batch: int
    geometry_alpha: float
    geometry_probes: Path | None
    tangent_alpha: float
    tangent_charts: int
    tangent_rank: int
    tangent_samples: int
    tangent_factors: int = 8
    tangent_factor_source: str = "kmeans"
    tangent_active_factors: int = 2
    tangent_block_width: int = 3
    tangent_anchor_eta: float = 0.0
    tangent_fit_nsamples: int = 0
    tangent_fit_seed: int | None = None
    tangent_exact_mode: str = "diag"
    tangent_exact_max_params: int = 4096
    pruner: str | None = None
    bias_fold: bool = False
    family: str = ""


def run_family(request: RunRequest) -> None:
    """Run the family loop. Magnitude pruning is not part of this command."""
    from sparsifier.models import gemma, llama, olmo, qwen

    namespace = SimpleNamespace(
        model=request.model,
        dataset=request.calibration,
        seed=request.seed,
        nsamples=request.nsamples,
        percdamp=request.percdamp,
        sparsity=request.sparsity,
        zeros_per_group=request.zeros_per_group,
        group_size=request.group_size,
        blocksize=request.blocksize,
        wbits=request.wbits,
        minlayer=request.minlayer,
        maxlayer=request.maxlayer,
        prune_only=request.prune_only,
        invert=request.invert,
        save=request.save,
        true_sequential=request.true_sequential,
        log_wandb=request.log_wandb,
        eval_datasets=request.eval_datasets,
        skip_eval=request.skip_eval,
        load_cached=request.load_cached,
        pytorch_attention=request.pytorch_attention,
        forward_batch=request.forward_batch,
        geometry_alpha=request.geometry_alpha,
        geometry_probes=request.geometry_probes,
        tangent_alpha=request.tangent_alpha,
        tangent_charts=request.tangent_charts,
        tangent_rank=request.tangent_rank,
        tangent_samples=request.tangent_samples,
        tangent_factors=request.tangent_factors,
        tangent_factor_source=request.tangent_factor_source,
        tangent_active_factors=request.tangent_active_factors,
        tangent_block_width=request.tangent_block_width,
        tangent_anchor_eta=request.tangent_anchor_eta,
        tangent_fit_nsamples=request.tangent_fit_nsamples,
        tangent_fit_seed=request.tangent_fit_seed,
        tangent_exact_mode=request.tangent_exact_mode,
        tangent_exact_max_params=request.tangent_exact_max_params,
        pruner=request.pruner,
        bias_fold=request.bias_fold,
        gmp=False,
    )
    runners = {
        "qwen": qwen.run,
        "olmo": olmo.run,
        "llama": llama.run,
        "gemma": gemma.run,
    }
    runners[request.family](namespace)
