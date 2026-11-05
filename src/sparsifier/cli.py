"""Click entry point for one prune and its evals.

``python -m sparsifier`` and the ``sparsifier`` console script both call
``main``. N:M sparsity is ``--zeros-per-group`` and ``--group-size``.
"""

from __future__ import annotations

from pathlib import Path

import click

from sparsifier.models.catalog import resolve
from sparsifier.prune.quantize import apply_quantization


@click.command()
@click.argument("model")
@click.argument("calibration", type=click.Choice(["wikitext2", "ptb", "c4"]))
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--nsamples", type=int, default=128, show_default=True)
@click.option("--percdamp", type=float, default=0.01, show_default=True)
@click.option("--sparsity", type=float, default=0.0, show_default=True)
@click.option("--zeros-per-group", type=int, default=0, show_default=True)
@click.option("--group-size", type=int, default=0, show_default=True)
@click.option("--blocksize", type=int, default=128, show_default=True)
@click.option("--wbits", type=int, default=16, show_default=True)
@click.option("--minlayer", type=int, default=-1, show_default=True)
@click.option("--maxlayer", type=int, default=1000, show_default=True)
@click.option("--prune-only", default="")
@click.option("--invert", is_flag=True, default=False)
@click.option("--save", default="")
@click.option("--true-sequential", is_flag=True, default=False)
@click.option("--log-wandb", is_flag=True, default=False)
@click.option(
    "--eval-datasets",
    multiple=True,
    type=click.Choice(["wikitext2", "ptb", "c4"]),
    default=("wikitext2", "ptb", "c4"),
    show_default=True,
)
@click.option("--skip-eval", is_flag=True, default=False)
@click.option("--load-cached", is_flag=True, default=False)
@click.option("--pytorch-attention", is_flag=True, default=False)
@click.option("--forward-batch", type=int, default=0, show_default=True)
@click.option("--geometry-alpha", type=float, default=0.0, show_default=True)
@click.option("--geometry-probes", type=click.Path(path_type=Path), default=None)
@click.option(
    "--tangent-alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Enable output-tangent mask weighting; 0 keeps ambient SparseGPT.",
)
@click.option("--tangent-charts", type=int, default=8, show_default=True, help="Local output charts.")
@click.option("--tangent-rank", type=int, default=2, show_default=True, help="PCA rank per chart.")
@click.option(
    "--tangent-samples",
    type=int,
    default=512,
    show_default=True,
    help="Paired input/output tokens retained per layer.",
)
def main(
    model: str,
    calibration: str,
    seed: int,
    nsamples: int,
    percdamp: float,
    sparsity: float,
    zeros_per_group: int,
    group_size: int,
    blocksize: int,
    wbits: int,
    minlayer: int,
    maxlayer: int,
    prune_only: str,
    invert: bool,
    save: str,
    true_sequential: bool,
    log_wandb: bool,
    eval_datasets: tuple[str, ...],
    skip_eval: bool,
    load_cached: bool,
    pytorch_attention: bool,
    forward_batch: int,
    geometry_alpha: float,
    geometry_probes: Path | None,
    tangent_alpha: float,
    tangent_charts: int,
    tangent_rank: int,
    tangent_samples: int,
) -> None:
    """Prune one catalog model and, unless skipped, score it."""
    import torch

    if wbits < 16:
        apply_quantization(torch.empty(1), wbits)
    record = resolve(model)
    if record.family not in {"qwen", "olmo", "llama", "gemma"}:
        raise click.ClickException(f"{model} is not a measurement family ({record.family})")
    from sparsifier.models.runner import RunRequest, run_family

    run_family(
        RunRequest(
            model=model,
            calibration=calibration,
            seed=seed,
            nsamples=nsamples,
            percdamp=percdamp,
            sparsity=sparsity,
            zeros_per_group=zeros_per_group,
            group_size=group_size,
            blocksize=blocksize,
            wbits=wbits,
            minlayer=minlayer,
            maxlayer=maxlayer,
            prune_only=prune_only,
            invert=invert,
            save=save,
            true_sequential=true_sequential,
            log_wandb=log_wandb,
            eval_datasets=eval_datasets,
            skip_eval=skip_eval,
            load_cached=load_cached,
            pytorch_attention=pytorch_attention,
            forward_batch=forward_batch,
            geometry_alpha=geometry_alpha,
            geometry_probes=geometry_probes,
            tangent_alpha=tangent_alpha,
            tangent_charts=tangent_charts,
            tangent_rank=tangent_rank,
            tangent_samples=tangent_samples,
            family=record.family,
        )
    )
