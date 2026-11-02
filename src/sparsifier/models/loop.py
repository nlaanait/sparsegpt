"""Shared pieces of the Qwen, OLMo, Llama, and Gemma layer loops.

Each family still owns its block forward. This module is the catcher, the
Hessian hook, and the load / prune / eval / save tail those loops repeat.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from sparsifier.data.cache import (
    find_pruned_model,
    prune_config_from_args,
    resolve_save_dir,
    write_prune_config,
)
from sparsifier.metrics import Likelihood, Perplexity
from sparsifier.models import log_perplexity
from sparsifier.models.layers import DEV
from sparsifier.prune import get_pruner


def leading_example(value):
    """One row of a batched mask or RoPE cache.

    Text RoPE cos/sin are floating ``(batch, seq, dim)``. Multimodal position
    ids are integer ``(4, batch, seq)`` or ``(3, batch, seq)``; their leading
    axis is not the batch. A float RoPE tensor whose first axis is 3 or 4
    stays on the batch-first path.
    """
    if isinstance(value, tuple):
        return tuple(leading_example(item) for item in value)
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] > 1:
        if (
            value.ndim >= 3
            and value.shape[0] in (3, 4)
            and value.shape[1] > 1
            and value.dtype in (torch.int32, torch.int64, torch.long)
        ):
            return value[:, :1]
        return value[:1]
    return value


def capture_cache(*, positions: bool = True) -> dict:
    """State the first-layer catcher fills."""
    cache = {"i": 0, "attention_mask": None}
    if positions:
        cache["position_embeddings"] = None
        cache["position_ids"] = None
    return cache


class Catcher(nn.Module):
    """Replace a layer, store its input, then abort the forward.

    ``batched`` writes ``inp.shape[0]`` rows, which is how Qwen captures a
    packed calibration batch. Otherwise one row is stored. ``squeeze`` keeps
    ``inp[0]`` when the hidden state is ``(batch, seq, hidden)``. ``fields``
    are copied from the forward kwargs. ``leading`` reduces each of those to
    one example. ``required`` fields must be present.
    """

    def __init__(
        self,
        module,
        inps,
        cache,
        *,
        batched: bool = False,
        squeeze: bool = False,
        fields: tuple[str, ...] = ("attention_mask", "position_embeddings", "position_ids"),
        leading: bool = True,
        required: tuple[str, ...] = (),
    ):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache
        self.batched = batched
        self.squeeze = squeeze
        self.fields = fields
        self.leading = leading
        self.required = required

    def forward(self, inp, **kwargs):
        if self.batched:
            n = inp.shape[0]
            self.inps[self.cache["i"] : self.cache["i"] + n] = inp
            self.cache["i"] += n
        else:
            stored = inp[0] if self.squeeze and inp.ndim == 3 else inp
            self.inps[self.cache["i"]] = stored
            self.cache["i"] += 1
        for field in self.fields:
            value = kwargs[field] if field in self.required else kwargs.get(field)
            self.cache[field] = leading_example(value) if self.leading else value
        raise ValueError


def decoder_groups() -> list[list[str]]:
    """Attention then MLP, the true-sequential order for a standard decoder block."""
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def eval_chunk_size(nsamples: int, seqlen: int, hidden: int, element_size: int) -> int:
    """How many windows to keep resident during layer-wise eval.

    Each window stores two hidden-state buffers. Windows are independent, so
    a smaller chunk is the same perplexity.
    """
    if nsamples <= 0:
        return 1
    pair = 2 * seqlen * hidden * element_size
    if pair <= 0 or nsamples * pair <= 16 * 1024**3:
        return nsamples
    return max(1, (8 * 1024**3) // pair)


def excluded(args, index: int, name: str) -> bool:
    """Whether this linear is outside the requested layer and name filter."""
    outside = not (args.minlayer <= index < args.maxlayer and args.prune_only in name)
    return outside == (not args.invert)


def pruner_for(args):
    """The pruner named by the run. Geometry carries its alpha and probes."""
    if args.geometry_alpha > 0:
        return get_pruner("geometry", alpha=args.geometry_alpha, probes=args.geometry_probes)
    return get_pruner("sparsegpt")


def bind_selected(subset, args, index: int, pruner=None):
    """Bind a pruner to each linear the filter keeps."""
    if pruner is None:
        pruner = pruner_for(args)
    bound = {}
    for name, module in subset.items():
        if excluded(args, index, name):
            continue
        bound[name] = pruner.bind(module, block_index=index, module_name=name)
    return bound


def hessian_hooks(bound, modules):
    """Forward hooks that fold each linear's activations into its pruner."""

    def add_batch(name):
        def tmp(_, inp, out):
            bound[name].add_batch(inp[0].data, out.data)

        return tmp

    return [modules[name].register_forward_hook(add_batch(name)) for name in bound]


def prune_bound(bound, args, index: int, *, flush: bool = False) -> None:
    """Prune every bound linear and drop its Hessian."""
    for name in bound:
        print(index, name, flush=flush)
        print("Pruning ...", flush=flush)
        bound[name].prune(
            sparsity=args.sparsity,
            zeros_per_group=args.zeros_per_group,
            group_size=args.group_size,
            percdamp=args.percdamp,
            blocksize=args.blocksize,
        )
        bound[name].free()


def likelihood_from_nlls(nlls, nsamples: int, seqlen: int) -> Likelihood:
    """The summed window losses the eval loops already compute."""
    nats = float(torch.stack(nlls).sum())
    return Likelihood(nats=nats, tokens=nsamples * seqlen)


def publish_likelihood(likelihood: Likelihood, dataset: str, log_wandb: bool) -> float:
    """Print perplexity from a likelihood and, when asked, send it to wandb."""
    value = Perplexity().score(likelihood).value
    print(f"Perplexity: {value:3f}")
    log_perplexity(dataset, value, log_wandb)
    return value


def report_perplexity(nlls, nsamples: int, seqlen: int, dataset: str, log_wandb: bool):
    """Print perplexity and, when asked, send it to wandb."""
    return publish_likelihood(likelihood_from_nlls(nlls, nsamples, seqlen), dataset, log_wandb)


def check_geometry(args) -> None:
    if not math.isfinite(args.geometry_alpha) or args.geometry_alpha < 0:
        raise ValueError("geometry alpha must be finite and nonnegative")
    if args.geometry_alpha > 0 and args.geometry_probes is None:
        raise ValueError("geometry probes are required when geometry alpha is positive")


def open_checkpoint(args):
    """Return ``(config, cached, pruning)``. A missing cache under ``--load-cached`` aborts."""
    pruning = bool(args.sparsity or args.zeros_per_group)
    config = prune_config_from_args(args)
    cached = find_pruned_model(config, args.save) if pruning else None
    if args.load_cached and pruning and cached is None:
        raise SystemExit("No pruned checkpoint matches this config; refusing to prune.")
    return config, cached, pruning


def load_model(args, cached, pruning: bool, loader):
    """Load a matching checkpoint, or the base model ``loader`` knows."""
    if cached is not None:
        print(f"Loading pruned model from {cached} (matches pruning config).")
        return loader(str(cached))
    if pruning:
        print("No pruned model in sparse_opt matches this config; pruning from the base model.")
    return loader(args.model)


def prune_decoder(family, model, batches, dev, args) -> None:
    """Walk decoder blocks. The family captures, forwards, and names linears."""
    print("Starting...")
    captured = family.capture(model, batches, args)
    hidden = captured["hidden"]
    print("Ready.")
    layers = model.model.layers
    for index in range(len(layers)):
        layer = layers[index].to(dev)
        modules = family.prunable(layer)
        if args.true_sequential:
            groups = family.sequential_groups(layer)
        else:
            groups = [list(modules)]
        for names in groups:
            subset = {name: modules[name] for name in names if name in modules}
            bound = bind_selected(subset, args, index)
            if not bound:
                continue
            handles = hessian_hooks(bound, subset)
            captured["propagate"] = False
            family.forward_layer(layer, hidden, captured)
            for handle in handles:
                handle.remove()
            prune_bound(bound, args, index, flush=bool(captured.get("flush")))
        captured["propagate"] = True
        previous = hidden
        hidden = family.forward_layer(layer, hidden, captured)
        captured["outs"] = previous
        captured["hidden"] = hidden
        layers[index] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    model.config.use_cache = captured["use_cache"]


def run_model(family, args, report) -> None:
    """Load, prune, score, and save through the family protocol."""
    from sparsifier.models import init_wandb

    check_geometry(args)
    init_wandb(args)
    config, cached, pruning = open_checkpoint(args)
    model = load_model(args, cached, pruning, family.load)
    if hasattr(family, "prepare"):
        family.prepare(model)
    else:
        model.eval()
    ran = False
    if cached is None and pruning:

        def sequential(model, dataloader, dev):
            if hasattr(family, "prune_model"):
                family.prune_model(model, dataloader, args, pruner_for(args))
            else:
                prune_decoder(family, model, dataloader, dev, args)

        run_prune(args, model, sequential, report)
        ran = True

    def score(model, testloader, _dev, dataset, log_wandb):
        _ = (dataset, log_wandb)
        return family.likelihood(model, testloader, args)

    each_eval(args, model, score)
    save_pruned(model, args, config, cached, ran)


def run_prune(args, model, sequential, report) -> None:
    """Calibrate, prune, then let ``report`` print the sparsities."""
    from sparsifier.data.loaders import get_loaders

    dataloader, _testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    tick = time.time()
    sequential(model, dataloader, DEV)
    report(model)
    print(time.time() - tick)


def each_eval(args, model, evaluate) -> None:
    """Score every requested dataset, unless eval was skipped."""
    if args.skip_eval:
        return
    from sparsifier.data.loaders import get_loaders

    for dataset in args.eval_datasets:
        _dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        result = evaluate(model, testloader, DEV, dataset, args.log_wandb)
        if isinstance(result, Likelihood):
            publish_likelihood(result, dataset, args.log_wandb)


def save_pruned(model, args, config, cached, ran: bool) -> None:
    if args.save and cached is None and ran:
        dest = resolve_save_dir(args.save, config)
        print(f"Saving pruned model to {dest}")
        model.save_pretrained(dest)
        write_prune_config(dest, config)
