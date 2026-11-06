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
from sparsifier.prune import get_pruner, list_pruners

OUTPUT_TANGENT_PRUNERS = frozenset(
    {
        "tangent",
        "tangent_factor_union",
        "tangent_chart_obs",
        "tangent_fixed_union",
        "tangent_knockout",
        "tangent_editability",
        "tangent_exact_obs",
    }
)

# Pruners that need paired outputs but are not output-tangent atlas techniques.
OUTPUT_AWARE_PRUNERS = OUTPUT_TANGENT_PRUNERS | frozenset({"wanda"})


def calibration_window_count(args) -> int:
    """Total captured windows: prune-cal plus optional tangent-fit split."""
    fit = int(getattr(args, "tangent_fit_nsamples", 0) or 0)
    if fit < 0:
        raise ValueError("tangent_fit_nsamples must be nonnegative")
    return int(args.nsamples) + fit


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


def write_bank(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy ``src`` into a preallocated activation bank without replacing storage.

    Assigning ``bank[i] = x.to("cpu")`` (or a fresh GPU tensor) into a slot of
    a preallocated bank can retain the source on some torch builds
    (GH200 / 2.10+cu128) and OOM mid SparseGPT. ``copy_`` keeps the bank's
    storage and drops the temporary.
    """
    value = src.detach()
    if value.shape != dst.shape and value.ndim == dst.ndim + 1 and value.shape[0] == 1:
        value = value[0]
    dst.copy_(value)


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
            write_bank(self.inps[self.cache["i"] : self.cache["i"] + n], inp)
            self.cache["i"] += n
        else:
            stored = inp[0] if self.squeeze and inp.ndim == 3 else inp
            write_bank(self.inps[self.cache["i"]], stored)
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


def resolve_pruner_name(args) -> str:
    """Ablation registry name for this run (explicit ``--pruner`` wins)."""
    explicit = getattr(args, "pruner", None) or ""
    if explicit:
        if explicit not in list_pruners():
            raise ValueError(f"unknown pruner {explicit!r}; known: {', '.join(list_pruners())}")
        return explicit
    if float(getattr(args, "tangent_alpha", 0.0) or 0.0) > 0:
        return "tangent"
    if float(getattr(args, "geometry_alpha", 0.0) or 0.0) > 0:
        return "geometry"
    return "sparsegpt"


def uses_output_tangents(args) -> bool:
    """Whether the selected technique needs paired layer outputs."""
    return resolve_pruner_name(args) in OUTPUT_TANGENT_PRUNERS


def uses_paired_outputs(args) -> bool:
    """Whether hooks must capture layer outputs (tangent atlas or Wanda)."""
    return resolve_pruner_name(args) in OUTPUT_AWARE_PRUNERS


def uses_separate_tangent_fit(args) -> bool:
    """Whether tangent fitting uses a disjoint window prefix."""
    return int(getattr(args, "tangent_fit_nsamples", 0) or 0) > 0 and uses_output_tangents(
        args
    )


def pruner_for(args):
    """Select ambient, geometry, or an output-tangent ablation technique."""
    name = resolve_pruner_name(args)
    bias_fold = bool(getattr(args, "bias_fold", False))
    charts = int(getattr(args, "tangent_charts", 8))
    rank = int(getattr(args, "tangent_rank", 2))
    samples = int(getattr(args, "tangent_samples", 512))
    seed = int(getattr(args, "seed", 0))
    alpha = float(getattr(args, "tangent_alpha", 0.0) or 0.0)
    anchor_eta = float(getattr(args, "tangent_anchor_eta", 0.0) or 0.0)
    fit_seed = getattr(args, "tangent_fit_seed", None)
    tangent_seed = int(seed if fit_seed is None else fit_seed)
    if name == "sparsegpt":
        return get_pruner("sparsegpt")
    if name == "wanda":
        return get_pruner(
            "wanda",
            calibration_samples=samples,
            seed=seed,
        )
    if name == "geometry":
        return get_pruner(
            "geometry",
            alpha=float(args.geometry_alpha),
            probes=args.geometry_probes,
        )
    if name == "tangent":
        return get_pruner(
            "tangent",
            alpha=alpha if alpha > 0 else 1.0,
            charts=charts,
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            bias_fold=bias_fold,
            anchor_eta=anchor_eta,
        )
    if name == "tangent_factor_union":
        return get_pruner(
            "tangent_factor_union",
            alpha=alpha if alpha > 0 else 1.0,
            factors=int(getattr(args, "tangent_factors", charts)),
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            bias_fold=bias_fold,
            active_factors=int(getattr(args, "tangent_active_factors", 2)),
            factor_source=str(getattr(args, "tangent_factor_source", "kmeans")),
            block_width=int(getattr(args, "tangent_block_width", 3)),
        )
    if name == "tangent_chart_obs":
        return get_pruner(
            "tangent_chart_obs",
            charts=charts,
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            percdamp=float(getattr(args, "percdamp", 0.01)),
            bias_fold=bias_fold,
        )
    if name == "tangent_fixed_union":
        return get_pruner(
            "tangent_fixed_union",
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            bias_fold=bias_fold,
        )
    if name == "tangent_knockout":
        return get_pruner(
            "tangent_knockout",
            charts=charts,
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            bias_fold=bias_fold,
        )
    if name == "tangent_editability":
        return get_pruner(
            "tangent_editability",
            charts=charts,
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            bias_fold=bias_fold,
        )
    if name == "tangent_exact_obs":
        return get_pruner(
            "tangent_exact_obs",
            charts=charts,
            rank=rank,
            calibration_samples=samples,
            seed=tangent_seed,
            percdamp=float(getattr(args, "percdamp", 0.01)),
            bias_fold=bias_fold,
            anchor_eta=anchor_eta,
            mode=str(getattr(args, "tangent_exact_mode", "diag")),
            max_full_params=int(getattr(args, "tangent_exact_max_params", 4096)),
        )
    raise ValueError(f"unhandled pruner {name!r}")


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


def hessian_hooks(bound, modules, *, mode: str = "both"):
    """Forward hooks that fold each linear's activations into its pruner.

    ``mode`` is ``both`` (default), ``tangent`` (reservoir / chart fit only),
    or ``hessian`` (ambient SparseGPT Hessian only). Split calibration uses
    the latter two sequentially when ``--tangent-fit-nsamples > 0``.
    """
    if mode not in {"both", "tangent", "hessian"}:
        raise ValueError("hook mode must be both, tangent, or hessian")

    def add_batch(name):
        def tmp(_, inp, out):
            wrapper = bound[name]
            if mode == "both":
                wrapper.add_batch(inp[0].data, out.data)
                return
            inner = getattr(wrapper, "solver", None)
            if mode == "tangent":
                if inner is not None and hasattr(inner, "add_tangent_batch"):
                    inner.add_tangent_batch(inp[0].data, out.data)
                else:
                    wrapper.add_batch(inp[0].data, out.data)
                return
            # hessian-only
            if inner is not None and hasattr(inner, "add_hessian_batch"):
                inner.add_hessian_batch(inp[0].data, out.data)
            else:
                from sparsifier.prune.sparsegpt import SparseGPT as _SG

                if inner is not None and isinstance(inner, _SG):
                    _SG.add_batch(inner, inp[0].data, out.data)
                else:
                    wrapper.add_batch(inp[0].data, out.data)

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
    print(f"Likelihood nats: {likelihood.nats:.6f}")
    print(f"Tokens: {likelihood.tokens}")
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
    if not math.isfinite(args.tangent_alpha) or not 0.0 <= args.tangent_alpha <= 1.0:
        raise ValueError("tangent alpha must be finite and in [0, 1]")
    name = resolve_pruner_name(args)
    if name in OUTPUT_TANGENT_PRUNERS and args.geometry_alpha > 0:
        raise ValueError("input geometry and output tangent modes are mutually exclusive")
    if name in OUTPUT_TANGENT_PRUNERS:
        if args.tangent_charts < 1 or args.tangent_rank < 1 or args.tangent_samples < 3:
            raise ValueError("tangent charts/rank must be positive and samples >= 3")
    if name == "tangent_factor_union":
        source = str(getattr(args, "tangent_factor_source", "kmeans"))
        if source not in {"kmeans", "block_dictionary"}:
            raise ValueError("tangent_factor_source must be kmeans or block_dictionary")
        factors = int(getattr(args, "tangent_factors", args.tangent_charts))
        active = int(getattr(args, "tangent_active_factors", 2))
        if factors < 1 or not 1 <= active <= factors:
            raise ValueError("tangent_factors/active_factors must satisfy 1 <= active <= factors")
        if source == "block_dictionary":
            width = int(getattr(args, "tangent_block_width", 3))
            if width < 1 or args.tangent_rank > width:
                raise ValueError("block_dictionary requires positive width and rank <= width")
    if name == "geometry" and float(args.geometry_alpha) <= 0:
        raise ValueError("geometry pruner requires positive geometry_alpha")
    anchor_eta = float(getattr(args, "tangent_anchor_eta", 0.0) or 0.0)
    if not math.isfinite(anchor_eta) or anchor_eta < 0.0:
        raise ValueError("tangent_anchor_eta must be finite and nonnegative")
    fit_n = int(getattr(args, "tangent_fit_nsamples", 0) or 0)
    if fit_n < 0:
        raise ValueError("tangent_fit_nsamples must be nonnegative")
    if fit_n > 0 and name not in OUTPUT_TANGENT_PRUNERS:
        raise ValueError("tangent_fit_nsamples requires an output-tangent pruner")
    if name == "tangent_exact_obs":
        mode = str(getattr(args, "tangent_exact_mode", "diag"))
        if mode not in {"diag", "full", "auto"}:
            raise ValueError("tangent_exact_mode must be diag, full, or auto")


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
    fit_n = int(getattr(args, "tangent_fit_nsamples", 0) or 0)
    split_fit = uses_separate_tangent_fit(args)
    if split_fit:
        if hidden.shape[0] < fit_n + int(args.nsamples):
            raise ValueError(
                "captured windows shorter than tangent_fit_nsamples + nsamples"
            )
        print(
            f"Calibration split: tangent-fit={fit_n} windows, "
            f"prune-cal={args.nsamples} windows",
            flush=True,
        )
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
            captured["propagate"] = False
            if split_fit:
                fit_hidden = hidden[:fit_n]
                cal_hidden = hidden[fit_n : fit_n + int(args.nsamples)]
                fit_caps = dict(captured)
                fit_caps["outs"] = captured["outs"][:fit_n]
                handles = hessian_hooks(bound, subset, mode="tangent")
                family.forward_layer(layer, fit_hidden, fit_caps)
                for handle in handles:
                    handle.remove()
                # Freeze tangent estimates before Hessian accumulation.
                for wrapper in bound.values():
                    wrapper.finalize()
                cal_caps = dict(captured)
                cal_caps["outs"] = captured["outs"][fit_n : fit_n + int(args.nsamples)]
                handles = hessian_hooks(bound, subset, mode="hessian")
                family.forward_layer(layer, cal_hidden, cal_caps)
                for handle in handles:
                    handle.remove()
            else:
                handles = hessian_hooks(bound, subset, mode="both")
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

    total = calibration_window_count(args)
    fit_n = int(getattr(args, "tangent_fit_nsamples", 0) or 0)
    fit_seed = getattr(args, "tangent_fit_seed", None)
    if fit_n > 0 and uses_output_tangents(args):
        # Disjoint windows: draw fit_n with tangent_fit_seed, then nsamples with seed.
        fit_seed = int(args.seed + 1 if fit_seed is None else fit_seed)
        fit_loader, _ = get_loaders(
            args.dataset,
            nsamples=fit_n,
            seed=fit_seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        cal_loader, _ = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        dataloader = list(fit_loader) + list(cal_loader)
        print(
            f"Loaded split calibration: fit_seed={fit_seed} fit_n={fit_n}, "
            f"cal_seed={args.seed} cal_n={args.nsamples}",
            flush=True,
        )
    else:
        dataloader, _testloader = get_loaders(
            args.dataset,
            nsamples=total,
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
