import math
import time

import torch
import torch.nn as nn

from .geometry import GeometrySparseGPT, add_geometry_arguments, apply_geometry
from .modelutils import *
from .prune_cache import (
    find_pruned_model,
    prune_config_from_args,
    resolve_save_dir,
    write_prune_config,
)
from .quant import *
from .sparsegpt import *

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# FLA's Triton gated-delta kernel and the causal-conv CUDA op are what the
# Qwen3.5 forward uses when those packages are installed. On this GB10 they
# can sit on a large allocation with the SMs idle. These names are the pure
# PyTorch bodies the wrappers fall back to.
_PYTORCH_ATTENTION = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def _unwrap_torch(fn):
    seen = set()
    while hasattr(fn, "__wrapped__") and id(fn.__wrapped__) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn


def force_pytorch_attention() -> None:
    """Replace FLA and causal-conv kernels with the PyTorch reference bodies."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    for name in _PYTORCH_ATTENTION:
        current = getattr(modeling, name)
        bare = _unwrap_torch(current)
        setattr(modeling, name, bare)
        print(f"attention backend: {name} -> pytorch", flush=True)


def _enable_gdn_kernel(model) -> None:
    """Swap Gated DeltaNet for Atlas-Inference/gdn without leaving the GPU.

    ``kernelize`` picks the hub repo from ``model.device``. The checkpoint is
    loaded on CPU and layers move to CUDA one at a time, so the first
    parameter is parked on CUDA only while the SM121 kernel is bound.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Atlas-Inference/gdn requires the GB10 CUDA device.")
    first = next(model.parameters())
    home = first.data
    first.data = home.to("cuda")
    try:
        model.set_use_kernels(True)
    finally:
        first.data = home


def get_qwen(model, *, use_kernels=True):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM

    # Qwen/Qwen3.5-2B is a multimodal repo; AutoModelForCausalLM loads the text tower.
    # Kernels are bound afterwards so the SM121 repo is visible to kernelize.
    model = AutoModelForCausalLM.from_pretrained(
        model, dtype="auto", trust_remote_code=True, use_kernels=False
    )
    if use_kernels:
        _enable_gdn_kernel(model)
    model.seqlen = 2048
    return model


def _true_sequential_groups(layer):
    """Module groups matching SparseGPT's LLaMA true-sequential schedule."""
    if getattr(layer, "block_type", None) == "linear_attention":
        return [
            [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
            ],
            ["linear_attn.out_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    return [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


def _run_layer(layer, hidden, *, attention_mask, position_embeddings, position_ids):
    # Qwen3.5 decoder layers return a tensor (not a tuple).
    return layer(
        hidden,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )


def forward_batch_size(nsamples: int, seqlen: int, intermediate: int, element_size: int) -> int:
    """How many independent windows to run through one decoder layer at once.

    Gate and up projections each keep a (seqlen, intermediate) activation.
    A 4 GiB budget is about 30 windows on Qwen3.6-27B in bf16, and the whole
    calibration set on the smaller models. Windows do not depend on each
    other, so the chunk size does not change the layer outputs.
    """
    if nsamples <= 1:
        return max(1, nsamples)
    pair = 2 * seqlen * intermediate * element_size
    if pair <= 0:
        return nsamples
    return max(1, min(nsamples, (4 * 1024**3) // pair))


def resolve_forward_batch(nsamples: int, seqlen: int, intermediate: int, element_size: int) -> int:
    requested = getattr(args, "forward_batch", 0)
    if requested:
        return max(1, min(nsamples, requested))
    return forward_batch_size(nsamples, seqlen, intermediate, element_size)


def _atlas_linear(layer) -> bool:
    """True when this block's mixer forward is the Atlas SM121 kernel.

    ``gdn_prefill`` computes batch row 0 correctly and mixes it into later
    rows. Identical rows hide the leak. Distinct windows must not share a
    real batch dimension.
    """
    attn = getattr(layer, "linear_attn", None)
    if attn is None:
        return False
    return "gdn" in getattr(attn.forward, "__module__", "")


def _run_layer_streams(
    layer,
    hidden,
    outs,
    *,
    attention_mask,
    position_embeddings,
    position_ids,
    width,
):
    """Run batch-1 forwards on separate CUDA streams.

    This is the batched path for Atlas linear layers. Each stream sees one
    window, which is the launch ``gdn_prefill`` gets right.
    """
    count = hidden.shape[0]
    step = max(1, min(width, count))
    current = torch.cuda.current_stream()
    for start in range(0, count, step):
        stop = min(start + step, count)
        streams = [torch.cuda.Stream() for _ in range(stop - start)]
        for stream in streams:
            stream.wait_stream(current)
        for offset, stream in enumerate(streams):
            row = start + offset
            with torch.cuda.stream(stream):
                outs[row : row + 1] = _run_layer(
                    layer,
                    hidden[row : row + 1],
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                )
        for stream in streams:
            current.wait_stream(stream)


def _run_layer_batched(
    layer,
    hidden,
    outs,
    *,
    attention_mask,
    position_embeddings,
    position_ids,
    batch,
    parallel=False,
):
    """Write layer(hidden) into outs.

    Full attention and the PyTorch delta rule take a real batch: the windows
    are independent and the batch-1 mask and RoPE broadcast. Atlas linear
    layers, when ``parallel`` is set, overlap batch-1 kernels on streams.
    The Hessian pass leaves ``parallel`` off so ``add_batch`` stays serial.
    """
    count = hidden.shape[0]
    step = max(1, min(batch, count))
    if parallel and step > 1 and _atlas_linear(layer):
        _run_layer_streams(
            layer,
            hidden,
            outs,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            width=step,
        )
        return
    if _atlas_linear(layer):
        step = 1
    for start in range(0, count, step):
        stop = min(start + step, count)
        outs[start:stop] = _run_layer(
            layer,
            hidden[start:stop],
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )


def _leading_example(value):
    """Keep the cache contract of a batch-1 catcher when the embed is batched.

    Text RoPE cos/sin are (batch, seq, dim). Multimodal position ids are
    (4, batch, seq) or (3, batch, seq) and the leading axis is not the batch.
    """
    if isinstance(value, tuple):
        return tuple(_leading_example(item) for item in value)
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] > 1:
        if value.ndim >= 3 and value.shape[0] in (3, 4) and value.shape[1] > 1:
            return value[:, :1]
        return value[:1]
    return value


def gdn_backend(model) -> str:
    """Module that owns the first Gated DeltaNet forward, after kernelize."""
    for layer in model.model.layers:
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            continue
        fwd = attn.forward
        module = getattr(fwd, "__module__", type(attn).__module__)
        name = getattr(fwd, "__qualname__", getattr(fwd, "__name__", type(fwd).__name__))
        return f"{module}.{name}"
    return "no linear attention"


@torch.no_grad()
def qwen_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_embeddings": None,
        "position_ids": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            n = inp.shape[0]
            inps[cache["i"] : cache["i"] + n] = inp
            cache["i"] += n
            # Later layers reuse one window's mask and RoPE. Every calibration
            # window has the same length, so the first row stands for the batch.
            cache["attention_mask"] = _leading_example(kwargs.get("attention_mask"))
            cache["position_embeddings"] = _leading_example(kwargs.get("position_embeddings"))
            cache["position_ids"] = _leading_example(kwargs.get("position_ids"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    tokens = torch.cat([batch[0] for batch in dataloader], dim=0)
    capture_batch = resolve_forward_batch(
        tokens.shape[0],
        model.seqlen,
        model.config.intermediate_size,
        torch.empty((), dtype=dtype).element_size(),
    )
    print(f"forward batch={capture_batch}", flush=True)
    for start in range(0, tokens.shape[0], capture_batch):
        stop = min(start + capture_batch, tokens.shape[0])
        try:
            model(tokens[start:stop].to(dev), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # Layer-0 mask is only valid for layer-0's mixer type; rebuild both kinds.
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    mask_kwargs = {
        "config": model.model.config,
        "inputs_embeds": inps[:1],
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": cache["position_ids"],
    }
    attention_masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }
    position_embeddings = cache["position_embeddings"]
    position_ids = cache["position_ids"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        layer_mask = attention_masks.get(getattr(layer, "block_type", "full_attention"))

        if args.true_sequential:
            sequential = _true_sequential_groups(layer)
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = (
                    GeometrySparseGPT if args.geometry_alpha > 0 else SparseGPT
                )(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            _run_layer_batched(
                layer,
                inps,
                outs,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                batch=capture_batch,
            )
            for h in handles:
                h.remove()

            for name in gpts:
                print(i, name)
                print("Pruning ...")
                apply_geometry(gpts[name], args, i, name)
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        _run_layer_batched(
            layer,
            inps,
            outs,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            batch=capture_batch,
            parallel=True,
        )

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


def eval_chunk_size(nsamples: int, seqlen: int, hidden: int, element_size: int) -> int:
    """How many windows to keep resident during layer-wise eval.

    Each window stores two hidden-state buffers. WikiText2 on Qwen3.5-27B is
    about 1.2k windows and those buffers are 24 GiB each, which does not fit
    beside the weights. Windows are independent, so a smaller chunk is the
    same perplexity. PTB and the smaller models stay in one pass.
    """
    if nsamples <= 0:
        return 1
    pair = 2 * seqlen * hidden * element_size
    if pair <= 0 or nsamples * pair <= 16 * 1024**3:
        return nsamples
    return max(1, (8 * 1024**3) // pair)


@torch.no_grad()
def qwen_eval(
    model,
    testenc,
    dev,
    dataset: str,
    log_wandb: bool = False,
    *,
    apply_gmp: bool | None = None,
):
    print("Evaluating ...")
    if apply_gmp is None:
        apply_gmp = args.gmp

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    dtype = next(iter(model.parameters())).dtype
    chunk = eval_chunk_size(
        nsamples,
        model.seqlen,
        model.config.hidden_size,
        torch.empty((), dtype=dtype).element_size(),
    )
    print(f"eval samples={nsamples} chunk={chunk}", flush=True)
    print("eval allocate", flush=True)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []
    for start in range(0, nsamples, chunk):
        stop = min(start + chunk, nsamples)
        if chunk < nsamples:
            print(f"eval chunk {start}:{stop}", flush=True)
        nlls.extend(
            _eval_windows(
                model,
                testenc,
                dev,
                start,
                stop,
                dtype,
                apply_gmp=apply_gmp and start == 0,
            )
        )
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})
    model.config.use_cache = use_cache


def _eval_windows(model, testenc, dev, start, stop, dtype, *, apply_gmp):
    """Layer-wise perplexity on the half-open window range [start, stop)."""
    layers = model.model.layers
    count = stop - start

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros(
        (count, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_embeddings": None,
        "position_ids": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            n = inp.shape[0]
            inps[cache["i"] : cache["i"] + n] = inp
            cache["i"] += n
            cache["attention_mask"] = _leading_example(kwargs.get("attention_mask"))
            cache["position_embeddings"] = _leading_example(kwargs.get("position_embeddings"))
            cache["position_ids"] = _leading_example(kwargs.get("position_ids"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    print("eval capture", flush=True)
    windows = torch.cat(
        [
            testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]
            for i in range(start, stop)
        ],
        dim=0,
    )
    capture_batch = resolve_forward_batch(
        count,
        model.seqlen,
        model.config.intermediate_size,
        torch.empty((), dtype=dtype).element_size(),
    )
    for offset in range(0, count, capture_batch):
        batch = windows[offset : offset + capture_batch].to(dev)
        try:
            model(batch, use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    mask_kwargs = {
        "config": model.model.config,
        "inputs_embeds": inps[:1],
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": cache["position_ids"],
    }
    attention_masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }
    position_embeddings = cache["position_embeddings"]
    position_ids = cache["position_ids"]
    print("eval layers", flush=True)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        layer_mask = attention_masks.get(getattr(layer, "block_type", "full_attention"))

        if apply_gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        _run_layer_batched(
            layer,
            inps,
            outs,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            batch=resolve_forward_batch(
                count,
                model.seqlen,
                model.config.intermediate_size,
                torch.empty((), dtype=dtype).element_size(),
            ),
            parallel=True,
        )
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for offset, i in enumerate(range(start, stop)):
        hidden_states = inps[offset].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to(dev)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * model.seqlen)
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return nlls


if __name__ == "__main__":
    import argparse
    from .datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="Qwen model to load; pass e.g. `Qwen/Qwen3.5-2B`.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--eval-datasets",
        type=str,
        nargs="+",
        default=["wikitext2", "ptb", "c4"],
        choices=["wikitext2", "ptb", "c4"],
        help="Datasets to evaluate after pruning.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip perplexity evaluation (prune / save only).",
    )
    parser.add_argument(
        "--load-cached",
        action="store_true",
        help="Evaluate a matching pruned checkpoint. Refuse to prune if none exists.",
    )
    parser.add_argument(
        "--pytorch-attention",
        action="store_true",
        help="Run Qwen3.5 linear attention and causal conv as pure PyTorch.",
    )
    parser.add_argument(
        "--forward-batch",
        type=int,
        default=0,
        help="Windows per decoder-layer forward. 0 picks a batch that fits in 4 GiB of MLP activations.",
    )

    add_geometry_arguments(parser)
    args = parser.parse_args()
    if not math.isfinite(args.geometry_alpha) or args.geometry_alpha < 0:
        parser.error("--geometry-alpha must be finite and nonnegative")
    if args.geometry_alpha > 0 and args.geometry_probes is None:
        parser.error("--geometry-probes is required with positive --geometry-alpha")
    if args.geometry_alpha > 0 and args.gmp:
        parser.error("geometry calibration requires SparseGPT, not --gmp")

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    if args.pytorch_attention:
        force_pytorch_attention()

    pruning = bool(args.sparsity or args.prunen)
    config = prune_config_from_args(args)
    cached = find_pruned_model(config, args.save) if pruning else None
    if args.load_cached and pruning and cached is None:
        raise SystemExit(
            "No pruned checkpoint matches this config; refusing to prune."
        )

    use_kernels = not args.pytorch_attention
    if cached is not None:
        print(f"Loading pruned model from {cached} (matches pruning config).")
        model = get_qwen(str(cached), use_kernels=use_kernels)
    else:
        if pruning:
            print(
                "No pruned model in sparse_opt matches this config; "
                "pruning from the base model."
            )
        model = get_qwen(args.model, use_kernels=use_kernels)
    print(f"gdn backend: {gdn_backend(model)}", flush=True)
    model.eval()

    ran_sparsegpt = False
    if cached is None and pruning and not args.gmp:
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        tick = time.time()
        qwen_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()).item())
            if "down_proj" in n:
                break
        print(time.time() - tick)
        ran_sparsegpt = True

    if not args.skip_eval:
        for dataset in args.eval_datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            # A loaded checkpoint is already pruned. Re-applying magnitude
            # pruning would sparsify it a second time.
            qwen_eval(
                model,
                testloader,
                DEV,
                dataset,
                args.log_wandb,
                apply_gmp=args.gmp and cached is None,
            )

    ran_gmp = cached is None and args.gmp and pruning and not args.skip_eval
    if args.save and cached is None and (ran_sparsegpt or ran_gmp):
        dest = resolve_save_dir(args.save, config)
        print(f"Saving pruned model to {dest}")
        model.save_pretrained(dest)
        write_prune_config(dest, config)
    elif args.save and cached is None:
        model.save_pretrained(args.save)
