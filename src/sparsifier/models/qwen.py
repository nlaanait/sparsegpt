import torch
import torch.nn as nn

from sparsifier.metrics import Likelihood
from sparsifier.models.layers import DEV, find_layers
from sparsifier.models.loop import (
    Catcher,
    capture_cache,
    eval_chunk_size,
    likelihood_from_nlls,
    run_model,
)

args = None

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
def _capture_qwen(model, dataloader, dev):
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
    cache = capture_cache()

    # Later layers reuse one window's mask and RoPE. Every calibration window
    # has the same length, so the first row stands for the batch.
    layers[0] = Catcher(layers[0], inps, cache, batched=True)
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

    # Layer-0 mask is only valid for layer-0's mixer type; rebuild both kinds.
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    mask_kwargs = {
        "config": model.model.config,
        "inputs_embeds": inps[:1],
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": cache["position_ids"],
    }
    return {
        "hidden": inps,
        "outs": torch.zeros_like(inps),
        "use_cache": use_cache,
        "propagate": False,
        "batch": capture_batch,
        "attention_masks": {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        },
        "position_embeddings": cache["position_embeddings"],
        "position_ids": cache["position_ids"],
    }


def _forward_qwen(layer, hidden, captured):
    mask = captured["attention_masks"].get(getattr(layer, "block_type", "full_attention"))
    _run_layer_batched(
        layer,
        hidden,
        captured["outs"],
        attention_mask=mask,
        position_embeddings=captured["position_embeddings"],
        position_ids=captured["position_ids"],
        batch=captured["batch"],
        parallel=bool(captured.get("propagate")),
    )
    return captured["outs"]


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
    model.config.use_cache = use_cache
    _ = (dataset, log_wandb)
    return likelihood_from_nlls(nlls, nsamples, model.seqlen)


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
    cache = capture_cache()
    layers[0] = Catcher(layers[0], inps, cache, batched=True)
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


def _report_sparsity(model) -> None:
    for name, param in model.named_parameters():
        print(name, torch.mean((param == 0).float()).item())
        if "down_proj" in name:
            break


class Qwen:
    """Decoder family for Qwen text towers."""

    name = "qwen"

    def matches(self, model_id: str) -> bool:
        return model_id.startswith("Qwen/")

    def load(self, model_id: str):
        if args.pytorch_attention:
            force_pytorch_attention()
        return get_qwen(model_id, use_kernels=not args.pytorch_attention)

    def prepare(self, model) -> None:
        print(f"gdn backend: {gdn_backend(model)}", flush=True)
        model.eval()

    def capture(self, model, batches, config):
        _ = config
        return _capture_qwen(model, batches, DEV)

    def forward_layer(self, layer, hidden, captured):
        return _forward_qwen(layer, hidden, captured)

    def prunable(self, layer) -> dict:
        return find_layers(layer)

    def sequential_groups(self, layer) -> list[list[str]]:
        return _true_sequential_groups(layer)

    def likelihood(self, model, encoded, config) -> Likelihood:
        _ = config
        return qwen_eval(model, encoded, DEV, "", False, apply_gmp=False)


QWEN = Qwen()


def run(cli_args) -> None:
    """Load, prune, evaluate, and optionally save one Qwen checkpoint."""
    global args
    args = cli_args
    run_model(QWEN, args, _report_sparsity)
