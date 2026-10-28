"""Unstructured SparseGPT for Gemma 4 text.

Perplexity is the text tower only: Gemma4ForCausalLM, input ids, SDPA, and
no cache. Vision and audio weights are never loaded. One decoder layer is on
the GPU at a time. Per-layer embeddings, sliding versus full masks, K=V on
global layers, and shared KV on the tail follow Gemma4TextModel.forward.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

# The first CUDA allocation otherwise tries to reserve a large pool. On this
# unified-memory machine that reservation fails once the 31B weights are loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from .geometry import GeometrySparseGPT, add_geometry_arguments, apply_geometry
from .modelutils import DEV, find_layers
from .prune_cache import (
    find_pruned_model,
    prune_config_from_args,
    resolve_save_dir,
    write_prune_config,
)
from .quant import Quantizer
from .qwen import eval_chunk_size
from .sparsegpt import SparseGPT

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

# Decoder linears SparseGPT is allowed to touch. PLE projections, embeddings,
# norms, routers, and the packed per-layer table stay dense. MoE expert
# weights are packed 3D Parameters (not nn.Linear); they are pruned separately.
PRUNE_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

args = None


def is_prunable_parameter(name: str) -> bool:
    if any(name.endswith(f"{leaf}.weight") or name.endswith(f"{leaf}.bias") for leaf in PRUNE_NAMES):
        return True
    return name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj")


def _skip_init() -> dict:
    def skip(*_args, **_kwargs):
        pass

    saved = {}
    for name in ("kaiming_uniform_", "uniform_", "normal_"):
        saved[name] = getattr(torch.nn.init, name)
        setattr(torch.nn.init, name, skip)
    return saved


def _restore_init(saved: dict) -> None:
    for name, fn in saved.items():
        setattr(torch.nn.init, name, fn)


def _text_config(config):
    return config.text_config if getattr(config, "text_config", None) is not None else config


def _remap_key(key: str) -> str | None:
    """Map a checkpoint key onto Gemma4ForCausalLM. Vision and audio keys are dropped."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return "model." + key[len("language_model.") :]
    if key == "lm_head.weight" or key.startswith("model."):
        if ".vision_" in key or ".audio_" in key or key.startswith("model.embed_vision") or key.startswith("model.embed_audio"):
            return None
        return key
    return None


def _weight_files(source: str) -> list[Path]:
    root = Path(source)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(source))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [root / name for name in names]
    single = root / "model.safetensors"
    if single.is_file():
        return [single]
    found = sorted(root.rglob("model.safetensors"))
    if not found:
        raise FileNotFoundError(f"no safetensors weights under {source}")
    return found


def _drop_file_cache(path: Path) -> None:
    """Drop a shard from the page cache once its bytes have been copied.

    The 31B text tower is about 62 GiB. Leaving the safetensors file cached
    beside that allocation exhausts the unified memory on the next CUDA copy.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def _prime_cuda() -> None:
    """Create the CUDA context while the weight file is not also resident."""
    if torch.cuda.is_initialized():
        return
    torch.cuda.init()
    torch.empty(1, device="cuda")


def get_gemma(source: str):
    """Load the text causal LM. Encoder tensors stay in the checkpoint files."""
    for path in _weight_files(source):
        _drop_file_cache(path)
    _prime_cuda()
    saved_init = _skip_init()
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4ForCausalLM, Gemma4UnifiedForCausalLM

    print(f"Loading Gemma text tower from {source}", flush=True)
    config = AutoConfig.from_pretrained(source)
    text_config = _text_config(config)
    text_config._attn_implementation = "sdpa"
    # Manual construct skips from_pretrained's experts dispatch. Without this,
    # Gemma4TextExperts stays on the eager Python loop (config None → default
    # forward). grouped_mm is the transformers default when dispatch runs.
    if getattr(text_config, "enable_moe_block", False):
        text_config._experts_implementation_internal = "grouped_mm"
    causal_cls = Gemma4UnifiedForCausalLM if "unified" in str(getattr(config, "model_type", "")) else Gemma4ForCausalLM
    # Build in bf16 so the 31B text tower is one allocation, not a fp32 copy.
    # RoPE inv_freq is created as fp32 inside the embedding and must stay that way.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = causal_cls(text_config)
    finally:
        torch.set_default_dtype(previous)
        _restore_init(saved_init)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    for path in _weight_files(source):
        _drop_file_cache(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dest = _remap_key(key)
                if dest is None:
                    continue
                tensor = handle.get_tensor(key)
                with torch.no_grad():
                    if dest in params:
                        params[dest].copy_(tensor.to(dtype=params[dest].dtype))
                        loaded.add(dest)
                    elif dest in buffers and buffers[dest].shape == tensor.shape:
                        buffers[dest].copy_(tensor.to(dtype=buffers[dest].dtype))
                        loaded.add(dest)
        _drop_file_cache(path)
    if "lm_head.weight" not in loaded:
        model.lm_head.weight = model.model.embed_tokens.weight
        loaded.add("lm_head.weight")
    else:
        model.lm_head.weight = model.model.embed_tokens.weight
    missing = [name for name in params if name not in loaded and name != "lm_head.weight"]
    if missing:
        raise RuntimeError("Gemma text weights did not load: " + ", ".join(missing[:12]))
    model.seqlen = 2048
    model.eval()
    model.requires_grad_(False)
    print(f"loaded {len(loaded)} text tensors", flush=True)
    return model


def _true_sequential_groups() -> list[list[str]]:
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


@contextmanager
def _noop_experts(layer):
    """Skip expert compute; router still runs. Used when the residual is unused."""
    if not getattr(layer, "enable_moe_block", False):
        yield
        return
    experts = layer.experts

    def noop(hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)

    previous = experts.forward
    experts.forward = noop
    try:
        yield
    finally:
        if "forward" in experts.__dict__:
            del experts.__dict__["forward"]
        else:
            experts.forward = previous


def _collect_expert_inputs(layer, inps: torch.Tensor, shared, positions, masks, position_ids, layer_type) -> list[list[torch.Tensor]]:
    """Per-expert token inputs (after pre_feedforward_layernorm_2), on CPU.

    Expert matmuls are skipped: only the router indices and normed tokens matter
    for SparseGPT. The layer residual from this pass is discarded.
    """
    experts = layer.experts
    buckets: list[list[torch.Tensor]] = [[] for _ in range(experts.num_experts)]

    def capture(hidden_states, top_k_index, top_k_weights):
        # hidden_states: [T, H], top_k_index: [T, K]
        for expert_idx in top_k_index.unique().tolist():
            expert_idx = int(expert_idx)
            if expert_idx >= experts.num_experts:
                continue
            hit = (top_k_index == expert_idx).any(dim=-1)
            if hit.any():
                buckets[expert_idx].append(hidden_states[hit].detach().to("cpu"))
        return torch.zeros_like(hidden_states)

    previous = experts.forward
    experts.forward = capture
    try:
        print(f"collect experts layer inputs ({inps.shape[0]} samples)", flush=True)
        for j in range(inps.shape[0]):
            _run_layer(
                layer,
                inps[j].unsqueeze(0).to(DEV),
                None,
                shared[j],
                positions[layer_type],
                masks[layer_type],
                position_ids,
            )
            if (j + 1) % 32 == 0 or j + 1 == inps.shape[0]:
                print(f"collect experts {j + 1}/{inps.shape[0]}", flush=True)
    finally:
        if "forward" in experts.__dict__:
            del experts.__dict__["forward"]
        else:
            experts.forward = previous
    return buckets


def _prune_moe_experts(layer, buckets: list[list[torch.Tensor]], layer_idx: int) -> None:
    """SparseGPT each expert's fused gate_up and down using tokens routed to it."""
    experts = layer.experts
    min_tokens = 64
    hidden = experts.hidden_dim
    inter = experts.intermediate_dim
    pruned = 0
    for expert_idx, parts in enumerate(buckets):
        if not parts:
            continue
        tokens = torch.cat(parts, dim=0)
        if tokens.shape[0] < min_tokens:
            continue
        tokens = tokens.to(device=DEV, dtype=experts.gate_up_proj.dtype)
        # Copy slices into real Linears so SparseGPT's nn.Linear path applies.
        gate_up = nn.Linear(hidden, 2 * inter, bias=False, device=DEV, dtype=tokens.dtype)
        down = nn.Linear(inter, hidden, bias=False, device=DEV, dtype=tokens.dtype)
        gate_up.weight.data.copy_(experts.gate_up_proj[expert_idx])
        down.weight.data.copy_(experts.down_proj[expert_idx])
        for name, module, batch in (
            (f"experts[{expert_idx}].gate_up_proj", gate_up, tokens),
            (f"experts[{expert_idx}].down_proj", down, None),
        ):
            if (not (args.minlayer <= layer_idx < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            print(layer_idx, name, flush=True)
            print("Pruning ...", flush=True)
            gpt = SparseGPT(module)
            if name.endswith("gate_up_proj"):
                gpt.add_batch(batch, None)
            else:
                with torch.no_grad():
                    gate, up = gate_up(tokens).chunk(2, dim=-1)
                    act = experts.act_fn(gate) * up
                gpt.add_batch(act, None)
            gpt.fasterprune(
                args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            gpt.free()
        experts.gate_up_proj.data[expert_idx].copy_(gate_up.weight.data)
        experts.down_proj.data[expert_idx].copy_(down.weight.data)
        del tokens, gate_up, down
        pruned += 1
    print(f"layer {layer_idx}: pruned {pruned} experts", flush=True)


def _has_ple(module) -> bool:
    return bool(getattr(module, "hidden_size_per_layer_input", 0))


def _ple_to(text, device, *, embeddings: bool = True) -> None:
    if embeddings:
        text.embed_tokens = text.embed_tokens.to(device)
    if not _has_ple(text):
        return
    text.embed_tokens_per_layer = text.embed_tokens_per_layer.to(device)
    text.per_layer_model_projection = text.per_layer_model_projection.to(device)
    text.per_layer_projection_norm = text.per_layer_projection_norm.to(device)


def _embed(text, input_ids):
    """Hidden state the first layer sees, and projected per-layer inputs or None."""
    weight_device = text.embed_tokens.weight.device
    if input_ids.device != weight_device:
        input_ids = input_ids.to(weight_device)
    embeds = text.embed_tokens(input_ids)
    if not _has_ple(text):
        return embeds, None
    # Token embeddings stay on CPU during a 31B prune, while the per-layer
    # tables are small enough to sit on the GPU. Look them up where they live.
    ple_device = text.embed_tokens_per_layer.weight.device
    ple_ids = input_ids.to(ple_device)
    ple_embeds = embeds.to(ple_device)
    raw = text.get_per_layer_inputs(ple_ids, ple_embeds)
    return embeds, text.project_per_layer_inputs(ple_embeds, raw)


def _context(text, sample, position_ids):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mask_kwargs = {
        "config": text.config,
        "inputs_embeds": sample,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    positions = {
        layer_type: text.rotary_emb(sample, position_ids, layer_type) for layer_type in text.unique_layer_types
    }
    return masks, positions


def _run_layer(layer, hidden, per_layer_input, shared_kv, position_embeddings, attention_mask, position_ids):
    # Gemma 4 E2B/E4B take a per-layer embedding as the second argument.
    # Gemma 4 12B Unified and 31B do not; a positional tensor there would be
    # read as shared KV.
    kwargs = {
        "shared_kv_states": shared_kv,
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    if _has_ple(layer):
        out = layer(hidden, per_layer_input, **kwargs)
    else:
        out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _prefix_bos(ids: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Gemma 4 uses BOS as the attention sink. A window that does not start
    with it assigns near-uniform mass to the next token. Keep the length fixed
    by dropping the last id.
    """
    if ids.numel() == 0 or int(ids[0, 0]) == bos_id:
        return ids
    bos = torch.full((ids.shape[0], 1), bos_id, dtype=ids.dtype, device=ids.device)
    return torch.cat([bos, ids[:, :-1]], dim=1)


def _softcap(logits, config):
    cap = getattr(config, "final_logit_softcapping", None)
    if cap is None:
        return logits
    return torch.tanh(logits / cap) * cap


@torch.no_grad()
def gemma_sequential(model, dataloader, dev):
    print("Starting...", flush=True)
    _prime_cuda()
    free, total = torch.cuda.mem_get_info()
    print(f"cuda free={free / 1024**3:.1f}GiB total={total / 1024**3:.1f}GiB", flush=True)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    text = model.model
    layers = text.layers
    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    # The 31B embedding table is a few GiB. On this unified-memory machine the
    # weights already occupy most of RAM, and copying that table to CUDA fails
    # once C4 is loaded. Lookups stay on CPU; one sample moves with the layer.
    _ple_to(text, dev, embeddings=False)
    text.rotary_emb = text.rotary_emb.to(dev)

    inps = torch.zeros((args.nsamples, model.seqlen, hidden), dtype=dtype, device="cpu")
    ple = None
    if _has_ple(text):
        ple = torch.zeros(
            (
                args.nsamples,
                model.seqlen,
                text.config.num_hidden_layers,
                text.hidden_size_per_layer_input,
            ),
            dtype=dtype,
            device="cpu",
        )
    bos_id = int(getattr(model.config, "bos_token_id", 2))
    for index, batch in enumerate(dataloader):
        embeds, projected = _embed(text, _prefix_bos(batch[0], bos_id))
        inps[index] = embeds[0].to("cpu")
        if projected is not None:
            ple[index] = projected[0].to("cpu")
    _ple_to(text, "cpu", embeddings=False)
    torch.cuda.empty_cache()

    position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
    masks, positions = _context(text, inps[:1].to(dev), position_ids)
    shared = [{} for _ in range(args.nsamples)]
    outs = torch.zeros_like(inps)
    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer_type = text.config.layer_types[i]
        full = {name: module for name, module in find_layers(layer).items() if name in PRUNE_NAMES}
        groups = _true_sequential_groups() if args.true_sequential else [list(full.keys())]
        for names in groups:
            subset = {name: full[name] for name in names if name in full}
            gpts = {}
            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                gpts[name] = (GeometrySparseGPT if args.geometry_alpha > 0 else SparseGPT)(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
            if not gpts:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
            # MoE expert outputs are unused while collecting attn/MLP Hessians.
            with _noop_experts(layer):
                for j in range(args.nsamples):
                    ple_j = None if ple is None else ple[j, :, i, :].to(dev).unsqueeze(0)
                    outs[j] = _run_layer(
                        layer,
                        inps[j].unsqueeze(0).to(dev),
                        ple_j,
                        shared[j],
                        positions[layer_type],
                        masks[layer_type],
                        position_ids,
                    ).to("cpu")
            for handle in handles:
                handle.remove()
            for name in gpts:
                print(i, name, flush=True)
                print("Pruning ...", flush=True)
                apply_geometry(gpts[name], args, i, name)
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()
        if getattr(layer, "enable_moe_block", False) and args.sparsity > 0:
            # Experts are packed Parameters, not nn.Linear. Capture routed
            # tokens from a full layer forward, then SparseGPT each expert.
            buckets = _collect_expert_inputs(
                layer, inps, shared, positions, masks, position_ids, layer_type
            )
            _prune_moe_experts(layer, buckets, i)
            del buckets
        print(f"layer {i}: propagate outs", flush=True)
        for j in range(args.nsamples):
            ple_j = None if ple is None else ple[j, :, i, :].to(dev).unsqueeze(0)
            outs[j] = _run_layer(
                layer,
                inps[j].unsqueeze(0).to(dev),
                ple_j,
                shared[j],
                positions[layer_type],
                masks[layer_type],
                position_ids,
            ).to("cpu")
            if (j + 1) % 32 == 0 or j + 1 == args.nsamples:
                print(f"layer {i}: outs {j + 1}/{args.nsamples}", flush=True)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def gemma_eval(
    model,
    testenc,
    dev,
    dataset: str,
    log_wandb: bool = False,
    *,
    tokenizer=None,
    metrics_out: dict | None = None,
):
    print("Evaluating ...")
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
    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []
    for start in range(0, nsamples, chunk):
        stop = min(start + chunk, nsamples)
        if chunk < nsamples:
            print(f"eval chunk {start}:{stop}", flush=True)
        nlls.extend(_eval_windows(model, testenc, dev, start, stop, dtype))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})
    if tokenizer is not None:
        from .likelihood_metrics import pack_metrics, print_metrics_line

        bos_id = int(getattr(model.config, "bos_token_id", 2))
        chunks = []
        for index in range(nsamples):
            window = testenc[:, index * model.seqlen : (index + 1) * model.seqlen]
            labels = _prefix_bos(window, bos_id)[:, 1:].reshape(-1)
            chunks.append(labels.detach().to("cpu").long())
        scored = torch.cat(chunks) if chunks else testenc.new_empty((0,), dtype=torch.long)
        metrics = pack_metrics(
            nlls,
            seqlen=model.seqlen,
            scored_token_ids=scored,
            tokenizer=tokenizer,
        )
        print_metrics_line(metrics)
        if metrics_out is not None:
            metrics_out.clear()
            metrics_out.update(metrics)
    model.config.use_cache = use_cache
    return ppl


def _eval_windows(model, testenc, dev, start, stop, dtype):
    text = model.model
    layers = text.layers
    count = stop - start
    hidden = model.config.hidden_size
    _ple_to(text, dev)
    text.rotary_emb = text.rotary_emb.to(dev)
    inps = torch.zeros((count, model.seqlen, hidden), dtype=dtype, device=dev)
    ple = None
    if _has_ple(text):
        ple = torch.zeros(
            (count, model.seqlen, text.config.num_hidden_layers, text.hidden_size_per_layer_input),
            dtype=dtype,
            device="cpu",
        )
    bos_id = int(getattr(model.config, "bos_token_id", 2))
    for offset, index in enumerate(range(start, stop)):
        ids = _prefix_bos(testenc[:, index * model.seqlen : (index + 1) * model.seqlen], bos_id).to(dev)
        embeds, projected = _embed(text, ids)
        inps[offset] = embeds[0]
        if projected is not None:
            ple[offset] = projected[0].to("cpu")
    _ple_to(text, "cpu")
    torch.cuda.empty_cache()

    position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
    masks, positions = _context(text, inps[:1], position_ids)
    shared = [{} for _ in range(count)]
    outs = torch.zeros_like(inps)
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer_type = text.config.layer_types[i]
        for j in range(count):
            ple_j = None if ple is None else ple[j, :, i, :].to(dev).unsqueeze(0)
            outs[j] = _run_layer(
                layer,
                inps[j].unsqueeze(0),
                ple_j,
                shared[j],
                positions[layer_type],
                masks[layer_type],
                position_ids,
            )
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    text.norm = text.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for offset, index in enumerate(range(start, stop)):
        hidden_states = text.norm(inps[offset].unsqueeze(0))
        logits = _softcap(model.lm_head(hidden_states), model.config)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = _prefix_bos(
            testenc[:, index * model.seqlen : (index + 1) * model.seqlen], bos_id
        )[:, 1:].to(dev)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    text.norm = text.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return nlls


@torch.no_grad()
def gemma_reference_nlls(model, testenc, dev):
    """One full forward per window, for the dense E2B check."""
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    model.to(dev)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    bos_id = int(getattr(model.config, "bos_token_id", 2))
    for index in range(nsamples):
        ids = _prefix_bos(testenc[:, index * model.seqlen : (index + 1) * model.seqlen], bos_id).to(dev)
        logits = model(ids, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    model.to("cpu")
    torch.cuda.empty_cache()
    return nlls


if __name__ == "__main__":
    import argparse

    from .datautils import get_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Gemma 4 model, e.g. google/gemma-4-E2B.")
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--gmp", action="store_true")
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune_only", type=str, default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument(
        "--eval-datasets",
        type=str,
        nargs="+",
        default=["wikitext2", "ptb", "c4"],
        choices=["wikitext2", "ptb", "c4"],
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--load-cached", action="store_true")
    add_geometry_arguments(parser)
    args = parser.parse_args()
    if args.gmp:
        parser.error("Gemma magnitude pruning is not part of this sweep")
    if not math.isfinite(args.geometry_alpha) or args.geometry_alpha < 0:
        parser.error("--geometry-alpha must be finite and nonnegative")
    if args.geometry_alpha > 0 and args.geometry_probes is None:
        parser.error("--geometry-probes is required with positive --geometry-alpha")

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    pruning = bool(args.sparsity or args.prunen)
    config = prune_config_from_args(args)
    cached = find_pruned_model(config, args.save) if pruning else None
    if args.load_cached and pruning and cached is None:
        raise SystemExit("No pruned checkpoint matches this config; refusing to prune.")

    if cached is not None:
        print(f"Loading pruned model from {cached} (matches pruning config).")
        model = get_gemma(str(cached))
    else:
        if pruning:
            print("No pruned model in sparse_opt matches this config; pruning from the base model.")
        model = get_gemma(args.model)
    model.eval()

    ran_sparsegpt = False
    if cached is None and pruning:
        dataloader, _testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        tick = time.time()
        gemma_sequential(model, dataloader, DEV)
        for name, param in model.named_parameters():
            if is_prunable_parameter(name):
                print(name, torch.mean((param == 0).float()).item())
        print(time.time() - tick)
        ran_sparsegpt = True

    if not args.skip_eval:
        for dataset in args.eval_datasets:
            _dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print("Dataset:", dataset)
            gemma_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save and cached is None and ran_sparsegpt:
        dest = resolve_save_dir(args.save, config)
        print(f"Saving pruned model to {dest}")
        model.save_pretrained(dest)
        write_prune_config(dest, config)
