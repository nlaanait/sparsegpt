"""Unstructured SparseGPT for OLMo 2.

Instruction-tuned Olmo2ForCausalLM checkpoints, SDPA, sequence length 2048.
Decoder linears only. Raw WikiText2 / PTB / C4 perplexity with no chat template.
"""

from __future__ import annotations

import math
import time

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
    return any(name.endswith(f"{leaf}.weight") or name.endswith(f"{leaf}.bias") for leaf in PRUNE_NAMES)


def get_olmo(source: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype="auto",
        attn_implementation="sdpa",
    )
    if getattr(model.config, "model_type", "") != "olmo2":
        raise ValueError(f"expected olmo2, got {model.config.model_type!r} for {source}")
    model.seqlen = 2048
    model.eval()
    return model


def _true_sequential_groups() -> list[list[str]]:
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def _leading_example(value):
    if isinstance(value, tuple):
        return tuple(_leading_example(item) for item in value)
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] > 1:
        return value[:1]
    return value


def _as_hidden(out):
    if isinstance(out, tuple):
        out = out[0]
    return out


def _run_layer(layer, hidden, *, attention_mask, position_embeddings, position_ids):
    return _as_hidden(
        layer(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
    )


def _causal_mask(model, sample, position_ids):
    from transformers.masking_utils import create_causal_mask

    return create_causal_mask(
        config=model.model.config,
        inputs_embeds=sample,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )


@torch.no_grad()
def olmo_sequential(model, dataloader, dev):
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
            inps[cache["i"]] = inp[0] if inp.ndim == 3 else inp
            cache["i"] += 1
            cache["attention_mask"] = _leading_example(kwargs.get("attention_mask"))
            cache["position_embeddings"] = _leading_example(kwargs.get("position_embeddings"))
            cache["position_ids"] = _leading_example(kwargs.get("position_ids"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev), use_cache=False)
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
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    attention_mask = _causal_mask(model, inps[:1], position_ids)
    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = {name: module for name, module in find_layers(layer).items() if name in PRUNE_NAMES}
        groups = _true_sequential_groups() if args.true_sequential else [list(full.keys())]
        for names in groups:
            subset = {name: full[name] for name in names if name in full}
            gpts = {}
            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (
                    not args.invert
                ):
                    continue
                gpts[name] = (GeometrySparseGPT if args.geometry_alpha > 0 else SparseGPT)(
                    subset[name]
                )
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )
            if not gpts:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
            for j in range(args.nsamples):
                outs[j] = _run_layer(
                    layer,
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                )[0]
            for handle in handles:
                handle.remove()
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

        for j in range(args.nsamples):
            outs[j] = _run_layer(
                layer,
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def olmo_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
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
    model.config.use_cache = use_cache
    return ppl


def _eval_windows(model, testenc, dev, start, stop, dtype):
    layers = model.model.layers
    count = stop - start
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((count, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
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
            inps[cache["i"]] = inp[0] if inp.ndim == 3 else inp
            cache["i"] += 1
            cache["attention_mask"] = _leading_example(kwargs.get("attention_mask"))
            cache["position_embeddings"] = _leading_example(kwargs.get("position_embeddings"))
            cache["position_ids"] = _leading_example(kwargs.get("position_ids"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for index in range(start, stop):
        batch = testenc[:, index * model.seqlen : (index + 1) * model.seqlen].to(dev)
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
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    attention_mask = _causal_mask(model, inps[:1], position_ids)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(count):
            outs[j] = _run_layer(
                layer,
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for offset, index in enumerate(range(start, stop)):
        hidden_states = inps[offset].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, index * model.seqlen : (index + 1) * model.seqlen][:, 1:].to(dev)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return nlls


if __name__ == "__main__":
    import argparse

    from .datautils import get_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="OLMo 2 model, e.g. allenai/OLMo-2-1124-7B-Instruct.")
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
        parser.error("OLMo 2 magnitude pruning is not part of this sweep")
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
        model = get_olmo(str(cached))
    else:
        if pruning:
            print("No pruned model in sparse_opt matches this config; pruning from the base model.")
        model = get_olmo(args.model)
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
        olmo_sequential(model, dataloader, DEV)
        for name, param in model.named_parameters():
            if is_prunable_parameter(name):
                print(name, torch.mean((param == 0).float()).item())
        print(time.time() - tick)
        ran_sparsegpt = True

    if not args.skip_eval:
        for dataset in args.eval_datasets:
            _dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            olmo_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save and cached is None and ran_sparsegpt:
        dest = resolve_save_dir(args.save, config)
        print(f"Saving pruned model to {dest}")
        model.save_pretrained(dest)
        write_prune_config(dest, config)
