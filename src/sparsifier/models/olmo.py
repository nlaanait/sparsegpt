"""Unstructured SparseGPT for OLMo 2.

Instruction-tuned Olmo2ForCausalLM checkpoints, SDPA, sequence length 2048.
Decoder linears only. Raw WikiText2 / PTB / C4 perplexity with no chat template.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sparsifier.metrics import Likelihood
from sparsifier.models.layers import DEV, find_layers
from sparsifier.models.loop import (
    Catcher,
    capture_cache,
    decoder_groups,
    eval_chunk_size,
    likelihood_from_nlls,
    run_model,
)

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
def _capture_olmo(model, dataloader, dev):
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
    layers[0] = Catcher(layers[0], inps, cache, squeeze=True)
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

    position_ids = cache["position_ids"]
    return {
        "hidden": inps,
        "outs": torch.zeros_like(inps),
        "use_cache": use_cache,
        "propagate": False,
        "attention_mask": _causal_mask(model, inps[:1], position_ids),
        "position_embeddings": cache["position_embeddings"],
        "position_ids": position_ids,
    }


def _forward_olmo(layer, hidden, captured):
    outs = captured["outs"]
    for index in range(hidden.shape[0]):
        outs[index] = _run_layer(
            layer,
            hidden[index].unsqueeze(0),
            attention_mask=captured["attention_mask"],
            position_embeddings=captured["position_embeddings"],
            position_ids=captured["position_ids"],
        )[0]
    return outs


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
    model.config.use_cache = use_cache
    _ = (dataset, log_wandb)
    return likelihood_from_nlls(nlls, nsamples, model.seqlen)


def _eval_windows(model, testenc, dev, start, stop, dtype):
    layers = model.model.layers
    count = stop - start
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((count, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = capture_cache()
    layers[0] = Catcher(layers[0], inps, cache, squeeze=True)
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


def _report_sparsity(model) -> None:
    for name, param in model.named_parameters():
        if is_prunable_parameter(name):
            print(name, torch.mean((param == 0).float()).item())


class Olmo:
    """Decoder family for OLMo 2 Instruct."""

    name = "olmo"

    def matches(self, model_id: str) -> bool:
        return model_id.startswith("allenai/OLMo")

    def load(self, model_id: str):
        return get_olmo(model_id)

    def prepare(self, model) -> None:
        model.eval()

    def capture(self, model, batches, config):
        _ = config
        return _capture_olmo(model, batches, DEV)

    def forward_layer(self, layer, hidden, captured):
        return _forward_olmo(layer, hidden, captured)

    def prunable(self, layer) -> dict:
        return {name: module for name, module in find_layers(layer).items() if name in PRUNE_NAMES}

    def sequential_groups(self, layer) -> list[list[str]]:
        _ = layer
        return decoder_groups()

    def likelihood(self, model, encoded, config) -> Likelihood:
        _ = config
        return olmo_eval(model, encoded, DEV, "", False)


OLMO = Olmo()


def run(cli_args) -> None:
    """Load, prune, evaluate, and optionally save one OLMo checkpoint."""
    global args
    args = cli_args
    run_model(OLMO, args, _report_sparsity)
