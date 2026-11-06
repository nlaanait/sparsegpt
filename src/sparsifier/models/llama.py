import torch
import torch.nn as nn

from sparsifier.metrics import Likelihood
from sparsifier.models.layers import DEV, find_layers
from sparsifier.models.loop import (
    Catcher,
    capture_cache,
    likelihood_from_nlls,
    run_model,
    write_bank,
)

args = None


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


_LLAMA_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


@torch.no_grad()
def _capture_llama(model, dataloader, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    from sparsifier.models.loop import calibration_window_count

    inps = torch.zeros(
        (calibration_window_count(args), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = capture_cache(positions=False)
    layers[0] = Catcher(
        layers[0],
        inps,
        cache,
        fields=("attention_mask",),
        leading=False,
        required=("attention_mask",),
    )
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    return {
        "hidden": inps,
        "outs": torch.zeros_like(inps),
        "use_cache": use_cache,
        "propagate": False,
        "attention_mask": cache["attention_mask"],
    }


def _forward_llama(layer, hidden, captured):
    outs = captured["outs"]
    attention_mask = captured["attention_mask"]
    for index in range(hidden.shape[0]):
        out = layer(hidden[index].unsqueeze(0), attention_mask=attention_mask)[0]
        write_bank(outs[index], out)
        del out
    return outs


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = capture_cache(positions=False)
    layers[0] = Catcher(
        layers[0],
        inps,
        cache,
        fields=("attention_mask",),
        leading=False,
        required=("attention_mask",),
    )
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            write_bank(outs[j], out)
            del out
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    model.config.use_cache = use_cache
    _ = (dataset, log_wandb)
    return likelihood_from_nlls(nlls, nsamples, model.seqlen)


def _report_sparsity(model) -> None:
    for name, param in model.named_parameters():
        print(name, torch.mean((param == 0).float()))
        if "down_proj" in name:
            break


class Llama:
    """Decoder family for Llama 3."""

    name = "llama"

    def matches(self, model_id: str) -> bool:
        return model_id.startswith("meta-llama/")

    def load(self, model_id: str):
        return get_llama(model_id)

    def prepare(self, model) -> None:
        model.eval()

    def capture(self, model, batches, config):
        _ = config
        return _capture_llama(model, batches, DEV)

    def forward_layer(self, layer, hidden, captured):
        return _forward_llama(layer, hidden, captured)

    def prunable(self, layer) -> dict:
        return find_layers(layer)

    def sequential_groups(self, layer) -> list[list[str]]:
        _ = layer
        return _LLAMA_GROUPS

    def likelihood(self, model, encoded, config) -> Likelihood:
        _ = config
        return llama_eval(model, encoded, DEV, "", False)


LLAMA = Llama()


def run(cli_args) -> None:
    """Load, prune, evaluate, and optionally save one Llama checkpoint."""
    global args
    args = cli_args
    run_model(LLAMA, args, _report_sparsity)
