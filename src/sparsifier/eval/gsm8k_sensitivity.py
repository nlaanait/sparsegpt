"""Aligned answer-conditioned gradients for the GSM8K empirical-Fisher control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.memory_guard import configure_cuda


def balanced_split(rows, split, count, seed=0):
    groups = {}
    for row in rows:
        if row["split"] == split:
            groups.setdefault(row["family"], []).append(row)
    rng = np.random.default_rng(seed)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while groups and len(result) < count:
        for key in sorted(list(groups)):
            result.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(result) == count:
                break
    if len(result) < count:
        raise ValueError(f"not enough {split} examples")
    return result


def answer_gradient(model, layer, tokens, prefix_length):
    """Gradient at the question readout, from the same forward as its input.

    Mark only this layer output as a differentiation leaf. Frozen upstream
    weights need no graph; frozen downstream weights still propagate dL/dh.
    """
    captured = {}
    def hook(_m, args, output):
        captured["x"] = args[0][0, prefix_length - 1].detach().float().cpu()
        output.requires_grad_(True)
        captured["y"] = output
    handle = layer.register_forward_hook(hook)
    try:
        length = tokens.shape[1] - prefix_length
        if length < 1:
            raise ValueError("answer target required")
        logits = model(input_ids=tokens, use_cache=False, logits_to_keep=length + 1).logits
        loss = F.cross_entropy(logits[0, -length-1:-1].float(), tokens[0, prefix_length:])
        gradient = torch.autograd.grad(loss, captured["y"])[0][0, prefix_length - 1]
        return captured["x"], gradient.detach().float().cpu(), float(loss.detach())
    finally:
        handle.remove()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    manifest = load_manifest(args.manifest)
    rows = balanced_split(manifest["rows"], "pruning_calibration", args.count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "block": 15, "module": "mlp.down_proj",
              "manifest_sha256": manifest["manifest_sha256"], "ids": [r["id"] for r in rows],
              "seed": args.seed, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gradient_source": "ground_truth_no_rationale_completion_mean_token_nll",
              "classification": "empirical_fisher_style_not_model_fisher"}
    write_immutable(args.output / "config.json", config)
    if (args.output / "report.json").exists():
        if json.loads((args.output / "report.json").read_text()).get("complete"):
            print("Already complete", flush=True)
            return
    memory = configure_cuda(56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer = model.model.layers[15].mlp.down_proj
    inputs, gradients, losses = [], [], []
    for i, row in enumerate(rows):
        path = args.output / (digest(row["id"])[:24] + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["config_sha256"]) != digest(config):
                    raise ValueError("gradient cache provenance mismatch")
                x, g, loss = saved["input"].copy(), saved["gradient"].copy(), float(saved["loss"])
        else:
            prefix = tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False)
            answer = tokenizer.encode("Final answer: " + row["answer"], add_special_tokens=False)
            tokens = torch.tensor([prefix + answer], device="cuda")
            x, g, loss = answer_gradient(model, layer, tokens, len(prefix))
            x, g = x.numpy(), g.numpy()
            with path.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(handle, input=x, gradient=g, loss=loss, config_sha256=np.array(digest(config)))
            path.with_suffix(".tmp").replace(path)
        if not np.isfinite(g).all() or not np.isfinite(x).all():
            raise RuntimeError("nonfinite gradient or activation")
        inputs.append(x); gradients.append(g); losses.append(loss)
        print(json.dumps({"gradients": i+1, "count": len(rows), "loss": loss}), flush=True)
    with (args.output / "aligned.npz").open("wb") as handle:
        np.savez_compressed(handle, inputs=np.stack(inputs), gradients=np.stack(gradients), losses=losses,
                            ids=np.array(config["ids"]), config_sha256=np.array(digest(config)))
    atomic_json(args.output / "report.json", {"complete": True, "config_sha256": digest(config),
        "count": len(rows), "mean_answer_nll": float(np.mean(losses)), "memory": memory,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})


if __name__ == "__main__":
    main()
