"""Shared batch-eight execution policy for GSM8K experiments.

Membership follows manifest order, including on resume. Partial batches are
regenerated intact so padding and companion examples do not change on resume.
"""
from __future__ import annotations

BATCH_SIZE = 8
VERSION = "gsm8k-batch8-v1"


def policy():
    from hashlib import sha256
    from pathlib import Path
    return {"version": VERSION, "batch_size": BATCH_SIZE, "padding_side": "left",
            "membership": "contiguous_manifest_order_regenerate_partial_batch",
            "seconds": "batch_wall_time_divided_by_batch_size",
            "backend": "verified_pytorch_reference",
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def batches(values, size=BATCH_SIZE):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("positive integer batch size required")
    for start in range(0, len(values), size):
        yield values[start:start+size]


def pad_id(tokenizer):
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", 0)
    return 0 if value is None else value


def pack(sequences, tokenizer, device):
    """Left pad without changing tokenizer state; all readouts remain at -1."""
    import torch
    if not len(sequences) or any(not len(s) or len(s) > 2048 for s in sequences):
        raise ValueError("prompt lengths must be in [1, 2048]")
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id(tokenizer), device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, -len(sequence):] = torch.as_tensor(sequence, device=device)
        mask[row, -len(sequence):] = 1
    return {"input_ids": ids, "attention_mask": mask}


def trim_generated(tokens, eos):
    """Remove padding after each example's first EOS, retaining that EOS."""
    for position, token in enumerate(tokens):
        if token in eos:
            return tokens[:position+1]
    return tokens


def reference_backend():
    """Unwrap and verify all four Qwen reference kernels before real batching."""
    import inspect
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    identities = {}
    for name in _PYTORCH_ATTENTION:
        fn = getattr(modeling, name)
        seen = set()
        while True:
            if id(fn) in seen:
                raise RuntimeError("cyclic reference backend wrapper")
            seen.add(id(fn))
            if isinstance(fn, torch.nn.Module):
                closure = inspect.getclosurevars(fn.forward).nonlocals
                if "func" not in closure:
                    raise RuntimeError(f"unsupported backend module wrapper for {name}")
                fn = closure["func"]
            elif hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
            else:
                break
        if not inspect.isfunction(fn) or fn.__name__ != name:
            raise RuntimeError(f"could not verify reference backend for {name}")
        setattr(modeling, name, fn)
        identities[name] = fn.__module__+"."+fn.__qualname__
    return identities
