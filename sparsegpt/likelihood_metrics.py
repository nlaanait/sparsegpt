"""Tokenizer-agnostic likelihood fields for offline BPB / BPC / perplexity.

SparseGPT eval scores ``seqlen - 1`` tokens per non-overlapping window (the
shifted labels). ``sum_nll_nats`` is the true sum of ``-ln p`` over those
positions. ``exp(sum_nll_nats / n_tokens)`` matches the printed SparseGPT
perplexity, which divides the same mean CE by ``nsamples * seqlen`` after
multiplying each window mean by ``seqlen``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

WORD_RULE = "whitespace"


def sparsity_key(sparsity: float) -> str:
    if abs(float(sparsity)) < 1e-12:
        return "0"
    text = f"{float(sparsity):.5f}".rstrip("0").rstrip(".")
    return text or "0"


def scored_label_ids(input_ids: torch.Tensor, seqlen: int) -> torch.Tensor:
    """Concatenate label token ids (drop the first token of each window)."""
    flat = input_ids.reshape(-1)
    nsamples = int(flat.numel() // seqlen)
    chunks = [
        flat[index * seqlen + 1 : (index + 1) * seqlen]
        for index in range(nsamples)
    ]
    if not chunks:
        return flat.new_empty((0,), dtype=torch.long)
    return torch.cat(chunks)


def text_unit_counts(token_ids: torch.Tensor, tokenizer) -> dict[str, int]:
    """UTF-8 bytes, Unicode chars, and whitespace words of decoded scored ids."""
    ids = token_ids.detach().to("cpu").long().reshape(-1).tolist()
    if not ids:
        return {"n_bytes": 0, "n_chars": 0, "n_words": 0}
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = text.encode("utf-8")
    return {
        "n_bytes": len(encoded),
        "n_chars": len(text),
        "n_words": len(text.split()),
    }


def pack_metrics(
    window_nlls: Sequence[torch.Tensor | float],
    *,
    seqlen: int,
    scored_token_ids: torch.Tensor,
    tokenizer,
) -> dict[str, Any]:
    """Build the saved fields from SparseGPT window NLLs and scored token ids.

    ``window_nlls`` entries are ``mean_ce * seqlen`` as in the existing evals.
    """
    if seqlen < 2:
        raise ValueError(f"seqlen must be >= 2, got {seqlen}")
    scored = int(seqlen - 1)
    total = 0.0
    for value in window_nlls:
        total += float(value) * scored / seqlen
    n_tokens = len(window_nlls) * scored
    units = text_unit_counts(scored_token_ids, tokenizer)
    if n_tokens != int(scored_token_ids.numel()):
        # Prefer the CE count; decode still uses the provided ids.
        pass
    perplexity = math.exp(total / n_tokens) if n_tokens else float("nan")
    cross_entropy_nats = total / n_tokens if n_tokens else float("nan")
    n_bytes = units["n_bytes"]
    n_chars = units["n_chars"]
    n_words = units["n_words"]
    ln2 = math.log(2.0)
    return {
        "sum_nll_nats": total,
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "n_words": n_words,
        "perplexity": perplexity,
        "cross_entropy_nats": cross_entropy_nats,
        "bits_per_token": cross_entropy_nats / ln2 if n_tokens else float("nan"),
        "bits_per_byte": total / (n_bytes * ln2) if n_bytes else float("nan"),
        "bits_per_character": total / (n_chars * ln2) if n_chars else float("nan"),
        "byte_perplexity": math.exp(total / n_bytes) if n_bytes else float("nan"),
        "word_perplexity": math.exp(total / n_words) if n_words else float("nan"),
    }


def empty_file(
    *,
    model: str,
    dataset: str,
    calibration: str,
    attention: str,
    seqlen: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "calibration": calibration,
        "attention": attention,
        "seqlen": seqlen,
        "word_rule": WORD_RULE,
    }


def load_metrics_file(path: Path, **meta) -> dict[str, Any]:
    if path.is_file():
        data = json.loads(path.read_text())
        for key, value in meta.items():
            data.setdefault(key, value)
        data.setdefault("word_rule", WORD_RULE)
        return data
    return empty_file(**meta)


def save_metrics_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def upsert_sparsity(data: dict[str, Any], sparsity: float, metrics: dict[str, Any]) -> None:
    data[sparsity_key(sparsity)] = metrics


def has_sparsity(data: dict[str, Any], sparsity: float) -> bool:
    entry = data.get(sparsity_key(sparsity))
    return isinstance(entry, dict) and "sum_nll_nats" in entry


def print_metrics_line(metrics: dict[str, Any]) -> None:
    payload = {
        key: metrics[key]
        for key in (
            "sum_nll_nats",
            "n_tokens",
            "n_bytes",
            "n_chars",
            "n_words",
            "perplexity",
            "bits_per_byte",
        )
        if key in metrics
    }
    print("LikelihoodMetrics:", json.dumps(payload, allow_nan=False), flush=True)


def accumulate_window_metrics(
    window_nlls: Iterable[torch.Tensor | float],
    label_chunks: Iterable[torch.Tensor],
    *,
    seqlen: int,
    tokenizer,
) -> dict[str, Any]:
    nlls = list(window_nlls)
    labels = [chunk.detach().to("cpu").long().reshape(-1) for chunk in label_chunks]
    scored = (
        torch.cat(labels)
        if labels
        else torch.empty((0,), dtype=torch.long)
    )
    return pack_metrics(nlls, seqlen=seqlen, scored_token_ids=scored, tokenizer=tokenizer)
