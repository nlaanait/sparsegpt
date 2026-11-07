"""Export Variant A input differences for the default concept-pair bank.

One Qwen forward per side of each knob. Only the edited token is stored.
The CUDA allocator is capped before the checkpoint loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sparsifier.data.concept_pairs import (
    DEFAULT_CONCEPT_PAIRS,
    PRESENCE_REFERENCES,
    edit_index,
    encode_pair,
    knob_ids,
)
from sparsifier.data.pair_deltas import capture_edit_deltas, capture_position_mean, save_pair_deltas
from sparsifier.models.layers import find_layers
from sparsifier.models.qwen import force_pytorch_attention, get_qwen


def cap_cuda(max_bytes: int) -> int:
    """Limit this process to ``max_bytes``, and to a quarter of whatever is free."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the concept-pair forward")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if total_bytes <= 0 or free_bytes <= 0:
        raise RuntimeError(f"CUDA memory info is unusable (free={free_bytes}, total={total_bytes})")
    cap = min(int(max_bytes), int(free_bytes * 0.25))
    floor = 3 * 1024**3
    if cap < floor:
        raise RuntimeError(
            f"refusing to load the model: cap would be {cap / 1024**3:.2f} GiB "
            f"with {free_bytes / 1024**3:.2f} GiB free"
        )
    torch.cuda.set_per_process_memory_fraction(cap / total_bytes)
    return cap


def decoder_linears(model) -> dict[str, torch.nn.Module]:
    layers = model.model.layers
    modules: dict[str, torch.nn.Module] = {}
    for index, layer in enumerate(layers):
        for name, module in find_layers(layer).items():
            modules[f"{index}/{name}"] = module
    if not modules:
        raise RuntimeError("no decoder linears found on model.model.layers")
    return modules


@torch.no_grad()
def export_concept_pairs(
    out_dir: Path,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_bytes: int = 8 * 1024**3,
) -> Path:
    """Run the nine knobs and write ``{block}/{module}.npy`` plus knob ids."""
    cap = cap_cuda(max_bytes)
    print(f"CUDA allocator cap: {cap / 1024**3:.2f} GiB", flush=True)
    force_pytorch_attention()
    model = get_qwen(model_id, use_kernels=False)
    model.eval()
    model.to("cuda")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    modules = decoder_linears(model)
    ids = knob_ids(DEFAULT_CONCEPT_PAIRS)
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    manifest = []

    def run(token_ids: torch.Tensor) -> None:
        model(input_ids=token_ids)

    for pair, knob_id in zip(DEFAULT_CONCEPT_PAIRS, ids.tolist()):
        index = edit_index(tokenizer, pair)
        left, right = encode_pair(tokenizer, pair)
        left_ids = torch.tensor([left], dtype=torch.long, device="cuda")
        right_ids = torch.tensor([right], dtype=torch.long, device="cuda")
        deltas = capture_edit_deltas(modules, run, left_ids, right_ids, index)
        for name, delta in deltas.items():
            rows[name].append(delta)
        manifest.append(
            {
                "knob": pair.knob,
                "domain": pair.domain,
                "knob_id": int(knob_id),
                "edit_index": index,
                "left": pair.left,
                "right": pair.right,
            }
        )
        print(f"captured {pair.knob} at token {index}", flush=True)
        del left_ids, right_ids, deltas

    stacked = {name: torch.stack(vectors, dim=0) for name, vectors in rows.items()}
    knobs = {name: ids for name in stacked}
    out_dir = Path(out_dir)
    save_pair_deltas(out_dir, stacked, knobs)
    (out_dir / "manifest.json").write_text(json.dumps({"model": model_id, "pairs": manifest}, indent=2))
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak CUDA allocated: {peak:.2f} GiB", flush=True)
    print(f"wrote {len(stacked)} linears under {out_dir}", flush=True)
    return out_dir


def _token_ids(tokenizer, text: str, positions: tuple[int, ...], device: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if any(index >= len(ids) for index in positions):
        raise ValueError(f"{text!r} is shorter than positions {positions}")
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def export_presence_pairs(
    out_dir: Path,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_bytes: int = 8 * 1024**3,
) -> Path:
    """Store h(concept) - h_0 at the agreed absent references, one row per knob."""
    cap = cap_cuda(max_bytes)
    print(f"CUDA allocator cap: {cap / 1024**3:.2f} GiB", flush=True)
    force_pytorch_attention()
    model = get_qwen(model_id, use_kernels=False)
    model.eval()
    model.to("cuda")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    modules = decoder_linears(model)
    cache: dict[tuple[str, tuple[int, ...]], dict[str, torch.Tensor]] = {}

    def means(text: str, positions: tuple[int, ...]) -> dict[str, torch.Tensor]:
        key = (text, positions)
        if key not in cache:
            token_ids = _token_ids(tokenizer, text, positions, "cuda")
            cache[key] = capture_position_mean(modules, lambda ids: model(input_ids=ids), token_ids, positions)
            del token_ids
        return cache[key]

    rows: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    manifest = []
    ids = knob_ids(DEFAULT_CONCEPT_PAIRS)
    if [item.knob for item in PRESENCE_REFERENCES] != [pair.knob for pair in DEFAULT_CONCEPT_PAIRS]:
        raise RuntimeError("presence references must follow the nine knob ids")
    for reference, knob_id in zip(PRESENCE_REFERENCES, ids.tolist()):
        absent = means(reference.absent, reference.positions)
        present_rows = [means(text, reference.positions) for text in reference.present]
        for name in modules:
            samples = torch.stack([present[name] - absent[name] for present in present_rows], dim=0)
            rows[name].append(samples.mean(dim=0))
        manifest.append(
            {
                "knob": reference.knob,
                "knob_id": int(knob_id),
                "present": list(reference.present),
                "absent": reference.absent,
                "positions": list(reference.positions),
            }
        )
        print(f"captured presence {reference.knob}", flush=True)

    stacked = {name: torch.stack(vectors, dim=0) for name, vectors in rows.items()}
    knobs = {name: ids for name in stacked}
    out_dir = Path(out_dir)
    save_pair_deltas(out_dir, stacked, knobs)
    (out_dir / "manifest.json").write_text(
        json.dumps({"model": model_id, "kind": "presence", "pairs": manifest}, indent=2)
    )
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak CUDA allocated: {peak:.2f} GiB", flush=True)
    print(f"wrote {len(stacked)} linears under {out_dir}", flush=True)
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tangent/qwen35-0.8b-concept-pairs"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-gib", type=float, default=8.0)
    parser.add_argument("--presence", action="store_true")
    args = parser.parse_args()
    if args.presence:
        out_dir = args.out_dir
        if out_dir == Path("results/tangent/qwen35-0.8b-concept-pairs"):
            out_dir = Path("results/tangent/qwen35-0.8b-presence-pairs")
        export_presence_pairs(out_dir, model_id=args.model, max_bytes=int(args.max_gib * 1024**3))
    else:
        export_concept_pairs(args.out_dir, model_id=args.model, max_bytes=int(args.max_gib * 1024**3))


if __name__ == "__main__":
    main()
