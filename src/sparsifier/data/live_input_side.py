"""One-layer Variant A prune on Qwen 3.5-0.8B using saved concept-pair deltas.

Calibration is a short random-token forward. The CUDA allocator is capped
before the checkpoint loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sparsifier.data.export_concept_pairs import cap_cuda, decoder_linears
from sparsifier.data.pair_deltas import load_knob_ids
from sparsifier.models.qwen import force_pytorch_attention, get_qwen
from sparsifier.prune.input_side import InputSideSparseGPT


def prune_recorded_layer(
    pairs: Path,
    *,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    module_name: str = "mlp.down_proj",
    block_index: int = 0,
    sparsity: float = 0.5,
    max_bytes: int = 8 * 1024**3,
    calibration_sequences: int = 8,
    sequence_length: int = 32,
) -> dict[str, float]:
    cap = cap_cuda(max_bytes)
    print(f"CUDA allocator cap: {cap / 1024**3:.2f} GiB", flush=True)
    delta_path = Path(pairs) / str(block_index) / f"{module_name}.npy"
    if not delta_path.is_file():
        raise FileNotFoundError(delta_path)
    force_pytorch_attention()
    model = get_qwen(model_id, use_kernels=False)
    model.eval()
    model.to("cuda")
    modules = decoder_linears(model)
    key = f"{block_index}/{module_name}"
    if key not in modules:
        raise KeyError(f"{key} is not a decoder linear")
    layer = modules[key]
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        activation = inputs[0].detach().float().cpu()
        captured.append(activation.reshape(-1, activation.shape[-1]))

    handle = layer.register_forward_pre_hook(hook)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    vocab = int(model.config.vocab_size)
    try:
        with torch.no_grad():
            for _ in range(calibration_sequences):
                token_ids = torch.randint(1, vocab, (1, sequence_length), device="cuda", generator=generator)
                model(input_ids=token_ids)
    finally:
        handle.remove()
    inputs = torch.cat(captured, dim=0).to(device=layer.weight.device)
    solver = InputSideSparseGPT(layer)
    solver.add_batch(inputs.unsqueeze(0), None)
    deltas = torch.as_tensor(np.load(delta_path))
    solver.add_pairs(deltas, knob_ids=load_knob_ids(delta_path))
    solver.finalize_pairs(alpha_in=1.0)
    before = layer.weight.data.detach().float().cpu().clone()
    solver.fasterprune(sparsity, blocksize=128)
    after = layer.weight.data.detach().float().cpu()
    zeros = int((after == 0).sum())
    report = {
        "cap_gib": cap / 1024**3,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "sparsity": zeros / after.numel(),
        "changed_entries": int((after != before).sum()),
        "finite": float(torch.isfinite(after).all()),
        "pairs": float(deltas.shape[0]),
        "input_dim": float(deltas.shape[1]),
    }
    print(
        f"pruned {key}: sparsity={report['sparsity']:.3f} "
        f"changed={report['changed_entries']} peak={report['peak_gib']:.2f} GiB",
        flush=True,
    )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("results/tangent/qwen35-0.8b-concept-pairs"))
    parser.add_argument("--module", default="mlp.down_proj")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--max-gib", type=float, default=8.0)
    args = parser.parse_args()
    prune_recorded_layer(
        args.pairs,
        module_name=args.module,
        block_index=args.block,
        max_bytes=int(args.max_gib * 1024**3),
    )


if __name__ == "__main__":
    main()
