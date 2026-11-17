"""Capture aligned task readouts and ambient inputs for the factor-atlas study.

Run only via the guarded gflow helper. This phase excludes controlled final
test and GSM8K test examples, and requires the dense competence gate.
"""
from __future__ import annotations
from sparsifier.eval.gsm8k_batching import policy as batch_policy

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.data.gsm8k_calibration import DEFAULT_MANIFEST, load_calibration, regime
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.eval.gsm8k_batching import batches, pack, policy


def capture_batch(model, layer, inputs, lengths, seeds, *, token_samples=32):
    """Capture unpadded positions and per-example seeded samples in one forward."""
    import torch
    stored = []
    def hook(_module, args, output):
        x_all, y_all = args[0], output
        if x_all.shape[:2] != inputs["input_ids"].shape or y_all.shape[:2] != x_all.shape[:2]:
            raise RuntimeError("token axes do not align at readout site")
        for row, (length, seed) in enumerate(zip(lengths, seeds)):
            x, y = x_all[row, -length:], y_all[row, -length:]
            generator = np.random.default_rng(seed)
            positions = generator.choice(len(x), min(token_samples, len(x)), replace=False)
            positions = np.unique(np.append(positions, len(x) - 1))
            ix = torch.as_tensor(positions, device=x.device)
            stored.append(dict(x=x[-1].detach().float().cpu().numpy(),
                      y=y[-1].detach().float().cpu().numpy(),
                      token_x=x[ix].detach().float().cpu().numpy(),
                      token_y=y[ix].detach().float().cpu().numpy(),
                      positions=positions, readout_position=np.array(len(x) - 1)))
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if len(stored) != len(lengths):
        raise RuntimeError("selected layer was not called")
    return stored


def capture_one(model, layer, tokens, *, token_samples=32, seed=0):
    """Single-example adapter for diagnostics and historical callers."""
    if tokens.shape[0] != 1:
        raise ValueError("capture_one expects one sequence")
    return capture_batch(model, layer, {"input_ids": tokens}, [tokens.shape[1]],
                         [seed], token_samples=token_samples)[0]


def check_competence(path: Path):
    report = json.loads(path.read_text())
    if not report.get("complete") or not report.get("competence_floor_passed"):
        raise RuntimeError("dense development pilot has not passed the competence gate")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.eval.gsm8k_batching import reference_backend as force_pytorch_attention
    from sparsifier.data.loaders import get_loaders

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=15)
    parser.add_argument("--module", default="mlp.down_proj")
    parser.add_argument("--limit-per-split", type=int, default=0)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("limit must be nonnegative")
    pilot_hash = check_competence(args.pilot_report)
    manifest = load_calibration(args.manifest)
    splits = ("geometry_fit", "pruning_calibration", "validation", "stage2_holdout")
    rows = []
    for split in splits:
        selected = [r for r in manifest["rows"] if r["split"] == split]
        rows.extend(selected[:args.limit_per_split] if args.limit_per_split else selected)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "manifest_sha256": manifest["manifest_sha256"],
              "calibration_regime": regime(manifest),
              "pilot_report_sha256": pilot_hash, "block": args.block, "module": args.module,
              "seed": args.seed, "ids": [r["id"] for r in rows],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "c4_windows": 32, "c4_seqlen": 128, "dtype": "bfloat16",
              "purpose": "development_capture_no_final_test", "gpu_cap_gib": 56}
    config["batching"] = batch_policy()
    write_immutable(args.output / "config.json", config)
    config_hash = digest(config)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    configure_cuda(56.)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        use_kernels=False, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer = model.model.layers[args.block].get_submodule(args.module)
    weight = layer.weight.detach().float().cpu().numpy()
    with (args.output / "weight.npy").open("wb") as handle:
        np.save(handle, weight)
    index = []
    pending = []
    for n, row in enumerate(rows):
        filename = digest(row["id"])[:24] + ".npz"
        path = args.output / filename
        prompt = render_prompt(tokenizer, row["question"])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > 2048:
            raise ValueError("capture prompt exceeds budget")
        metadata = {"id": row["id"], "split": row["split"], "family": row["family"],
                    "config_sha256": config_hash, "prompt_sha256": digest(prompt),
                    "input_ids": ids, "readout_position": len(ids) - 1}
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if str(old["metadata"]) != json.dumps(metadata, sort_keys=True):
                    raise ValueError("capture provenance mismatch")
        pending.append((row, ids, metadata, path))
        index.append({**metadata, "file": filename})
    for group in batches(pending):
        if all(path.exists() for _, _, _, path in group):
            continue
        sequences = [ids for _, ids, _, _ in group]
        captured = capture_batch(model, layer, pack(sequences, tokenizer, "cuda"), list(map(len, sequences)),
            [int(digest(row["id"])[:8], 16)+args.seed for row, _, _, _ in group])
        for (_, _, metadata, path), arrays in zip(group, captured):
            if path.exists():
                continue
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))
            temporary.replace(path)
        print(json.dumps({"captured_batch": len(group), "total": len(rows), "host": host_memory()}), flush=True)
    # Ambient C4 is kept separate so every task-aware arm can match data access.
    c4_path = args.output / "c4.npz"
    if not c4_path.exists():
        calibration, _ = get_loaders("c4", 32, args.seed, 128, MODEL)
        xs, token_rows = [], []
        token_rows = [tokens[0].numpy() for tokens, _ in calibration]
        for group in batches(token_rows):
            captured = capture_batch(model, layer, pack(group, tokenizer, "cuda"), list(map(len, group)),
                                     [args.seed]*len(group), token_samples=128)
            xs.extend(arrays["token_x"] for arrays in captured)
        with c4_path.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, inputs=np.concatenate(xs), tokens=np.stack(token_rows),
                                config_sha256=np.array(config_hash))
        c4_path.with_suffix(".tmp").replace(c4_path)
    else:
        with np.load(c4_path) as saved:
            if str(saved["config_sha256"]) != config_hash:
                raise ValueError("C4 capture provenance mismatch")
    atomic_json(args.output / "index.json", {"config_sha256": config_hash, "rows": index,
                                              "complete": True, "host": host_memory(),
                                              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
    print(f"Completed {len(rows)} controlled captures and 4096 C4 tokens", flush=True)


if __name__ == "__main__":
    main()
