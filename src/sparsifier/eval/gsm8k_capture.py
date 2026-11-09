"""Capture aligned task readouts and ambient inputs for the factor-atlas study.

Run only via the guarded gflow helper. This phase excludes controlled final
test and GSM8K test examples, and requires the dense competence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
from sparsifier.eval.memory_guard import configure_cuda, host_memory


def capture_one(model, layer, tokens, *, token_samples=32, seed=0):
    """Capture the same token positions at both sides of one actual forward."""
    import torch
    stored = {}
    def hook(_module, args, output):
        x, y = args[0][0], output[0]
        if x.shape[0] != tokens.shape[1] or y.shape[0] != x.shape[0]:
            raise RuntimeError("token axes do not align at readout site")
        generator = np.random.default_rng(seed)
        positions = generator.choice(len(x), min(token_samples, len(x)), replace=False)
        positions = np.unique(np.append(positions, len(x) - 1))
        ix = torch.as_tensor(positions, device=x.device)
        stored.update(x=x[-1].detach().float().cpu().numpy(),
                      y=y[-1].detach().float().cpu().numpy(),
                      token_x=x[ix].detach().float().cpu().numpy(),
                      token_y=y[ix].detach().float().cpu().numpy(),
                      positions=positions, readout_position=np.array(len(x) - 1))
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=tokens, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if not stored:
        raise RuntimeError("selected layer was not called")
    return stored


def check_competence(path: Path):
    report = json.loads(path.read_text())
    if not report.get("complete") or not report.get("competence_floor_passed"):
        raise RuntimeError("dense development pilot has not passed the competence gate")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.data.loaders import get_loaders

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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
    manifest = load_manifest(args.manifest)
    splits = ("geometry_fit", "pruning_calibration", "validation", "stage2_holdout")
    rows = []
    for split in splits:
        selected = [r for r in manifest["rows"] if r["split"] == split]
        rows.extend(selected[:args.limit_per_split] if args.limit_per_split else selected)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "manifest_sha256": manifest["manifest_sha256"],
              "pilot_report_sha256": pilot_hash, "block": args.block, "module": args.module,
              "seed": args.seed, "ids": [r["id"] for r in rows],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "c4_windows": 32, "c4_seqlen": 128, "dtype": "bfloat16",
              "purpose": "development_capture_no_final_test", "gpu_cap_gib": 56}
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
        else:
            tokens = torch.tensor([ids], device="cuda")
            arrays = capture_one(model, layer, tokens, seed=int(digest(row["id"])[:8], 16) + args.seed)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))
            temporary.replace(path)
        index.append({**metadata, "file": filename})
        if n % 20 == 0:
            print(json.dumps({"captured": n + 1, "total": len(rows), "host": host_memory()}), flush=True)
    # Ambient C4 is kept separate so every task-aware arm can match data access.
    c4_path = args.output / "c4.npz"
    if not c4_path.exists():
        calibration, _ = get_loaders("c4", 32, args.seed, 128, MODEL)
        xs, token_rows = [], []
        for tokens, _ in calibration:
            arrays = capture_one(model, layer, tokens.to("cuda"), token_samples=128, seed=args.seed)
            xs.append(arrays["token_x"])
            token_rows.append(tokens.numpy())
        with c4_path.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, inputs=np.concatenate(xs), tokens=np.concatenate(token_rows),
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
