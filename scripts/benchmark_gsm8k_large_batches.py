"""Compare equal-work batches 8/16/32 in one guarded model process.

Use gflow normally; direct execution requires explicit user authorization.
Generation length is fixed for timing, not an accuracy evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("positive repeats/tokens and explicit GPU selection required")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.data.gsm8k import load_manifest, write_immutable
    from sparsifier.eval.gsm8k import MODEL, REVISION, render_prompt, atomic_json
    from sparsifier.eval.gsm8k_batching import pack, pad_id, reference_backend
    from sparsifier.eval.gsm8k_whole_neuron import package_versions
    from sparsifier.eval.memory_guard import configure_cuda, host_memory

    torch.set_num_threads(8)
    torch.manual_seed(0)
    manifest = load_manifest(args.manifest)
    rows = manifest["rows"][:32]
    if len({r["id"] for r in rows}) != 32:
        raise ValueError("32 distinct prompts required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"measurements.jsonl").exists():
        raise ValueError("use a fresh diagnostic output")
    backend, memory = reference_backend(), configure_cuda(56.)
    config = {"model": MODEL, "revision": REVISION, "ids": [r["id"] for r in rows],
        "manifest_sha256": manifest["manifest_sha256"], "batches": [8, 16, 32], "repeats": args.repeats,
        "tokens": args.tokens, "seed": 0, "backend": backend, "packages": package_versions(),
        "memory": memory, "interleaving": "rotate_batch_order_each_round", "purpose": "throughput_not_accuracy",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_immutable(args.output/"config.json", config)
    records = []
    def emit(record):
        records.append(record)
        with (args.output/"measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    sequences = [tokenizer.encode(render_prompt(tokenizer, r["question"]), add_special_tokens=False) for r in rows]
    inputs = {b: [pack(sequences[i:i+b], tokenizer, "cuda") for i in range(0, 32, b)] for b in (8,16,32)}
    emit({"event": "loaded", "prompt_lengths": list(map(len, sequences)), "memory": memory, "host": host_memory()})
    def work(batch, phase):
        outputs = []
        with torch.inference_mode():
            for group in inputs[batch]:
                if phase == "prefill":
                    outputs.append(model(**group, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu())
                else:
                    result = model.generate(**group, do_sample=False, min_new_tokens=args.tokens,
                        max_new_tokens=args.tokens, use_cache=True, pad_token_id=pad_id(tokenizer))
                    outputs.append(result[:, group["input_ids"].shape[1]:].cpu())
        return torch.cat(outputs)
    active = []
    reference = {}
    for batch in (8,16,32):
        try:
            for phase in ("prefill", "generation"):
                value = work(batch, phase)
                if batch == 8:
                    reference[phase] = value
                if phase == "generation":
                    parity = {"exact_sequences_matching_batch8": int((value == reference[phase]).all(-1).sum())}
                else:
                    parity = {"max_logit_difference_from_batch8": float((value-reference[phase]).abs().max()),
                              "next_tokens_matching_batch8": int((value.argmax(-1)==reference[phase].argmax(-1)).sum())}
                emit({"event":"warmup", "batch":batch, "phase":phase, "examples":32, **parity})
            active.append(batch)
        except torch.cuda.OutOfMemoryError as error:
            emit({"event":"oom", "batch":batch, "error":str(error)})
            torch.cuda.empty_cache()
    for round_index in range(args.repeats):
        order = active[round_index % len(active):]+active[:round_index % len(active)] if active else []
        for batch in order:
            for phase in ("prefill", "generation"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                value = work(batch, phase)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                del value
                emit({"event":"timing", "round":round_index, "batch":batch, "phase":phase,
                    "examples":32, "seconds":elapsed,
                    "units_per_second":(32 if phase=="prefill" else 32*args.tokens)/elapsed,
                    "peak_cuda_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                    "peak_cuda_reserved_gib":torch.cuda.max_memory_reserved()/2**30, "host":host_memory()})
    summary = []
    for batch in active:
        for phase in ("prefill", "generation"):
            samples = [r for r in records if r.get("event")=="timing" and r["batch"]==batch and r["phase"]==phase]
            seconds = statistics.median(r["seconds"] for r in samples)
            summary.append({"batch":batch,"phase":phase,"median_seconds":seconds,
                "units_per_second":(32 if phase=="prefill" else 32*args.tokens)/seconds,
                "peak_cuda_allocated_gib":max(r["peak_cuda_allocated_gib"] for r in samples)})
    atomic_json(args.output/"report.json", {"complete":len(active)==3, "config":config,"summary":summary,
        "limitations":["Background job may change phases between timings", "32 decode tokens, not full reasoning",
                       "Output differences must be evaluated before changing scientific experiment defaults"]})
    print(json.dumps({"event":"summary","rows":summary}),flush=True)


if __name__ == "__main__":
    main()
