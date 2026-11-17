"""Bounded GB10 batching diagnostic; submit through gflow, never bare Python.

This does not prune weights or write evaluation answers. It uses the decision
runner's verified reference backend and fixed 32-token generations, rather
than changing the scientific evaluation's 2,048-token budget.
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
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(b not in (1, 2, 4, 8) for b in args.batches):
        parser.error("positive repeats and batch sizes in 1, 2, 4, 8 required")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("scheduler allocation (CUDA_VISIBLE_DEVICES) required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.data.gsm8k import load_manifest, write_immutable
    from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, render_prompt
    from sparsifier.eval.gsm8k_downstream import AnchorPrefix
    from sparsifier.eval.gsm8k_whole_neuron import reference_backend, package_versions
    from sparsifier.eval.memory_guard import configure_cuda, host_memory

    torch.set_num_threads(8)
    torch.manual_seed(0)
    manifest = load_manifest(args.manifest)
    rows = manifest["rows"][:8]
    if len(rows) != 8 or len({r["id"] for r in rows}) != 8:
        raise ValueError("eight distinct prompts required")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = reference_backend()
    config = {"model": MODEL, "revision": REVISION, "seed": 0,
              "backend": backend, "packages": package_versions(),
              "manifest_sha256": manifest["manifest_sha256"],
              "ids": [r["id"] for r in rows], "batches": args.batches,
              "repeats": args.repeats, "generation_tokens": 32,
              "purpose": "throughput_and_parity_only_not_accuracy",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_immutable(args.output/"config.json", config)
    if (args.output/"measurements.jsonl").exists():
        raise ValueError("use a new output directory for a fresh diagnostic")
    records = []

    def emit(**record):
        records.append(record)
        with (args.output/"measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)

    def measure(fn):
        fn()  # Warmup outside timed iterations.
        torch.cuda.synchronize()
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        return {"seconds": statistics.median(samples), "samples_seconds": samples,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
                "host": host_memory()}

    configure_cuda(56.)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    prompts = [render_prompt(tokenizer, r["question"]) for r in rows]
    singles = [tokenizer(p, return_tensors="pt", add_special_tokens=False).to("cuda") for p in prompts]

    def generate(inputs):
        with torch.inference_mode():
            value = model.generate(**inputs, do_sample=False, min_new_tokens=32,
                                   max_new_tokens=32, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id)
        return value[:, inputs.input_ids.shape[1]:].cpu()

    with torch.inference_mode():
        singleton_logits = torch.cat([model(**s, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
                                      for s in singles])
    singleton_tokens = torch.cat([generate(s) for s in singles])
    for batch in args.batches:
        inputs = [tokenizer(prompts[i:i+batch], padding=True, return_tensors="pt",
                            add_special_tokens=False).to("cuda") for i in range(0, 8, batch)]
        def forward_all():
            with torch.inference_mode():
                return torch.cat([model(**s, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
                                  for s in inputs])
        def generate_all():
            return torch.cat([generate(s) for s in inputs])
        try:
            logits = forward_all()
            tokens = generate_all()
            difference = (logits-singleton_logits).abs()
            parity = {"max_logit_abs_difference": difference.max().item(),
                      "rms_logit_difference": difference.square().mean().sqrt().item(),
                      "next_token_matches": int((logits.argmax(-1) == singleton_logits.argmax(-1)).sum()),
                      "exact_32_token_sequence_matches": int((tokens == singleton_tokens).all(-1).sum()),
                      "examples": 8}
            for phase, fn, units in (("prefill", forward_all, 8), ("generation", generate_all, 8*32)):
                timing = measure(fn)
                emit(phase=phase, batch=batch, **parity, **timing,
                     units_per_second=units/timing["seconds"],
                     unit="prompts" if phase == "prefill" else "generated_tokens")
        except (RuntimeError, ValueError) as error:
            emit(phase="inference", batch=batch, error=str(error))
            torch.cuda.empty_cache()

    # Real downstream graph: block-15 down output -> block-16 residual readout.
    # Fixed rank-two probe vectors for every chunk size; no atlas fitting timed.
    captured = {}
    target = model.model.layers[15].mlp.down_proj
    anchor = model.model.layers[16]
    def at_target(module, inputs, output):
        value = output.detach().requires_grad_(True)
        captured["target"] = value
        return value
    def at_anchor(module, inputs, output):
        captured["anchor"] = (output[0] if isinstance(output, tuple) else output)[:, -1]
    handles = [target.register_forward_hook(at_target), anchor.register_forward_hook(at_anchor)]
    try:
        with torch.enable_grad():
            AnchorPrefix(model, anchor)(**singles[0], use_cache=False, logits_to_keep=1)
        generator = torch.Generator(device="cpu").manual_seed(0)
        width = captured["anchor"].shape[-1]
        basis, _ = torch.linalg.qr(torch.randn(width, 2, generator=generator))
        signs = 2*torch.randint(2, (8, 2), generator=generator)-1
        probes = (signs.float() @ basis.T).unsqueeze(1).to(captured["anchor"])
        def vjps(chunk):
            pieces = []
            for start in range(0, 8, chunk):
                seed = probes[start:start+chunk]
                gradient, = torch.autograd.grad(captured["anchor"], captured["target"],
                    grad_outputs=seed if chunk > 1 else seed[0], retain_graph=True,
                    is_grads_batched=chunk > 1)
                pieces.append(gradient if chunk > 1 else gradient.unsqueeze(0))
            return torch.cat(pieces)
        reference = vjps(1)
        for chunk in (1, 2, 4, 8):
            try:
                actual = vjps(chunk)
                difference = (actual.float()-reference.float()).abs()
                max_abs = difference.max().item()
                scale = reference.float().abs().max().clamp_min(1e-8).item()
                del actual, difference
                emit(phase="backward_only", probe_chunk=chunk, probes=8,
                     max_abs_difference=max_abs, max_abs_relative_to_reference_max=max_abs/scale,
                     **measure(lambda: vjps(chunk)))
            except RuntimeError as error:
                emit(phase="backward_only", probe_chunk=chunk, error=str(error))
    except RuntimeError as error:
        emit(phase="backward_setup", error=str(error))
    finally:
        for handle in handles:
            handle.remove()
    atomic_json(args.output/"report.json", {"complete": True, "config": config, "measurements": records,
        "limitations": ["Dense checkpoint only; pruned candidates require their own validation",
            "32 forced decode steps are a diagnostic, not a 2048-token accuracy evaluation",
            "VJP timing excludes weight contractions, atlas fitting and full calibration",
            "Passing these prompts is necessary but insufficient to enable production batching",
            "CUDA allocator and system unified-memory measurements must not be added"]})


if __name__ == "__main__":
    main()
