"""Resumable behavioral evaluation for the GSM8K pruning experiment.

All GPU invocations must be scheduled through the guarded gflow helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable
from sparsifier.eval.memory_guard import configure_cuda, host_memory

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
INSTRUCTION = "Solve the following problem. You may show your working. End with a line containing only 'Final answer: <number>', with no units after the number.\n\n"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_completed(path: Path, config_hash: str) -> dict:
    result = {}
    if not path.exists():
        return result
    # A partial trailing write is not silently accepted or discarded.
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["config_sha256"] != config_hash or row["id"] in result:
            raise ValueError("result provenance mismatch or duplicate ID")
        result[row["id"]] = row
    return result


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "complete": False}
    correct = [row for row in rows if row["correct"]]
    return {"count": len(rows), "correct": len(correct),
            "accuracy": len(correct) / len(rows),
            "dense_correct_families": len({r["family"] for r in correct}),
            "truncated": sum(r["truncated"] for r in rows),
            "parse_failures": sum(r["parsed_answer"] is None for r in rows),
            "generation_seconds": sum(r["seconds"] for r in rows),
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "competence_floor_passed": len(correct) >= 100 and len({r["family"] for r in correct}) >= 10}


def render_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention

    manifest = load_manifest(args.manifest)
    rows = manifest["rows"][:args.limit] if args.limit else manifest["rows"]
    if not rows:
        raise ValueError("empty evaluation")
    if manifest["source_split"] != "train" or any(r["split"] != "development" for r in rows):
        raise ValueError("dense pilot accepts development training rows only")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (Path(__file__), Path(__file__).with_name("memory_guard.py"),
                         Path(__file__).parents[1] / "data/gsm8k.py")}
    config = {"schema": 1, "model": MODEL, "revision": REVISION,
              "manifest_sha256": manifest["manifest_sha256"], "ids": [r["id"] for r in rows],
              "max_new_tokens": args.max_new_tokens, "do_sample": False,
              "enable_thinking": False, "instruction": INSTRUCTION,
              "dtype": "bfloat16", "attention": "sdpa_with_pytorch_delta",
              "gpu_cap_gib": args.gpu_cap_gib, "seed": 0, "sources": sources,
              "torch": torch.__version__}
    import transformers
    config["transformers"] = transformers.__version__
    write_immutable(args.output / "config.json", config)
    config_hash = digest(config)
    completed = read_completed(args.output / "answers.jsonl", config_hash)
    if set(completed) - set(config["ids"]):
        raise ValueError("unexpected result IDs")
    if len(completed) == len(rows):
        print("Already complete", flush=True)
        return
    torch.set_num_threads(8)
    torch.manual_seed(0)
    memory = configure_cuda(args.gpu_cap_gib)
    print(json.dumps({"event": "before_load", **memory}), flush=True)
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        trust_remote_code=True, use_kernels=False, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    model.to("cuda")
    print(json.dumps({"event": "loaded", "seconds": time.monotonic() - started,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "host": host_memory(), "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3}), flush=True)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with (args.output / "answers.jsonl").open("a", buffering=1) as out:
        for row in rows:
            if row["id"] in completed:
                continue
            prompt = render_prompt(tokenizer, row["question"])
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            if inputs.input_ids.shape[1] > 2048:
                raise ValueError("pilot prompt exceeds 2048-token budget")
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                        use_cache=True, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            generated = output[0, inputs.input_ids.shape[1]:].tolist()
            truncated = len(generated) >= args.max_new_tokens and generated[-1] not in eos_ids
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_answer(decoded, truncated=truncated)
            from fractions import Fraction
            record = {"id": row["id"], "family": row["family"], "config_sha256": config_hash,
                      "source_sha256": row["source_sha256"], "prompt_sha256": digest(prompt),
                      "expected_answer": row["answer"], "parsed_answer": str(parsed) if parsed is not None else None,
                      "correct": parsed is not None and parsed == Fraction(row["answer"]),
                      "text": decoded, "truncated": truncated, "generated_tokens": len(generated),
                      "input_tokens": inputs.input_ids.shape[1], "seconds": time.monotonic() - started,
                      "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                      "host": host_memory()}
            out.write(json.dumps(record) + "\n")
            out.flush()
            os.fsync(out.fileno())
            completed[row["id"]] = record
            report = {**summarize(list(completed.values())), "config_sha256": config_hash,
                      "complete": len(completed) == len(rows), "memory": memory}
            atomic_json(args.output / "report.json", report)
            print(json.dumps({"event": "answer", "id": row["id"], "correct": record["correct"],
                              "tokens": len(generated), "seconds": record["seconds"],
                              "completed": len(completed), "accuracy": report["accuracy"]}), flush=True)
            del output, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-cap-gib", type=float, default=56.)
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1:
        parser.error("limit must be nonnegative and token budget positive")
    run(args)


if __name__ == "__main__":
    main()
