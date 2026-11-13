"""Measure loaded compact artifact prefill/decode; CUDA only through gflow."""
import argparse
import json
import os
from pathlib import Path
from sparsifier.prune.whole_neuron_io import load_compact, benchmark_compact
from sparsifier.eval.gsm8k_whole_neuron import reference_backend, package_versions


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", default="Explain how to calculate the sum of two numbers.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("CUDA runtime measurements require a scheduler GPU allocation")
    backend = reference_backend()
    model, manifest = load_compact(args.artifact, device=args.device)
    provenance = manifest["provenance"]
    tokenizer = AutoTokenizer.from_pretrained(provenance["model"], revision=provenance["revision"], local_files_only=True)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    result = benchmark_compact(model, ids, repeats=args.repeats, decode_tokens=args.decode_tokens)
    result.update(backend=backend, packages=package_versions(),
                  artifact_total_bytes=sum(p.stat().st_size for p in args.artifact.iterdir() if p.is_file()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
