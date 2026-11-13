"""Evaluate completed triplet artifacts on original GSM8K training validation.

Same behavior/language helpers as entrywise original validation; only the
triplet checkpoint loader differs. Run exclusively through guarded gflow.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json
from sparsifier.eval.gsm8k_pruning import behavior
from sparsifier.eval.gsm8k_whole_neuron import restore_chain, reference_backend, package_versions
from sparsifier.prune.whole_neuron_io import file_hash
from sparsifier.eval.memory_guard import configure_cuda


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.data.loaders import get_loaders
    from sparsifier.eval.natural_layer import vocabulary_reference, language_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    rows = manifest["rows"]
    if (args.max_new_tokens < 1 or manifest["source_split"] != "train" or not rows
            or any(row["split"] != "validation" or not row["id"].startswith("gsm8k/train/") for row in rows)):
        parser.error("original training-validation questions and positive generation budget required")
    source_config = json.loads((args.source/"config.json").read_text())
    if source_config["model"] != MODEL or source_config["revision"] != REVISION:
        raise ValueError("whole-neuron evaluation base mismatch")
    config = {"model": MODEL, "revision": REVISION, "arm": source_config["arm"],
        "method_version": source_config["method_version"], "manifest_sha256": manifest["manifest_sha256"],
        "source_config_sha256": file_hash(args.source/"config.json"),
        "source_report_sha256": file_hash(args.source/"report.json"),
        "evaluation_ids": [row["id"] for row in rows], "max_new_tokens": args.max_new_tokens,
        "thinking": False, "backend": reference_backend(), "packages": package_versions(),
        "source_sha256": file_hash(Path(__file__)), "purpose": "original_GSM8K_validation_not_test"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/"config.json", config)
    path = args.output/"report.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["config_sha256"] != digest(config):
            raise ValueError("evaluation report drift")
        if old.get("complete"):
            return
    memory = configure_cuda(56.)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    _, wiki = get_loaders("wikitext2", 1, 0, 128, MODEL)
    windows = [wiki.input_ids[:, i*128:(i+1)*128] for i in range(8)]
    reference = vocabulary_reference(model, windows, "cuda")
    if not np.isfinite([r["nll"] for r in reference]).all() or np.mean([r["nll"] for r in reference]) > np.log(100):
        raise ValueError("implausible dense language loss")
    _, report, _ = restore_chain(model, args.source)
    language = language_metrics(model, windows, reference, "cuda")
    evaluated = behavior(model, tokenizer, rows, args.output, digest(config), max_new_tokens=args.max_new_tokens)
    atomic_json(path, {"complete": True, "config_sha256": digest(config), "behavior": evaluated,
        "language": language, "removed_weights": report["removed_weights"],
        "architectural_removed_slots": report["architectural_removed_slots"],
        "total_parameters": report["total_parameters"], "eligible_parameters": report["eligible_parameters"],
        "whole_model_removed_fraction": report["whole_model_removed_fraction"], "memory": memory})


if __name__ == "__main__":
    main()
