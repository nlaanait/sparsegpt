"""Evaluate completed triplet artifacts on original GSM8K training validation.

Same behavior/language helpers as entrywise original validation; only the
triplet checkpoint loader differs. Run exclusively through guarded gflow.
"""
from sparsifier.eval.gsm8k_batching import policy as batch_policy
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


def validate_source_separation(source_config, calibration_manifest, validation_manifest):
    """Bind the source manifest and compare original GSM8K parent indices.

    Controlled variant IDs differ from original IDs; their source_index is
    the original training parent. Missing parent provenance fails closed.
    Call only with manifests verified by load_manifest.
    """
    if calibration_manifest["manifest_sha256"] != source_config["manifest_sha256"]:
        raise ValueError("source calibration manifest hash mismatch")
    by_id = {row["id"]: row for row in calibration_manifest["rows"]}
    used = source_config["calibration_ids"]
    if not used or any(identifier not in by_id for identifier in used):
        raise ValueError("source calibration IDs missing from bound manifest")
    def parent(row):
        index = row.get("source_index")
        if type(index) is not int or index < 0:
            raise ValueError("original parent source_index required for separation")
        if row["id"].startswith("gsm8k/train/") and row["id"] != f"gsm8k/train/{index}":
            raise ValueError("original GSM8K ID and source_index disagree")
        return index
    sources = [by_id[identifier] for identifier in used]
    if any(row["split"] not in ("geometry_fit", "pruning_calibration") for row in sources):
        raise ValueError("source calibration split mismatch")
    calibration_indices = {parent(row) for row in sources}
    evaluation_indices = {parent(row) for row in validation_manifest["rows"]}
    if calibration_indices & evaluation_indices:
        raise ValueError("original-validation/calibration parent-source overlap")
    return {"calibration_manifest_sha256": calibration_manifest["manifest_sha256"],
            "calibration_parent_count": len(calibration_indices),
            "validation_parent_count": len(evaluation_indices), "disjoint_parent_sources": True}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.data.loaders import get_loaders
    from sparsifier.eval.natural_layer import vocabulary_reference, language_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True,
                        help="source pruning manifest; its stored hash and parent sources are verified")
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
    separation = validate_source_separation(source_config, load_manifest(args.calibration_manifest), manifest)
    config = {"model": MODEL, "revision": REVISION, "arm": source_config["arm"],
        "method_version": source_config["method_version"], "manifest_sha256": manifest["manifest_sha256"],
        "source_config_sha256": file_hash(args.source/"config.json"),
        "source_report_sha256": file_hash(args.source/"report.json"),
        "source_separation": separation,
        "evaluation_ids": [row["id"] for row in rows], "max_new_tokens": args.max_new_tokens,
        "thinking": False, "backend": reference_backend(), "packages": package_versions(),
        "source_sha256": file_hash(Path(__file__)), "purpose": "original_GSM8K_validation_not_test"}
    args.output.mkdir(parents=True, exist_ok=True)
    config["batching"] = batch_policy()
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
