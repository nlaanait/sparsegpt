"""CPU-only audit of entrywise pruning tensors and saved GSM8K answers.

Reads one candidate and one dense tensor at a time; never loads a model.
Use CUDA_VISIBLE_DEVICES='' and the installed environment, without Pixi resolution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sparsifier.data.gsm8k import digest, parse_answer
from sparsifier.eval.gsm8k_sequential import sites_for, verify_stage
from sparsifier.prune.whole_neuron_io import file_hash


def audit_source(source, snapshot):
    config = json.loads((source / "config.json").read_text())
    report = json.loads((source / "report.json").read_text())
    config_hash = digest(config)
    assert snapshot.name == config['revision']
    assert report["complete"] and report["config_sha256"] == config_hash
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    # This audit deliberately requires complete, model-wide, three-projection runs.
    expected_sites = sites_for(list(range(32)),
        ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"], 32)
    assert [tuple(s["site"]) for s in report["stages"]] == expected_sites
    upstream, removed, stages = config_hash, 0, []
    for n, stage in enumerate(report["stages"]):
        path = source / f"site-{n:03d}.pt"
        checksum = file_hash(path)
        assert checksum == stage["candidate_sha256"]
        stored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        verify_stage(stored, config_hash, upstream, tuple(stage["site"]))
        w = stored["weight"]
        block, module = stage["site"]
        key = f"model.language_model.layers.{block}.{module}.weight"
        if key not in index:
            key = f"model.layers.{block}.{module}.weight"
        with safe_open(snapshot / index[key], framework="pt", device="cpu") as handle:
            dense = handle.get_tensor(key).float()
        assert w.shape == dense.shape
        actual_removed = int(((dense != 0) & (w == 0)).sum())
        assert actual_removed == stored["removed"] == stage["removed"]
        zero_fraction = float((w == 0).double().mean())
        assert abs(zero_fraction - config["sparsity"]) < 1e-4
        dense_norm = float(dense.norm())
        stages.append({"site": stage["site"], "sha256": checksum,
            "shape": list(w.shape), "zero_fraction": zero_fraction,
            "finite": bool(torch.isfinite(w).all()),
            "all_zero_rows": int((w == 0).all(1).sum()),
            "all_zero_columns": int((w == 0).all(0).sum()),
            "norm_over_dense": float(w.norm()) / dense_norm,
            "relative_weight_error": float((w - dense).norm()) / dense_norm,
            "max_abs_over_dense": float(w.abs().max() / dense.abs().max())})
        removed += actual_removed
        upstream = checksum
        del stored, w, dense
        if (n+1) % 24 == 0:
            print(json.dumps({"source": source.name, "stages_checked": n+1}), flush=True)
    assert removed == report["removed_weights"]
    assert sum(s["shape"][0] * s["shape"][1] for s in stages) == report["eligible_parameters"]
    ds = [s["diagnostics"] for s in report["stages"]]
    ratios = [d["added_curvature_trace_ratio"] for d in ds if "added_curvature_trace_ratio" in d]
    return {"source": str(source.resolve()), "config_sha256": config_hash,
        "report_sha256": file_hash(source / "report.json"),
        "arm": config["arm"], "target_sparsity": config["sparsity"],
        "stages": stages, "removed_weights": removed,
        "eligible_removed_fraction": removed / report["eligible_parameters"],
        "whole_model_removed_fraction": removed / report["total_parameters"],
        "geometry_trace_ratio_range": [min(ratios), max(ratios)] if ratios else None,
        "damping_attempt_counts": dict(Counter(d.get("selection_damping", {}).get(
            "factorization_attempts", "not_recorded") for d in ds)),
        "dead_input_columns_max": max((d.get("dead_input_columns", 0) for d in ds), default=0),
        "geometry_scale_max": max((d.get("scale_max", 1) for d in ds), default=1)}


def audit_answers(folder):
    config = json.loads((folder / "config.json").read_text())
    config_hash = digest(config)
    answers = [json.loads(line) for line in (folder / "answers.jsonl").read_text().splitlines()]
    assert len({a["id"] for a in answers}) == len(answers)
    if "evaluation_ids" in config:
        assert {a["id"] for a in answers} <= set(config["evaluation_ids"])
    permissive = mismatches = 0
    for a in answers:
        assert a["config_sha256"] == config_hash
        parsed = parse_answer(a["text"], truncated=a["truncated"])
        correct = parsed is not None and parsed == Fraction(a["expected_answer"])
        mismatches += correct != a["correct"]
        # Diagnostic only: keep the same marker grammar but ignore budget censoring.
        uncensored = parse_answer(a["text"], truncated=False)
        permissive += uncensored is not None and uncensored == Fraction(a["expected_answer"])
    return {"folder": str(folder.resolve()), "config_sha256": config_hash,
        "answers_sha256": file_hash(folder / "answers.jsonl"), "count": len(answers),
        "correct": sum(a["correct"] for a in answers),
        "truncated": sum(a["truncated"] for a in answers),
        "rescoring_mismatches": mismatches,
        "correct_ignoring_truncation_same_strict_parser": permissive,
        "batch_size": config.get("batching", {}).get("batch_size", 1),
        "complete": json.loads((folder / "report.json").read_text()).get("complete", False)
            if (folder / "report.json").exists() else False,
        "examples": [{"id": a["id"], "prefix": a["text"][:300],
                      "suffix": a["text"][-300:]} for a in answers[:2]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--answers", type=Path, action="append", default=[])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {"schema": 1, "purpose": "CPU_artifact_audit_not_behavioral_retention",
              "script_sha256": file_hash(Path(__file__)), "sources": [], "answers": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source in args.source:
        result["sources"].append(audit_source(source, args.snapshot))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for folder in args.answers:
        result["answers"].append(audit_answers(folder))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
