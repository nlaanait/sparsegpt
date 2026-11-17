"""Shared seeded 64-question decision screen, then 128-question light GSM8K.

New immutable protocol; historical prefix-128/full-512 runs are unchanged.
No official GSM8K test questions. All GPU execution goes through gflow.
"""
from __future__ import annotations
from sparsifier.eval.gsm8k_batching import policy as batch_policy
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import torch

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, read_completed
from sparsifier.eval.gsm8k_pruning import behavior
from sparsifier.eval.gsm8k_batching import batches, policy
from sparsifier.eval.gsm8k_whole_neuron import (
    reference_backend, package_versions, restore_chain,
)
from sparsifier.eval.gsm8k_whole_neuron_evaluate import validate_source_separation
from sparsifier.prune.whole_neuron_io import file_hash

PROTOCOL = "gsm8k-decision64-light128-v1"


def prepare_manifest(reserved, calibration, seed=0):
    if reserved.get("source_split") != "train" or len(reserved["rows"]) != 512:
        raise ValueError("expected the reserved 512-question training-validation pool")
    rows = reserved["rows"]
    if any(r["split"] != "validation" or r["id"] != f"gsm8k/train/{r['source_index']}" for r in rows):
        raise ValueError("original GSM8K validation identities required")
    if len({r["source_index"] for r in rows}) != 512:
        raise ValueError("duplicate reserved validation source indices")
    if {r["source_index"] for r in rows} & {r["source_index"] for r in calibration["rows"]}:
        raise ValueError("reserved validation/calibration parent leakage")
    ordered = sorted(rows, key=lambda r: r["source_index"])
    random.Random(seed).shuffle(ordered)  # Dedicated RNG, never per arm.
    selected = ordered[:128]
    result = {"schema":1,"dataset":"openai/gsm8k","source_split":"train",
        "purpose":"development_decision_screen_not_final_test", "protocol":PROTOCOL,
        "parent_manifest_sha256":reserved["manifest_sha256"],
        "calibration_manifest_sha256":calibration["manifest_sha256"],
        "shuffle_seed":seed,"shuffle_algorithm":"python.Random.shuffle_on_source_index_sorted_pool",
        "rows":selected,"screen_count":64,"light_count":128,"max_new_tokens":2048,
        "screen_ids_sha256":digest([r["id"] for r in selected[:64]]),
        "light_ids_sha256":digest([r["id"] for r in selected]),
        "max_accuracy_drop_pp":15}
    result["manifest_sha256"] = digest(result)
    return result


def validate_protocol(manifest):
    ids = [r["id"] for r in manifest["rows"]]
    if (manifest.get("protocol") != PROTOCOL or manifest.get("source_split") != "train"
        or len(ids) != 128 or len(set(ids)) != 128
        or manifest["screen_ids_sha256"] != digest(ids[:64])
        or manifest["light_ids_sha256"] != digest(ids)
        or manifest.get("max_new_tokens") != 2048 or manifest.get("max_accuracy_drop_pp") != 15
        or any(r["split"] != "validation" or r["id"] != f"gsm8k/train/{r['source_index']}" for r in manifest["rows"])):
        raise ValueError("decision protocol/identity drift")


def screen_decision(manifest, baseline, answers):
    ids = [r["id"] for r in manifest["rows"][:64]]
    if not set(ids).issubset(baseline) or not set(ids).issubset(answers):
        return None
    dense = sum(bool(baseline[i]["correct"]) for i in ids)
    candidate = sum(bool(answers[i]["correct"]) for i in ids)
    return {"purpose":"exploratory_compute_screen_not_retention_claim", "screen_count":64,
        "screen_ids_sha256":manifest["screen_ids_sha256"], "dense_correct":dense,
        "candidate_correct":candidate,"max_accuracy_drop_pp":15,
        "accuracy_drop_pp":100*(dense-candidate)/64,
        "passed":Fraction(dense-candidate,64) <= Fraction(15,100)}


def run_tiers(evaluate, model, tokenizer, manifest, output, config_hash, *, baseline=None):
    """Resume fixed batches; decide on the same first 64 before any new suffix."""
    output = Path(output)
    rows = manifest["rows"]
    done = read_completed(output/"answers.jsonl", config_hash)
    if set(done)-{r["id"] for r in rows}:
        raise ValueError("unexpected evaluation IDs")
    with tempfile.TemporaryDirectory(prefix="decision-", dir=output) as folder:
        temp = Path(folder)
        for start, stop in ((0,64),(64,128)):
            if start == 64 and baseline is not None:
                decision = screen_decision(manifest, baseline, done)
                if decision is None:
                    raise ValueError("incomplete decision screen")
                write_immutable(output/"screen64.json", decision)
                if not decision["passed"]:
                    return done, decision
            for group in batches(rows[start:stop]):
                if all(row["id"] in done for row in group):
                    continue
                evaluate(model, tokenizer, group, temp, config_hash, max_new_tokens=2048)
                one = read_completed(temp/"answers.jsonl", config_hash)
                if set(one) != {row["id"] for row in group}:
                    raise ValueError("per-batch evaluator identity mismatch")
                with (output/"answers.jsonl").open("a", buffering=1) as stream:
                    for row in group:
                        if row["id"] in done:
                            continue
                        record = one[row["id"]]
                        stream.write(json.dumps(record)+"\n")
                        done[row["id"]] = record
                (temp/"answers.jsonl").unlink()
            ids = [row["id"] for row in rows[:stop]]
            write_immutable(output/f"summary-{stop}.json", {"count":stop,
                "ids_sha256":digest(ids), "correct":sum(done[i]["correct"] for i in ids),
                "config_sha256":config_hash})
    return done, screen_decision(manifest, baseline, done) if baseline is not None else None


def dense_answers(reference, manifest, backend, packages):
    config = json.loads((reference/"config.json").read_text())
    report = json.loads((reference/"report.json").read_text())
    if (not report.get("complete") or config.get("arm") != "dense"
        or config.get("manifest_sha256") != manifest["manifest_sha256"]
        or config.get("evaluation_ids") != [r["id"] for r in manifest["rows"]]
        or config.get("backend") != backend or config.get("packages") != packages
        or config.get("model") != MODEL or config.get("revision") != REVISION
        or config.get("max_new_tokens") != 2048 or config.get("batching") != policy()
        or report["config_sha256"] != digest(config)):
        raise ValueError("dense decision reference protocol/runtime mismatch")
    answers = read_completed(reference/"answers.jsonl", digest(config))
    if set(answers) != {r["id"] for r in manifest["rows"]}:
        raise ValueError("dense reference must cover exactly the shared 128 questions")
    if any(record.get("batching") != policy() for record in answers.values()):
        raise ValueError("dense answer batching provenance mismatch")
    return answers, digest(config)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.eval.gsm8k_original import frozen_stages
    from sparsifier.eval.memory_guard import configure_cuda
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    validate_protocol(manifest)
    calibration = load_manifest(args.calibration_manifest)
    if calibration["manifest_sha256"] != manifest["calibration_manifest_sha256"]:
        raise ValueError("decision calibration manifest drift")
    source = json.loads((args.source/"config.json").read_text()) if args.source else None
    if source is not None:
        validate_source_separation(source, calibration, manifest)
        if source["model"] != MODEL or source["revision"] != REVISION or args.reference is None:
            raise ValueError("candidate requires matching checkpoint and dense reference")
    backend, packages = reference_backend(), package_versions()
    baseline, baseline_hash = dense_answers(args.reference, manifest, backend, packages) if source else (None, None)
    config = {"model":MODEL,"revision":REVISION,"arm":source["arm"] if source else "dense",
        "manifest_sha256":manifest["manifest_sha256"],"evaluation_ids":[r["id"] for r in manifest["rows"]],
        "max_new_tokens":2048,"protocol":PROTOCOL,"shuffle_seed":manifest["shuffle_seed"],
        "backend":backend,"packages":packages,"dense_config_sha256":baseline_hash,
        "source_config_sha256":digest(source) if source else None,
        "source_report_sha256":file_hash(args.source/"report.json") if source else None,
        "source_sha256":file_hash(Path(__file__)),"torch_seed":0,"thinking":False}
    args.output.mkdir(parents=True, exist_ok=True)
    config["batching"] = batch_policy()
    write_immutable(args.output/"config.json", config)
    if (args.output/"report.json").exists():
        report = json.loads((args.output/"report.json").read_text())
        if report["config_sha256"] != digest(config):
            raise ValueError("decision report drift")
        if report.get("complete") or report.get("screen_rejected"):
            return
    torch.manual_seed(0)
    torch.set_num_threads(8)
    memory = configure_cuda(56.)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda").requires_grad_(False)
    removed, eligible = 0, 0
    if source:
        if source.get("method_version", "").startswith("whole-neuron-"):
            _, report, _ = restore_chain(model, args.source)
        else:
            _, report = frozen_stages(args.source)
            for n, stage in enumerate(report["stages"]):
                stored = torch.load(args.source/f"site-{n:03d}.pt", map_location="cpu", weights_only=True)
                b, module = stage["site"]
                layer = model.model.layers[b].get_submodule(module)
                if stored["weight"].shape != layer.weight.shape:
                    raise ValueError("entrywise decision candidate shape mismatch")
                with torch.no_grad():
                    layer.weight.copy_(stored["weight"])
        removed, eligible = report["removed_weights"], report["eligible_parameters"]
    done, decision = run_tiers(behavior, model, tokenizer, manifest, args.output, digest(config), baseline=baseline)
    atomic_json(args.output/"report.json", {"complete":len(done)==128, "config_sha256":digest(config),
        "screen_rejected":bool(decision and not decision["passed"]),"screen":decision,
        "behavior":{"count":len(done),"accuracy":sum(r["correct"] for r in done.values())/len(done),
            "truncated":sum(r["truncated"] for r in done.values())},"removed_weights":removed,
        "eligible_parameters":eligible,"total_parameters":sum(p.numel() for p in model.parameters()),
        "memory":memory,"scope":"64 screen / 128 light validation; no 512 or official test"})


if __name__ == "__main__":
    main()
