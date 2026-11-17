"""Representative original training questions, disjoint from reserved evaluation.

Sampling is uniform without replacement, not a claim of annotated semantic
stratification. Controlled variants and the development pilot remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable

PROTOCOL = "gsm8k-original-calibration-v1"
CALIBRATION = "data/representative-calibration-seed0-v1.json"
DECISION = "data/decision64-light128-repcal-seed0-v1.json"
FULL_DECISION = "data/decision128-full512-repcal-seed0-batch8-v1.json"
DEFAULT_MANIFEST = Path("results/tangent/gsm8k")/CALIBRATION


def normalized_question(text):
    return " ".join(text.casefold().split())


def prepare_calibration(dataset, evaluation_manifests, diagnostic_manifests=(), *,
                        fit_count=512, calibration_count=128, diagnostic_count=64,
                        seed=0, source_file_sha256):
    counts = (fit_count, calibration_count, diagnostic_count)
    if any(type(n) is not int or n < 1 for n in counts) or not evaluation_manifests:
        raise ValueError("positive split counts and the complete reserved evaluation pool required")
    excluded, excluded_questions = set(), set()
    for manifest in [*evaluation_manifests, *diagnostic_manifests]:
        if manifest.get("source_split") != "train":
            raise ValueError("only original training sources may define exclusions")
        for row in manifest["rows"]:
            parent = row.get("source_index")
            if type(parent) is not int or not 0 <= parent < len(dataset):
                raise ValueError("valid original parent source_index required")
            if parent not in excluded:
                original = dataset[parent]
                if row.get("source_sha256") is not None and row["source_sha256"] != digest(original):
                    raise ValueError("exclusion manifest refers to a different training source")
                excluded_questions.add(normalized_question(original["question"]))
                excluded.add(parent)
            if "question" in row:
                excluded_questions.add(normalized_question(row["question"]))
    eligible = [n for n in range(len(dataset)) if n not in excluded]
    random.Random(seed).shuffle(eligible)
    seen, rows = set(excluded_questions), []
    for parent in eligible:
        original = dataset[parent]
        question = normalized_question(original["question"])
        if question in seen:
            continue
        answer = parse_answer(original["answer"])
        if answer is None:
            raise ValueError(f"invalid original training answer at {parent}")
        seen.add(question)
        split = ("geometry_fit" if len(rows) < fit_count else
                 "pruning_calibration" if len(rows) < fit_count+calibration_count else "validation")
        rows.append({"id": f"gsm8k/train/{parent}", "source_index": parent,
            "family": f"gsm8k/train/{parent}", "split": split, "question": original["question"],
            "answer": str(answer), "source_sha256": digest(original)})
        if len(rows) == sum(counts):
            break
    if len(rows) != sum(counts):
        raise ValueError("insufficient distinct original training questions after exclusions")
    manifest = {"schema": 1, "protocol": PROTOCOL, "dataset": "openai/gsm8k", "source_split": "train",
        "purpose": "representative_original_training_calibration_not_controlled_diagnostic",
        "seed": seed, "sampling": "uniform_without_replacement_python.Random.shuffle_on_original_indices",
        "family_unit": "original_parent_question_not_an_annotated_semantic_category",
        "fit_count": fit_count, "calibration_count": calibration_count, "diagnostic_count": diagnostic_count,
        "backward_problem_count": calibration_count, "source_file_sha256": source_file_sha256,
        "evaluation_exclusions": [m["manifest_sha256"] for m in evaluation_manifests],
        "diagnostic_exclusions": [m["manifest_sha256"] for m in diagnostic_manifests],
        "excluded_parent_indices": sorted(excluded),
        "excluded_question_sha256": sorted(digest(q) for q in excluded_questions),
        "deduplication": "original parent plus casefolded whitespace-normalized question; semantic duplicates not inferred",
        "rows": rows, "pairs": []}
    manifest["manifest_sha256"] = digest(manifest)
    validate_calibration(manifest, evaluation_manifests)
    return manifest


def validate_calibration(manifest, evaluation_manifests=()):
    if manifest.get("protocol") != PROTOCOL or manifest.get("source_split") != "train":
        raise ValueError("expected representative original training calibration")
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != digest(payload):
        raise ValueError("calibration manifest hash mismatch")
    rows = manifest["rows"]
    counts = Counter(r["split"] for r in rows)
    expected = {"geometry_fit": manifest["fit_count"], "pruning_calibration": manifest["calibration_count"],
                "validation": manifest["diagnostic_count"]}
    if counts != expected or any(type(r.get("source_index")) is not int or r["source_index"] < 0
        or r["id"] != f"gsm8k/train/{r['source_index']}" for r in rows):
        raise ValueError("original calibration identities or split counts drifted")
    if manifest.get("backward_problem_count") != manifest["calibration_count"]:
        raise ValueError("backward calibration budget drifted")
    parents = {r["source_index"] for r in rows}
    questions = {normalized_question(r["question"]) for r in rows}
    if len(parents) != len(rows) or len(questions) != len(rows):
        raise ValueError("duplicate original parents or normalized questions across calibration splits")
    if parents & set(manifest["excluded_parent_indices"]) or {digest(q) for q in questions} & set(manifest["excluded_question_sha256"]):
        raise ValueError("calibration/excluded-source overlap")
    for evaluation in evaluation_manifests:
        if evaluation["manifest_sha256"] not in manifest["evaluation_exclusions"]:
            raise ValueError("evaluation pool was not excluded during calibration preparation")
        if parents & {r["source_index"] for r in evaluation["rows"]} or questions & {
            normalized_question(r["question"]) for r in evaluation["rows"]}:
            raise ValueError("calibration/evaluation parent or normalized-question overlap")


def load_calibration(path):
    manifest = load_manifest(path)
    if manifest.get("protocol") == PROTOCOL:
        validate_calibration(manifest)
    return manifest


def regime(manifest):
    rows = manifest["rows"]
    def parents(split):
        return len({r["source_index"] for r in rows if r["split"] == split})
    return {"protocol": manifest.get("protocol", "controlled-template-diagnostic"),
            "atlas_fit_rows": sum(r["split"] == "geometry_fit" for r in rows),
            "atlas_fit_parent_count": parents("geometry_fit"),
            "pruning_calibration_rows": sum(r["split"] == "pruning_calibration" for r in rows),
            "pruning_calibration_parent_count": parents("pruning_calibration"),
            "backward_problem_count": manifest.get("backward_problem_count", 32)}


def main():
    from datasets import Dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fit-count", type=int, default=512)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--diagnostic-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-evaluation", action="store_true",
                        help="also bind the existing seeded 64/128 and 128/512 evaluation orders to this calibration")
    args = parser.parse_args()
    if args.train_arrow.name != "gsm8k-train.arrow":
        parser.error("only the cached GSM8K training Arrow source is accepted")
    evaluations = [load_manifest(p) for p in args.evaluation_manifest]
    manifest = prepare_calibration(Dataset.from_file(str(args.train_arrow)),
        evaluations, [load_manifest(p) for p in args.diagnostic_manifest],
        fit_count=args.fit_count, calibration_count=args.calibration_count, diagnostic_count=args.diagnostic_count,
        seed=args.seed, source_file_sha256=hashlib.sha256(args.train_arrow.read_bytes()).hexdigest())
    write_immutable(args.output, manifest)
    if args.prepare_evaluation:
        if len(evaluations) != 1:
            parser.error("evaluation preparation requires one complete reserved 512-question pool")
        from sparsifier.eval.gsm8k_decision import prepare_manifest as light
        from sparsifier.eval.gsm8k_decision_full import prepare_manifest as full
        write_immutable(args.output.parent/Path(DECISION).name, light(evaluations[0], manifest, seed=0))
        write_immutable(args.output.parent/Path(FULL_DECISION).name, full(evaluations[0], manifest, seed=0))
    print(f"Prepared {args.fit_count} fit / {args.calibration_count} calibration / {args.diagnostic_count} diagnostic original questions")
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
