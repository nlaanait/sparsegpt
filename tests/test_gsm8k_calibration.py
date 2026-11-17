import copy
import random
import pytest
from sparsifier.data.gsm8k import digest
from sparsifier.data.gsm8k_calibration import prepare_calibration, validate_calibration
from sparsifier.eval.gsm8k_pruning import select_validation


def dataset():
    return [{"question": f"Question {i}?", "answer": "Reasoning\n#### 3"} for i in range(100)]


def manifest(data, indices, split="validation"):
    result = {"source_split": "train", "rows": [{"id": f"gsm8k/train/{i}", "source_index": i,
        "split": split, "question": data[i]["question"], "source_sha256": digest(data[i])} for i in indices]}
    result["manifest_sha256"] = digest(result)
    return result


def prepare(data=None, seed=0):
    data = dataset() if data is None else data
    evaluation = manifest(data, [5, 7, 9])
    diagnostics = manifest(data, [0, 1], "development")
    result = prepare_calibration(data, [evaluation], [diagnostics], fit_count=20, calibration_count=10,
                                 diagnostic_count=4, seed=seed, source_file_sha256="train-source")
    return result, evaluation


def test_representative_split_is_deterministic_disjoint_original_and_rng_local():
    state = random.getstate()
    calibration, evaluation = prepare()
    assert state == random.getstate()
    assert calibration == prepare()[0] and calibration != prepare(seed=1)[0]
    validate_calibration(calibration, [evaluation])
    assert len(calibration["rows"]) == len({r["source_index"] for r in calibration["rows"]}) == 34
    assert all(r["id"] == f"gsm8k/train/{r['source_index']}" for r in calibration["rows"])
    assert not {r["source_index"] for r in calibration["rows"]} & {0, 1, 5, 7, 9}
    assert calibration["backward_problem_count"] == 10
    assert calibration["pairs"] == []


def test_exact_question_duplicates_are_excluded_across_every_boundary():
    data = dataset()
    data[22]["question"] = " QUESTION   5? "
    data[23]["question"] = data[24]["question"]
    calibration, evaluation = prepare(data)
    assert 22 not in {r["source_index"] for r in calibration["rows"]}
    questions = [" ".join(r["question"].casefold().split()) for r in calibration["rows"]]
    assert len(questions) == len(set(questions))
    validate_calibration(calibration, [evaluation])


def test_invalid_exclusions_and_insufficient_sources_fail_closed():
    data = dataset()
    evaluation = manifest(data, [5, 7, 9])
    with pytest.raises(ValueError, match="complete reserved"):
        prepare_calibration(data, [], fit_count=1, calibration_count=1, diagnostic_count=1, source_file_sha256="x")
    changed = copy.deepcopy(data)
    changed[5]["answer"] = "#### 4"
    with pytest.raises(ValueError, match="different training source"):
        prepare_calibration(changed, [evaluation], source_file_sha256="x")
    with pytest.raises(ValueError, match="insufficient"):
        prepare_calibration(data, [evaluation], fit_count=100, source_file_sha256="x")
    bad = copy.deepcopy(evaluation)
    bad["source_split"] = "test"
    with pytest.raises(ValueError, match="training sources"):
        prepare_calibration(data, [bad], source_file_sha256="x")


def test_manifest_tampering_and_parent_leakage_are_rejected():
    calibration, evaluation = prepare()
    broken = copy.deepcopy(calibration)
    broken["rows"][0]["source_index"] = 5
    broken["rows"][0]["id"] = "gsm8k/train/5"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_calibration(broken, [evaluation])
    del broken["manifest_sha256"]
    broken["manifest_sha256"] = digest(broken)
    with pytest.raises(ValueError, match="overlap"):
        validate_calibration(broken, [evaluation])


def test_original_selection_is_seeded_not_sorted_unique_family_prefix():
    data = dataset()
    original = manifest(data, list(range(40)))["rows"]
    for row in original:
        row["family"] = row["id"]
    first = select_validation(original, 12, 0)
    assert first == select_validation(list(reversed(original)), 12, 0)
    assert first != select_validation(original, 12, 1)
    assert {r["source_index"] for r in first} != set(range(12))
    assert select_validation(original, 0, 0) == []


def test_prepared_original_protocol_keeps_existing_evaluation_order_and_binds_all_plans():
    from pathlib import Path
    from sparsifier.data.gsm8k import load_manifest
    from sparsifier.data.gsm8k_calibration import CALIBRATION, DECISION, FULL_DECISION, load_calibration
    from scripts.submit_gsm8k_decision_pilot import build_representative_plan
    from scripts import submit_gsm8k_whole_neuron_full as full
    root = Path(__file__).resolve().parents[1]/"results/tangent/gsm8k"
    calibration = load_calibration(root/CALIBRATION)
    reserved = load_manifest(root/"data/original-validation-v1.json")
    validate_calibration(calibration, [reserved])
    assert calibration["fit_count"] == 512 and calibration["calibration_count"] == 128
    assert len({r["source_index"] for r in calibration["rows"]}) == 704
    light, complete = load_manifest(root/DECISION), load_manifest(root/FULL_DECISION)
    old_light = load_manifest(root/"data/decision64-light128-seed0-v1.json")
    old_full = load_manifest(root/"data/decision128-full512-seed0-batch8-v1.json")
    assert light["rows"] == old_light["rows"] and complete["rows"] == old_full["rows"]
    assert light["screen_ids_sha256"] == old_light["screen_ids_sha256"]
    assert complete["screen_ids_sha256"] == old_full["screen_ids_sha256"]
    assert complete["reference_manifest_sha256"] == light["manifest_sha256"]
    plan = build_representative_plan(light, calibration)
    assert len(plan["tasks"]) == 13
    assert plan["probe_count"] == 128 and plan["calibration_manifest"] == CALIBRATION
    assert plan["calibration_regime"]["atlas_fit_parent_count"] == 512
    assert plan["batching"]["batch_size"] == 32
    assert all("batch32" in task["name"] for task in plan["tasks"])
    assert all("repcal-v1" in task["name"] for task in plan["tasks"])
    assert full.REFERENCE.name == plan["dense"]
    assert full.MANIFEST == root/FULL_DECISION
    assert all("repcal-v1" in name for name, _, _ in full.tasks())
    assert all("batch32" in name for name, _, _ in full.tasks())
    assert "batch32" in full.JOURNAL.name
