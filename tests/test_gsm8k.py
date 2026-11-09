from fractions import Fraction
import json

import pytest

from sparsifier.data.gsm8k import exact_number, parse_answer, digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import read_completed, summarize
from sparsifier.eval.memory_guard import GIB, check_budget, choose_allocator_cap


@pytest.mark.parametrize("text,expected", [
    ("Final answer: -1/2", Fraction(-1, 2)),
    ("Working 12 * 2 = 24\nFinal answer: 24.", Fraction(24)),
    ("#### 1,250.50", Fraction(2501, 2)),
    ("<think>Final answer: 10</think>\nFinal answer: 20", Fraction(20)),
    ("The answer is 12", None), ("Final answer: 12 apples", None),
    ("Final answer: 1,2", None), ("Final answer: 1/0", None),
    ("Final answer: 3\nFinal answer: 4", None),
    ("<think>Final answer: 3", None),
])
def test_strict_exact_answer(text, expected):
    assert parse_answer(text) == expected
    assert parse_answer(text, truncated=True) is None


def test_manifest_integrity_and_resume_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    value = {"rows": [{"id": "a"}]}
    value["manifest_sha256"] = digest(value)
    write_immutable(path, value)
    assert load_manifest(path) == value
    write_immutable(path, value)
    with pytest.raises(ValueError):
        write_immutable(path, {"changed": True})
    value["rows"].append({"id": "b"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)
    results = tmp_path / "answers.jsonl"
    results.write_text(json.dumps({"id": "a", "config_sha256": "one"}) + "\n")
    with pytest.raises(ValueError):
        read_completed(results, "two")


def test_memory_budget_reserves_headroom():
    check_budget(6, 56)
    check_budget(20, 80)  # unified memory: do not add CUDA reservation onto host used
    with pytest.raises(RuntimeError):
        check_budget(90, 56)
    with pytest.raises(ValueError):
        check_budget(1, 110)
    with pytest.raises(ValueError):
        check_budget(1, 56, 110)


def test_allocator_cap_honors_host_validated_local_request():
    # CUDA free alone would crush an 80 GiB local request; host available must win.
    assert choose_allocator_cap(80, int(26 * GIB), available_gib=114.0) == pytest.approx(80.0)
    assert choose_allocator_cap(56, int(26 * GIB)) == pytest.approx(23.4)
    with pytest.raises(RuntimeError):
        choose_allocator_cap(56, int(10 * GIB), available_gib=12.0)



def test_retention_gate_does_not_confuse_instances_and_families():
    rows = [{"family": "one", "correct": True, "truncated": False,
             "parsed_answer": "1", "seconds": 1, "generated_tokens": 2}] * 100
    assert not summarize(rows)["competence_floor_passed"]
