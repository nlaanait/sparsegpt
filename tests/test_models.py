"""Catalog lookup. No model load."""

from __future__ import annotations

import pytest

from sparsifier.models import models, resolve, stored_count
from sparsifier.models.catalog import snapshot_billions


def test_registered_ids_are_sorted_and_have_a_stored_count():
    rows = models()
    ids = [row.model_id for row in rows]
    assert ids == sorted(ids)
    assert all(row.stored_b is not None and row.stored_b > 0 for row in rows)


def test_exact_id_returns_the_stored_count():
    record = resolve("Qwen/Qwen3.5-0.8B")
    assert record.family == "qwen"
    assert record.runner == "sparsifier"
    assert stored_count(record.model_id) == record.stored_b


def test_prefix_match_has_no_stored_count():
    record = resolve("Qwen/Qwen3.5-1B")
    assert record.family == "qwen"
    assert record.loop == "decoder"
    assert record.stored_b is None
    with pytest.raises(LookupError, match="no stored parameter count"):
        stored_count("Qwen/Qwen3.5-1B")


def test_snapshot_billions_reads_the_index_and_skips_a_missing_marker(tmp_path):
    org = tmp_path / "models--google--gemma-4-E2B"
    missing = org / "snapshots" / ".no_exist" / "model.safetensors.index.json"
    missing.parent.mkdir(parents=True)
    missing.write_text("")
    index = org / "snapshots" / "abc" / "model.safetensors.index.json"
    index.parent.mkdir()
    index.write_text('{"metadata": {"total_size": 2000000000}}')
    assert snapshot_billions("google/gemma-4-E2B", hub=tmp_path) == pytest.approx(1.0)


def test_snapshot_billions_uses_a_single_file_when_there_is_no_index(tmp_path):
    weights = tmp_path / "models--allenai--OLMo-2-0425-1B-Instruct" / "snapshots" / "abc"
    weights.mkdir(parents=True)
    path = weights / "model.safetensors"
    path.write_bytes(b"x" * 2_000_000)
    assert snapshot_billions(
        "allenai/OLMo-2-0425-1B-Instruct", hub=tmp_path
    ) == pytest.approx(0.001)


def test_gemma_prefix_is_an_own_loop():
    record = resolve("google/gemma-4-E2B-it")
    assert record.family == "gemma"
    assert record.loop == "own"
    assert record.stored_b is None
