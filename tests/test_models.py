"""Catalog lookup. No model load."""

from __future__ import annotations

import pytest

from sparsifier.models import models, resolve, stored_count


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


def test_gemma_prefix_is_an_own_loop():
    record = resolve("google/gemma-4-E2B-it")
    assert record.family == "gemma"
    assert record.loop == "own"
    assert record.stored_b is None
