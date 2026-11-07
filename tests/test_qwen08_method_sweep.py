"""Argv and resume rules for the Qwen 3.5-0.8B method sweep."""

from __future__ import annotations

from drivers.qwen08_method_sweep import (
    METHODS,
    SPARSITIES,
    pending_points,
    point_command,
)


def test_sparsity_ladder_starts_at_half_and_ends_at_99():
    assert SPARSITIES[0] == 0.5
    assert SPARSITIES[-1] == 0.99
    assert 0.9 in SPARSITIES


def test_sparsegpt_is_the_baseline_and_fisher_is_absent():
    assert METHODS[0] == "sparsegpt"
    assert "tangent_fisher" not in METHODS
    assert "tangent_input_side" in METHODS


def test_point_command_carries_the_method_flags():
    baseline = point_command("sparsegpt", 0.5, python="python")
    assert baseline[baseline.index("--sparsity") + 1] == "0.5"
    assert "--pruner" not in baseline
    assert baseline[baseline.index("--forward-batch") + 1] == "1"
    assert "drivers.capped_sparsifier" in baseline
    variant = point_command("tangent_input_side", 0.99, python="python")
    assert variant[variant.index("--sparsity") + 1] == "0.99"
    assert variant[variant.index("--pruner") + 1] == "tangent_input_side"
    assert "--alpha-in" in variant
    assert "--pair-deltas" in variant


def test_pending_points_skip_saved_perplexities():
    saved = {
        "points": [
            {"dataset": "wikitext2", "method": "sparsegpt", "sparsity": 0.5, "perplexity": 12.0}
        ]
    }
    missing = pending_points(saved)
    assert ("sparsegpt", 0.5) not in missing
    assert ("sparsegpt", 0.6) in missing
    assert missing[0] == ("sparsegpt", 0.6)
