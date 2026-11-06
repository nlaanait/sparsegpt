"""CLI / loop selection of ablation pruners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sparsifier.models.loop import pruner_for, resolve_pruner_name, uses_output_tangents
from sparsifier.prune.pruners import list_pruners


def test_list_pruners_is_stable_ablation_set():
    names = list_pruners()
    assert names == [
        "sparsegpt",
        "geometry",
        "tangent",
        "tangent_factor_union",
        "tangent_chart_obs",
        "tangent_fixed_union",
        "tangent_knockout",
        "tangent_editability",
    ]


def test_explicit_pruner_overrides_alphas():
    args = SimpleNamespace(
        pruner="tangent_knockout",
        tangent_alpha=0.0,
        geometry_alpha=0.0,
        tangent_charts=4,
        tangent_rank=1,
        tangent_samples=32,
        seed=0,
        bias_fold=False,
        percdamp=0.01,
    )
    assert resolve_pruner_name(args) == "tangent_knockout"
    assert uses_output_tangents(args)
    assert pruner_for(args).name == "tangent_knockout"


def test_tangent_editability_selected_via_loop():
    args = SimpleNamespace(
        pruner="tangent_editability",
        tangent_alpha=0.0,
        geometry_alpha=0.0,
        tangent_charts=4,
        tangent_rank=1,
        tangent_samples=32,
        seed=0,
        bias_fold=True,
        percdamp=0.01,
    )
    assert resolve_pruner_name(args) == "tangent_editability"
    assert uses_output_tangents(args)
    pruner = pruner_for(args)
    assert pruner.name == "tangent_editability"
    assert pruner.bias_fold is True


def test_legacy_tangent_alpha_selects_option1():
    args = SimpleNamespace(
        pruner=None,
        tangent_alpha=0.5,
        geometry_alpha=0.0,
        tangent_charts=4,
        tangent_rank=1,
        tangent_samples=32,
        seed=0,
        bias_fold=False,
    )
    assert resolve_pruner_name(args) == "tangent"
    assert pruner_for(args).alpha == 0.5


def test_unknown_pruner_is_rejected():
    args = SimpleNamespace(pruner="nope", tangent_alpha=0.0, geometry_alpha=0.0)
    with pytest.raises(ValueError, match="unknown pruner"):
        resolve_pruner_name(args)
