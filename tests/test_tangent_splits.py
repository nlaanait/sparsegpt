"""Disjoint tangent-fit / prune-cal CLI helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sparsifier.models.loop import (
    calibration_window_count,
    check_geometry,
    uses_separate_tangent_fit,
)


def test_calibration_window_count_adds_fit_windows():
    args = SimpleNamespace(nsamples=128, tangent_fit_nsamples=32)
    assert calibration_window_count(args) == 160


def test_separate_fit_requires_tangent_pruner():
    args = SimpleNamespace(
        pruner="tangent",
        tangent_alpha=0.0,
        geometry_alpha=0.0,
        tangent_fit_nsamples=8,
    )
    assert uses_separate_tangent_fit(args) is True
    args.pruner = "sparsegpt"
    assert uses_separate_tangent_fit(args) is False


def test_check_geometry_rejects_fit_split_on_wanda():
    args = SimpleNamespace(
        pruner="wanda",
        geometry_alpha=0.0,
        geometry_probes=None,
        tangent_alpha=0.0,
        tangent_charts=4,
        tangent_rank=1,
        tangent_samples=64,
        tangent_anchor_eta=0.0,
        tangent_fit_nsamples=4,
    )
    with pytest.raises(ValueError, match="tangent_fit_nsamples"):
        check_geometry(args)
