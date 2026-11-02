"""Geometry calibration edges. Parity with upstream lives in test_prune_sparsegpt."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sparsifier.prune.geometry import GeometrySparseGPT


def _solver() -> tuple[GeometrySparseGPT, torch.Tensor]:
    layer = nn.Linear(4, 3, bias=False)
    solver = GeometrySparseGPT(layer)
    inputs = torch.randn(2, 5, 4)
    solver.add_batch(inputs, None)
    return solver, inputs


def test_zero_alpha_leaves_the_hessian_unchanged():
    solver, _inputs = _solver()
    before = solver.H.clone()
    solver.add_geometry(alpha=0.0)
    assert torch.equal(solver.H, before)
    assert solver.geometry_finalized


def test_geometry_is_finalized_only_once():
    solver, _inputs = _solver()
    solver.add_geometry(alpha=0.0)
    with pytest.raises(RuntimeError, match="only once"):
        solver.add_geometry(alpha=0.0)


def test_activations_after_geometry_are_rejected():
    solver, inputs = _solver()
    solver.add_geometry(alpha=0.0)
    with pytest.raises(RuntimeError, match="before adding geometry"):
        solver.add_batch(inputs, None)


def test_activation_rank_must_match_the_layer():
    layer = nn.Linear(4, 3, bias=False)
    solver = GeometrySparseGPT(layer)
    with pytest.raises(ValueError, match="nonempty"):
        solver.add_batch(torch.randn(2, 3), None)


def test_positive_alpha_requires_probes():
    solver, _inputs = _solver()
    with pytest.raises(ValueError, match="probes"):
        solver.add_geometry(alpha=0.5)


def test_nonfinite_probes_are_rejected():
    solver, _inputs = _solver()
    probes = torch.ones(4, 2)
    probes[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        solver.add_geometry(probes, alpha=0.5)
