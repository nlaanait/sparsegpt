"""Revised note §1–2: raw energies in losses, enrichment in comparisons.

Keep the original rank-normalized helpers for the historical note. These
tests lock the revised energy / enrichment / participation forms.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparsifier.prune.scores import (
    ambient_knockout_energy,
    enrichment,
    editability,
    editability_enrichment,
    per_dimension_energy,
    participation_ratio,
    presence_energy,
    tangent_energy,
    unconditional_energy,
)


def test_tangent_energy_is_raw_projected_square():
    delta = torch.tensor([2.0, 1.0, 0.0])
    basis = torch.eye(3, 2)
    energy = tangent_energy(delta, basis)
    assert energy == pytest.approx(4.0 + 1.0)


def test_per_dimension_energy_divides_raw_energy_by_rank():
    delta = torch.tensor([2.0, 1.0, 0.0])
    basis = torch.eye(3, 2)
    assert per_dimension_energy(delta, basis) == pytest.approx(tangent_energy(delta, basis) / 2.0)


def test_enrichment_is_one_on_average_for_a_random_plane():
    torch.manual_seed(0)
    delta = torch.randn(32)
    q, _ = torch.linalg.qr(torch.randn(32, 4), mode="reduced")
    got = enrichment(delta, q)
    expected = (32 / 4) * tangent_energy(delta, q) / ambient_knockout_energy(delta)
    assert got == pytest.approx(expected)


def test_editability_enrichment_is_rank_normalized_over_isotropic():
    jac = torch.tensor([3.0, 0.0, 4.0, 0.0])
    basis = torch.eye(4, 2)
    got = editability_enrichment(jac, basis)
    isotropic = torch.linalg.norm(jac).item() / math.sqrt(4)
    expected = editability(jac, basis) / isotropic
    assert got == pytest.approx(expected)


def test_presence_energy_uses_the_presence_projector():
    delta = torch.tensor([1.0, 2.0, 3.0])
    presence = torch.tensor([0.0, 1.0, 0.0])
    projector = torch.outer(presence, presence)
    assert presence_energy(delta, projector) == pytest.approx(4.0)


def test_participation_uses_per_dimension_energies():
    # Two concepts, ranks 1 and 4, same raw energy 4 => q = [4, 1].
    q = torch.tensor([4.0, 1.0])
    pr = participation_ratio(q)
    assert pr == pytest.approx((5.0**2) / (16.0 + 1.0))


def test_unconditional_energy_multiplies_conditional_by_presence_rate():
    conditional = 5.0
    rate = 0.2
    assert unconditional_energy(conditional, rate) == pytest.approx(1.0)
