"""Revised note §4.5: Variant B rescaling trick (replaces Option 1).

scale = sqrt(1 + γ G); SparseGPT on W ⊙ scale; unscale. Do not multiply
scores by raw G. A score-only 1+γG path is a labeled heuristic ablation.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sparsifier.prune.regularizer import geometric_weight
from sparsifier.prune.rescale import (
    heuristic_score_weight,
    prune_with_rescaling,
    row_scales,
)
from sparsifier.prune.sparsegpt import SparseGPT


def test_row_scales_are_sqrt_one_plus_gamma_g_when_hessians_match():
    coupling = torch.tensor([[0.0, 1.0], [0.5, 0.25]])
    gamma = 4.0
    scale = row_scales(coupling, gamma)
    torch.testing.assert_close(scale, torch.sqrt(1.0 + gamma * coupling))


def test_row_scales_use_h0_over_hprime_when_combined_with_variant_a():
    coupling = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    h0 = torch.tensor([2.0, 4.0])
    hprime = torch.tensor([4.0, 4.0])
    gamma = 2.0
    scale = row_scales(coupling, gamma, h0_diag=h0, hprime_diag=hprime)
    adjusted = coupling * (h0 / hprime)[None, :]
    torch.testing.assert_close(scale, torch.sqrt(1.0 + gamma * adjusted))


def test_row_scales_are_one_on_features_with_no_support():
    coupling = torch.tensor([[0.3, 0.9], [0.1, 0.2]])
    support = torch.tensor([True, False])
    scale = row_scales(coupling, gamma=5.0, support=support)
    torch.testing.assert_close(scale[:, 1], torch.ones(2))
    assert float(scale[0, 0]) == pytest.approx((1.0 + 5.0 * 0.3) ** 0.5)


def test_heuristic_score_weight_is_one_plus_gamma_g_not_raw_g():
    coupling = torch.tensor([[0.2, 1.0], [0.0, 0.5]])
    weight = heuristic_score_weight(coupling, gamma=3.0)
    torch.testing.assert_close(weight, 1.0 + 3.0 * coupling)
    assert not torch.allclose(weight, coupling)


def test_rescaling_changes_compensation_when_scale_varies_within_a_row():
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]))
    x = torch.randn(3, 32, 4)
    coupling = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]], dtype=torch.float32
    )
    gamma = 4.0
    scale = row_scales(coupling, gamma)
    sparsity = 0.5

    rescaled = nn.Linear(4, 2, bias=False)
    rescaled.load_state_dict(layer.state_dict())
    solver = SparseGPT(rescaled)
    solver.add_batch(x, rescaled(x))
    prune_with_rescaling(solver, scale, sparsity=sparsity, blocksize=4)

    heuristic = nn.Linear(4, 2, bias=False)
    heuristic.load_state_dict(layer.state_dict())
    other = SparseGPT(heuristic)
    other.add_batch(x, heuristic(x))
    other.fasterprune(
        sparsity, blocksize=4, mask_weight=heuristic_score_weight(coupling, gamma)
    )

    # Same support (positive diagonal rescaling), different compensation.
    torch.testing.assert_close(
        (rescaled.weight == 0).to(torch.int),
        (heuristic.weight == 0).to(torch.int),
    )
    assert not torch.allclose(rescaled.weight, heuristic.weight)


def test_static_row_weight_is_a_noop_under_nm():
    """A static output metric is a row-only factor; 2:4 masks ignore it."""
    torch.manual_seed(2)
    dense = torch.randn(3, 8)
    x = torch.randn(4, 20, 8)
    row_factor = torch.tensor([0.2, 1.7, 4.0])[:, None]
    coupling = row_factor.expand_as(dense)
    gamma = 1.0
    scale = row_scales(coupling, gamma)

    ambient = nn.Linear(8, 3, bias=False)
    with torch.no_grad():
        ambient.weight.copy_(dense)
    a_solver = SparseGPT(ambient)
    a_solver.add_batch(x, ambient(x))
    a_solver.fasterprune(0.0, zeros_per_group=2, group_size=4, blocksize=4)

    scaled = nn.Linear(8, 3, bias=False)
    with torch.no_grad():
        scaled.weight.copy_(dense)
    s_solver = SparseGPT(scaled)
    s_solver.add_batch(x, scaled(x))
    prune_with_rescaling(
        s_solver, scale, sparsity=0.0, zeros_per_group=2, group_size=4, blocksize=4
    )

    torch.testing.assert_close(
        (ambient.weight == 0).to(torch.int),
        (scaled.weight == 0).to(torch.int),
    )


def test_geometric_weight_feeds_rescaling_not_convex_alpha():
    gamma = geometric_weight(alpha=3.0, ambient_dim=8, mean_geo_rank=2.0)
    coupling = torch.ones(2, 3) * 0.5
    scale = row_scales(coupling, gamma)
    torch.testing.assert_close(scale, torch.full((2, 3), math.sqrt(1.0 + 12.0 * 0.5)))
