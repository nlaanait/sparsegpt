"""BlockDictionary factor state for tangent_factor_union (note-faithful path)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.prune.factor_union import (
    FactorUnionSparseGPT,
    estimate_factor_union_coupling,
    union_projector_diagonal,
)
from sparsifier.prune.geometry_features import (
    BlockDictionary,
    block_dictionary_factor_state,
)
from sparsifier.prune.pruners import get_pruner


def test_block_dictionary_active_mask_matches_encode_support():
    rng = np.random.default_rng(0)
    d, n, groups, width, active = 24, 60, 6, 3, 2
    h = rng.normal(size=(d, n))
    model = BlockDictionary(groups, width, active, seed=0).fit(h, iterations=5)
    bases, active_mask = model.factor_state(h, neighbors=12, rank=1, seed=1)
    assert bases.shape == (d, groups, 1)
    assert active_mask.shape == (groups, n)
    expected = model.encode_support(h)
    np.testing.assert_array_equal(active_mask, expected)
    # Exactly ``active`` blocks per sample after hard thresholding.
    assert int(active_mask.sum(axis=0).min()) == active
    assert int(active_mask.sum(axis=0).max()) == active


def test_block_dictionary_overlapping_factors_union_rank_below_sum():
    """Coactive factors with parallel ambient charts QR to rank < sum r_i."""
    # Two identical rank-1 ambient bases labeled as distinct factors.
    bases = torch.zeros(2, 4, 1)
    bases[0, 0, 0] = 1.0
    bases[1, 0, 0] = 1.0
    diag, rank = union_projector_diagonal([bases[0], bases[1]])
    assert rank == 1
    torch.testing.assert_close(diag, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    n, din = 16, 3
    x = torch.randn(n, din).abs() + 0.1
    y = torch.randn(n, 4)
    active = torch.ones(n, 2, dtype=torch.bool)
    score, diagnostics = estimate_factor_union_coupling(
        x, y, factor_bases=bases, active=active, alpha=1.0
    )
    assert diagnostics["mean_union_rank"] == pytest.approx(1.0)
    assert score.shape == (4, din)


def test_block_dictionary_factor_state_helper_shapes_and_support():
    torch.manual_seed(0)
    n, d = 48, 16
    outputs = torch.randn(n, d)
    bases, active = block_dictionary_factor_state(
        outputs,
        groups=4,
        width=2,
        active=2,
        rank=1,
        seed=0,
        fit_iterations=4,
    )
    assert bases.shape == (4, d, 1)
    assert active.shape == (n, 4)
    assert bool(active.any())
    # Each sample activates exactly two factors.
    assert torch.equal(active.sum(dim=1), torch.full((n,), 2))


def test_factor_union_block_dictionary_binds_and_prunes_tiny_linear():
    torch.manual_seed(1)
    layer = nn.Linear(6, 8, bias=True)
    pruner = get_pruner(
        "tangent_factor_union",
        alpha=1.0,
        factors=4,
        rank=1,
        calibration_samples=32,
        seed=0,
        active_factors=2,
        factor_source="block_dictionary",
        block_width=2,
    )
    bound = pruner.bind(layer, block_index=0, module_name="x")
    x = torch.randn(2, 16, 6)
    y = layer(x)
    bound.add_batch(x, y)
    bound.finalize()
    assert bound.solver.tangent_diagnostics["factor_source"] == "block_dictionary"
    before = layer.weight.data.clone()
    bound.prune(sparsity=0.25, blocksize=3)
    assert int((layer.weight.data == 0).sum()) > 0
    assert not torch.equal(layer.weight.data, before)
    bound.free()


def test_factor_union_kmeans_path_still_works():
    torch.manual_seed(2)
    layer = nn.Linear(6, 4, bias=False)
    solver = FactorUnionSparseGPT(
        layer,
        alpha=1.0,
        factors=2,
        rank=1,
        calibration_samples=24,
        seed=0,
        active_factors=1,
        factor_source="kmeans",
    )
    x = torch.randn(2, 12, 6)
    solver.add_batch(x, layer(x))
    solver.fasterprune(0.25, blocksize=3)
    assert solver.tangent_diagnostics["factor_source"] == "kmeans"
    assert int((layer.weight.data == 0).sum()) > 0
    solver.free()
