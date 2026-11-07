"""Registry and smoke contracts for the revised-note pruners."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.prune.pruners import get_pruner, list_pruners


REVISED_PRUNERS = (
    "tangent_input_side",
    "tangent_rescale",
    "tangent_heuristic",
)


def test_revised_pruners_are_registered():
    names = list_pruners()
    for name in REVISED_PRUNERS:
        assert name in names


def test_rescale_pruner_binds_and_prunes_with_unbounded_alpha():
    torch.manual_seed(0)
    layer = nn.Linear(6, 4, bias=False)
    pruner = get_pruner(
        "tangent_rescale",
        alpha=3.0,
        charts=2,
        rank=1,
        calibration_samples=24,
        seed=0,
    )
    bound = pruner.bind(layer, block_index=0, module_name="x")
    x = torch.randn(2, 12, 6)
    bound.add_batch(x, layer(x))
    bound.finalize()
    bound.prune(sparsity=0.5, blocksize=3)
    assert int((layer.weight.data == 0).sum()) > 0
    gamma = bound.solver.gamma
    assert gamma == pytest.approx(3.0 * 4 / bound.solver.tangent_diagnostics["mean_geo_rank"])


def test_heuristic_pruner_is_score_only_one_plus_gamma_g():
    torch.manual_seed(0)
    layer = nn.Linear(6, 4, bias=False)
    pruner = get_pruner(
        "tangent_heuristic",
        alpha=1.0,
        charts=2,
        rank=1,
        calibration_samples=24,
        seed=0,
    )
    bound = pruner.bind(layer, block_index=0, module_name="x")
    assert bound.solver.heuristic is True
    x = torch.randn(2, 12, 6)
    bound.add_batch(x, layer(x))
    bound.finalize()
    bound.prune(sparsity=0.25, blocksize=3)
    assert int((layer.weight.data == 0).sum()) > 0


def test_input_side_pruner_loads_pairs_and_prunes(tmp_path):
    torch.manual_seed(0)
    layer = nn.Linear(5, 3, bias=False)
    deltas = torch.randn(8, 5).numpy()
    layer_dir = tmp_path / "0"
    layer_dir.mkdir()
    np.save(layer_dir / "x.npy", deltas)
    pruner = get_pruner("tangent_input_side", alpha_in=0.5, pairs=tmp_path)
    bound = pruner.bind(layer, block_index=0, module_name="x")
    x = torch.randn(2, 10, 5)
    bound.add_batch(x, layer(x))
    bound.finalize()
    bound.prune(sparsity=0.4, blocksize=5)
    assert int((layer.weight.data == 0).sum()) > 0
    assert bound.solver.pairs_finalized is True
