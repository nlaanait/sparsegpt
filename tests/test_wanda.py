"""Wanda baseline pruner contracts."""

from __future__ import annotations

import torch
from torch import nn

from sparsifier.prune.pruners import get_pruner, list_pruners
from sparsifier.prune.wanda import WandaPruner, wanda_scores


def test_wanda_is_registered():
    assert "wanda" in list_pruners()


def test_wanda_scores_match_squared_activation_form():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    # E[phi^2] = [(1+0+1)/3, (0+4+1)/3] = [2/3, 5/3]
    scores = wanda_scores(weight, inputs)
    expected = weight.square() * torch.tensor([2.0 / 3.0, 5.0 / 3.0])
    torch.testing.assert_close(scores, expected)


def test_wanda_prunes_without_obs_compensation():
    torch.manual_seed(0)
    layer = nn.Linear(6, 4, bias=False)
    dense = layer.weight.data.clone()
    bound = get_pruner("wanda").bind(layer, block_index=0, module_name="x")
    x = torch.randn(2, 16, 6)
    bound.add_batch(x, layer(x))
    bound.finalize()
    bound.prune(sparsity=0.5, blocksize=3)
    zeros = layer.weight.data == 0
    assert int(zeros.sum()) == int(0.5 * layer.weight.numel())
    # No OBS: surviving weights equal their dense values.
    alive = ~zeros
    torch.testing.assert_close(layer.weight.data[alive], dense[alive])
