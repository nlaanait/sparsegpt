"""Parity: sparsifier.prune solvers match the original sparsegpt package."""

from __future__ import annotations

import contextlib
import io

import pytest
import torch
from torch import nn

from sparsegpt import SparseGPT as RefSparseGPT
from sparsegpt.geometry import GeometrySparseGPT as RefGeometry
from sparsifier.prune.geometry import GeometrySparseGPT
from sparsifier.prune.sparsegpt import SparseGPT


def _pair_layers(in_features: int = 32, out_features: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    ref = nn.Linear(in_features, out_features, bias=False)
    new = nn.Linear(in_features, out_features, bias=False)
    new.weight.data.copy_(ref.weight.data)
    x = torch.randn(4, 8, in_features)
    return ref, new, x


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def test_add_batch_matches_reference_hessian():
    ref_layer, new_layer, x = _pair_layers()
    ref = RefSparseGPT(ref_layer)
    new = SparseGPT(new_layer)
    ref.add_batch(x, None)
    new.add_batch(x, None)
    assert new.nsamples == ref.nsamples
    assert torch.allclose(new.H, ref.H)


@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("blocksize", [8, 16])
def test_unstructured_fasterprune_matches_reference(sparsity: float, blocksize: int):
    ref_layer, new_layer, x = _pair_layers()
    ref = RefSparseGPT(ref_layer)
    new = SparseGPT(new_layer)
    ref.add_batch(x, None)
    new.add_batch(x, None)
    with _quiet():
        ref.fasterprune(sparsity, blocksize=blocksize, percdamp=0.01)
        new.fasterprune(sparsity, blocksize=blocksize, percdamp=0.01)
    assert torch.allclose(new_layer.weight.data, ref_layer.weight.data)


def test_nm_fasterprune_matches_reference():
    ref_layer, new_layer, x = _pair_layers(in_features=32, out_features=16)
    ref = RefSparseGPT(ref_layer)
    new = SparseGPT(new_layer)
    ref.add_batch(x, None)
    new.add_batch(x, None)
    with _quiet():
        ref.fasterprune(0.0, prunen=2, prunem=4, blocksize=16, percdamp=0.01)
        new.fasterprune(0.0, zeros_per_group=2, group_size=4, blocksize=16, percdamp=0.01)
    assert torch.allclose(new_layer.weight.data, ref_layer.weight.data)


def test_geometry_hessian_and_prune_match_reference():
    ref_layer, new_layer, x = _pair_layers()
    torch.manual_seed(1)
    probes = torch.randn(x.shape[-1], 6)

    ref = RefGeometry(ref_layer)
    new = GeometrySparseGPT(new_layer)
    ref.add_batch(x, None)
    new.add_batch(x, None)
    ref.add_geometry(probes.numpy(), alpha=0.5)
    new.add_geometry(probes.numpy(), alpha=0.5)
    assert torch.allclose(new.H, ref.H)

    with _quiet():
        ref.fasterprune(0.25, blocksize=8, percdamp=0.01)
        new.fasterprune(0.25, blocksize=8, percdamp=0.01)
    assert torch.allclose(new_layer.weight.data, ref_layer.weight.data)
