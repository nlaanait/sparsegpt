"""Revised note §3.3 / §4.4: Variant A input-side Hessian augmentation.

H' = H + γ_in H_Δ with γ_in = α_in tr(H) / tr(H_Δ). SparseGPT is unmodified
after the Hessian is rewritten. Tests lock the formulas before the solver.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sparsifier.prune.input_side import (
    InputSideSparseGPT,
    augment_hessian,
    difference_covariance,
    pair_mix_scale,
)
from sparsifier.prune.sparsegpt import SparseGPT


def test_difference_covariance_is_mean_outer_product():
    deltas = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    got = difference_covariance(deltas)
    expected = deltas.T @ deltas / deltas.shape[0]
    torch.testing.assert_close(got, expected)


def test_pair_mix_scale_matches_trace_share():
    hessian = torch.diag(torch.tensor([2.0, 4.0, 6.0]))
    h_delta = torch.diag(torch.tensor([1.0, 1.0, 1.0]))
    scale = pair_mix_scale(hessian, h_delta, alpha_in=0.5)
    assert scale == pytest.approx(0.5 * 12.0 / 3.0)


def test_augment_hessian_is_trace_normalized_mix():
    hessian = torch.eye(3) * 4.0
    deltas = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    alpha_in = 0.25
    mixed = augment_hessian(hessian, deltas, alpha_in=alpha_in)
    h_delta = difference_covariance(deltas)
    gamma_in = pair_mix_scale(hessian, h_delta, alpha_in)
    torch.testing.assert_close(mixed, hessian + gamma_in * h_delta)
    extra = mixed - hessian
    assert torch.trace(extra) == pytest.approx(alpha_in * torch.trace(hessian))


def test_zero_alpha_in_leaves_hessian_unchanged():
    hessian = torch.randn(4, 4)
    hessian = hessian @ hessian.T + torch.eye(4)
    deltas = torch.randn(6, 4)
    torch.testing.assert_close(augment_hessian(hessian, deltas, alpha_in=0.0), hessian)


def test_per_knob_normalization_equalizes_knob_energy():
    loud = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    quiet = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    deltas = torch.cat((loud, quiet), dim=0)
    knob_ids = torch.tensor([0, 0, 1, 1])
    h_delta = difference_covariance(deltas, knob_ids=knob_ids)
    loud_cov = difference_covariance(loud)
    quiet_cov = difference_covariance(quiet)
    # After per-knob trace normalization, each knob contributes equally.
    assert torch.trace(h_delta) == pytest.approx(1.0)
    loud_dir = torch.tensor([1.0, 0.0])
    quiet_dir = torch.tensor([0.0, 1.0])
    assert float(loud_dir @ h_delta @ loud_dir) == pytest.approx(
        float(quiet_dir @ h_delta @ quiet_dir)
    )
    assert torch.trace(loud_cov) != pytest.approx(torch.trace(quiet_cov))


def test_variant_a_solver_matches_manual_hessian_mix():
    torch.manual_seed(0)
    layer = nn.Linear(5, 3, bias=False)
    x = torch.randn(2, 16, 5)
    deltas = torch.randn(8, 5)
    alpha_in = 0.4

    ambient = SparseGPT(layer)
    ambient.add_batch(x, layer(x))
    expected = augment_hessian(ambient.H.clone(), deltas, alpha_in=alpha_in)

    other = nn.Linear(5, 3, bias=False)
    other.load_state_dict(layer.state_dict())
    solver = InputSideSparseGPT(other)
    solver.add_batch(x, other(x))
    solver.add_pairs(deltas)
    solver.finalize_pairs(alpha_in=alpha_in)
    torch.testing.assert_close(solver.H, expected)


def test_variant_a_changes_nm_mask_via_column_curvature():
    """Input-side H_Δ changes per-column curvature, so 2:4 masks can move."""
    torch.manual_seed(1)
    layer = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    # Ambient calibration never fires column 0, so SparseGPT treats it as dead.
    x = torch.eye(4).unsqueeze(0).repeat(8, 1, 1).clone()
    x[..., 0] = 0.0
    deltas = torch.zeros(16, 4)
    deltas[:, 0] = 3.0

    ambient_layer = nn.Linear(4, 2, bias=False)
    ambient_layer.load_state_dict(layer.state_dict())
    ambient = SparseGPT(ambient_layer)
    ambient.add_batch(x, ambient_layer(x))
    ambient.fasterprune(0.0, zeros_per_group=2, group_size=4, blocksize=4)

    geo_layer = nn.Linear(4, 2, bias=False)
    geo_layer.load_state_dict(layer.state_dict())
    geo = InputSideSparseGPT(geo_layer)
    geo.add_batch(x, geo_layer(x))
    geo.add_pairs(deltas)
    geo.finalize_pairs(alpha_in=3.0)
    geo.fasterprune(0.0, zeros_per_group=2, group_size=4, blocksize=4)

    assert int((ambient_layer.weight[:, 0] == 0).sum()) == 2
    assert int((geo_layer.weight[:, 0] != 0).sum()) == 2


def test_variant_a_rejects_nonfinite_alpha_in():
    hessian = torch.eye(2)
    deltas = torch.eye(2)
    with pytest.raises(ValueError, match="alpha"):
        augment_hessian(hessian, deltas, alpha_in=-1.0)
