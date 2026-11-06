"""η P_anchor metric helpers (note §3)."""

from __future__ import annotations

import torch

from sparsifier.prune.metric import (
    anchor_projector,
    blend_anchor_into_diagonals,
    coupling_from_diagonals,
    union_plus_anchor_metric,
)
from sparsifier.prune.tangent import estimate_output_tangent_coupling


def test_anchor_projector_is_rank1_idempotent():
    v = torch.tensor([3.0, 0.0, 4.0])
    p = anchor_projector(v)
    torch.testing.assert_close(p @ p, p, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.trace(p), torch.tensor(1.0), atol=1e-6, rtol=1e-5)


def test_union_plus_anchor_scales_with_eta():
    eye = torch.eye(3)
    anchor = torch.tensor([1.0, 0.0, 0.0])
    m0 = union_plus_anchor_metric(eye * 0.0, anchor, 0.0)
    m1 = union_plus_anchor_metric(eye * 0.0, anchor, 2.0)
    torch.testing.assert_close(m0, torch.zeros(3, 3))
    torch.testing.assert_close(m1, 2.0 * anchor_projector(anchor))


def test_blend_anchor_raises_diagonals_on_center_support():
    diags = torch.zeros(4, 3)
    centers = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assignments = torch.tensor([0, 0, 1, 1])
    blended = blend_anchor_into_diagonals(diags, centers, assignments, eta=1.0)
    assert float(blended[0, 0]) == 1.0
    assert float(blended[2, 1]) == 1.0
    assert float(blended[:, 2].sum()) == 0.0


def test_estimate_coupling_with_anchor_eta_changes_g():
    torch.manual_seed(0)
    n, d_in, d_out = 64, 6, 5
    x = torch.randn(n, d_in)
    # Two separated output clusters so charts are stable.
    y = torch.randn(n, d_out)
    y[: n // 2, 0] += 5.0
    y[n // 2 :, 1] += 5.0
    g0, d0 = estimate_output_tangent_coupling(x, y, charts=2, rank=1, alpha=1.0, seed=0)
    g1, d1 = estimate_output_tangent_coupling(
        x, y, charts=2, rank=1, alpha=1.0, seed=0, anchor_eta=1.0
    )
    assert d1["anchor_eta"] == 1.0
    assert not torch.allclose(g0, g1)
    assert d1["sample_projector_diagonals"].mean() >= d0["sample_projector_diagonals"].mean()


def test_coupling_from_diagonals_matches_manual():
    diags = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    g = coupling_from_diagonals(diags, inputs)
    # denom = [1, 4]; numerator = diags.T @ x2 = [[1, 2], [0, 2]]
    expected = torch.tensor([[1.0, 0.5], [0.0, 0.5]])
    torch.testing.assert_close(g, expected)
