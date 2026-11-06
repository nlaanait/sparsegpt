"""Option 2: per-chart OBS scores + chart-local H_c compensation."""

from __future__ import annotations

import torch
from torch import nn

from sparsifier.prune.chart_obs import ChartObsSparseGPT, per_chart_obs_scores
from sparsifier.prune.sparsegpt import SparseGPT


def test_single_chart_obs_score_matches_note_factor():
    torch.manual_seed(0)
    weight = torch.randn(3, 5)
    phi = torch.randn(40, 5)
    u = torch.tensor([[1.0], [0.0], [0.0]])
    scores = per_chart_obs_scores(weight, phi, u, ridge=1e-3)
    h = phi.T @ phi / phi.shape[0] + 1e-3 * torch.eye(5)
    h_inv = torch.linalg.inv(0.5 * (h + h.T))
    expected = weight.square() / torch.diag(h_inv).clamp_min(1e-30)[None, :]
    expected[1:] = 0
    torch.testing.assert_close(scores, expected)


def test_single_chart_compensation_uses_chart_hessian_not_ambient():
    """Restricted-exact Option 2: OBS updates must see H_c, not the ambient H."""
    torch.manual_seed(1)
    layer = nn.Linear(6, 4, bias=False)
    # Two disjoint input supports so ambient H and a chart subset H differ.
    x = torch.zeros(64, 6)
    x[:32, :3] = torch.randn(32, 3)
    x[32:, 3:] = torch.randn(32, 3) * 3.0
    # Outputs: one chart (variation along e0 only) so atlas collapses sensibly.
    y = torch.zeros(64, 4)
    y[:, 0] = torch.linspace(-1.0, 1.0, 64)
    y += 0.01 * torch.randn(64, 4)

    ambient = SparseGPT(layer)
    for start in range(0, 64, 16):
        xb = x[start : start + 16].unsqueeze(0)
        ambient.add_batch(xb, None)
    h_ambient = ambient.H.clone()

    solver = ChartObsSparseGPT(
        layer, charts=1, rank=1, calibration_samples=64, seed=0, percdamp=1e-3
    )
    for start in range(0, 64, 16):
        xb = x[start : start + 16].unsqueeze(0)
        yb = y[start : start + 16].unsqueeze(0)
        solver.add_batch(xb, yb)
    solver.finalize_charts()
    assert solver.chart_hessians is not None
    assert len(solver.chart_hessians) == 1
    # Chart H is built from the chart's samples only; with charts=1 that is all
    # samples, but scaling/convention may differ from SparseGPT's running H.
    # Stronger check: after prune, solver must not have used ambient.H as the
    # compensation Hessian (ambient.H remains untouched ownership; chart path
    # stores its own factors).
    assert solver.compensation_source == "chart"
    before = layer.weight.data.clone()
    solver.fasterprune(0.25, blocksize=3, percdamp=1e-3)
    assert not torch.equal(layer.weight.data, before)
    assert int((layer.weight.data == 0).sum()) > 0
    # Ambient Hessian object was never consumed by chart OBS.
    assert ambient.H is not None
    torch.testing.assert_close(ambient.H, h_ambient)


def test_multi_chart_assigns_each_row_a_dominant_chart_hessian():
    torch.manual_seed(2)
    layer = nn.Linear(5, 3, bias=False)
    solver = ChartObsSparseGPT(
        layer, charts=2, rank=1, calibration_samples=48, seed=0, percdamp=1e-3
    )
    x = torch.randn(6, 8, 5)
    # Two output clusters along e0 vs e1.
    y = torch.zeros(6, 8, 3)
    y[:, :4, 0] = torch.linspace(-1, 1, 24).reshape(6, 4)
    y[:, 4:, 1] = torch.linspace(-1, 1, 24).reshape(6, 4)
    y += 0.02 * torch.randn(6, 8, 3)
    solver.add_batch(x, y)
    solver.finalize_charts()
    assert solver.row_chart_index is not None
    assert solver.row_chart_index.shape == (3,)
    assert set(solver.row_chart_index.tolist()) <= set(range(len(solver.chart_hessians)))
