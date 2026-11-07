"""CPU contract for the Qwen stage runner. The GPU path is not imported here."""

from __future__ import annotations

import pytest
import torch

from sparsifier.eval.live_stages import cap_under_brick, complete_layer_stages


def test_cap_stops_before_the_110_gib_brick():
    with pytest.raises(RuntimeError, match="110"):
        cap_under_brick(8.0, total_gib=122.0, available_gib=12.0)
    assert cap_under_brick(8.0, total_gib=122.0, available_gib=65.0) == 8.0
    assert cap_under_brick(80.0, total_gib=122.0, available_gib=65.0) == pytest.approx(45.0)


def test_complete_layer_stages_covers_stage_zero_through_four():
    torch.manual_seed(0)
    weight = torch.randn(8, 8) / 8**0.5
    phi = torch.randn(16, 8)
    knobs = torch.randn(4, 8)
    presence = knobs + torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    gradients = 0.85 * torch.nn.functional.normalize(knobs @ weight.T, dim=1)[0] + 0.15 * torch.randn(16, 8)
    report = complete_layer_stages(
        weight, phi, knobs, presence, gradients, sparsity=0.5, seed=0
    )
    for diagnostic in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert "decision" in report["stage0"][diagnostic]
    assert "h1_disconfirmed" in report["stage1"]
    for name in ("magnitude", "wanda", "ambient_obs", "fisher", "geometric"):
        assert name in report["stage2"]["spearman_vs_compensated_tangent"]
    assert [point["alpha_in"] for point in report["stage3"]["variant_a"]["unstructured"]] == [
        0.0,
        0.3,
        1.0,
        3.0,
    ]
    assert "nm_2_4" in report["stage3"]["variant_a"]
    for rung in ("ambient", "static_union", "global_bsf", "local_tangents", "tangents_and_presence", "fisher"):
        assert rung in report["stage3"]["ladder"]
    assert "projected_cg" in report["stage3"]["exactness"]
    assert "cross_row" in report["stage3"]["exactness"]
    assert report["stage3"]["variant_d"]["storage_used"] <= report["stage3"]["variant_d"]["storage_budget"] + 1e-6
    for arm in ("no_bias", "centered", "ones_column", "posthoc_global_mean"):
        assert "within_tolerance" in report["stage4"]["arms"][arm]
