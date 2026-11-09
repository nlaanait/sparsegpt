import pytest
import torch

from sparsifier.eval.gsm8k_exactness import compare_fixed_mask, mask_overlap


def test_controlled_audit_is_stationary_on_same_mask_and_separates_gaps():
    torch.manual_seed(18)
    w, x = torch.randn(3, 4), torch.randn(30, 4)
    q = torch.randn(8, 3, 1, dtype=torch.double)
    q /= q.norm(dim=1, keepdim=True)
    result = compare_fixed_mask(w, x, x[:8], q @ q.transpose(1, 2),
                                alpha=2.)
    assert max(result["free_gradient_max"].values()) < 1e-10
    assert sum(sum(row) for row in result["keep_mask"]) == 6
    for gap in ("output_coupling_gap", "row_congruence_gap", "ordered_solver_gap"):
        assert result[gap] >= -1e-10
    losses = result["losses"]
    for values in losses.values():
        assert values["full"] >= losses["full_coupled_exact"]["full"] - 1e-10
        assert values["full"] >= values["full_unregularized"]
        assert values["full_isotropic_ridge"] >= values["full_unregularized"]
    # All rows protect a nonzero projector: scales >= 1 make the common
    # scaled-coordinate ridge stronger than the isotropic ridge.
    assert losses["ordered_sparsegpt"]["full"] > losses["ordered_sparsegpt"]["full_isotropic_ridge"]


def test_alpha_zero_removes_metric_approximations_but_retains_ordered_solver_gap():
    torch.manual_seed(3)
    w, x = torch.randn(3, 4), torch.randn(20, 4)
    metrics = torch.eye(3).expand(4, -1, -1)
    result = compare_fixed_mask(w, x, x[:4], metrics, alpha=0)
    assert result["output_coupling_gap"] == pytest.approx(0, abs=1e-12)
    assert result["row_congruence_gap"] == pytest.approx(0, abs=1e-12)
    assert result["ordered_solver_gap"] > 0


def test_mask_overlap_measures_removed_weights():
    assert mask_overlap([[True, False, False]], [[False, True, False]]) == {
        "removed_jaccard": 1/3, "changed_fraction": 2/3}
