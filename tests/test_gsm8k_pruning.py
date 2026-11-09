import numpy as np
import pytest
import torch

from sparsifier.eval.gsm8k_pruning import pair_inputs, prune_candidate, select_validation


def test_alpha_zero_matches_equal_data_sparsegpt_and_masks_match():
    torch.manual_seed(20)
    weight, inputs, deltas = torch.randn(8, 8), torch.randn(30, 8), torch.randn(12, 8)
    baseline, _ = prune_candidate(weight, inputs, arm="sparsegpt_task", device="cpu")
    zero, _ = prune_candidate(weight, inputs, arm="input_side", alpha=0,
                             deltas=deltas, labels=torch.arange(12), device="cpu")
    torch.testing.assert_close(baseline, zero, rtol=0, atol=0)
    assert int((zero == 0).sum()) == 32
    for arm in ("input_side", "local"):
        result, _ = prune_candidate(weight, inputs, arm=arm, device="cpu", deltas=deltas,
            labels=torch.arange(12), readout_inputs=inputs[:12],
            pdiag=np.tile([1, 0, 0, 0, 0, 0, 0, 0], (12, 1)) if arm == "local" else None)
        assert torch.isfinite(result).all()
        assert int((result == 0).sum()) == 32


def test_nulls_preserve_rotation_spectrum_but_repair_changes_covariance():
    x = np.random.default_rng(7).normal(size=(12, 6)).astype(np.float32)
    mapping = {str(i): i for i in range(12)}
    manifest = {"pairs": [{"left": str(i), "right": str(i+6), "factor": "one", "split": "geometry_fit"}
                           for i in range(6)]}
    original, labels = pair_inputs(manifest, mapping, x)
    rotated, labels2 = pair_inputs(manifest, mapping, x, mode="input_rotated")
    repaired, _ = pair_inputs(manifest, mapping, x, mode="input_repaired")
    torch.testing.assert_close(original @ original.T, rotated @ rotated.T)
    torch.testing.assert_close(labels, labels2)
    assert not torch.allclose(original.T @ original, repaired.T @ repaired)


def test_validation_selection_never_uses_final_test_and_balances_families():
    rows = [{"id": f"{family}-{i}", "family": family, "split": split}
            for family in ("a", "b") for i in range(5)
            for split in ("controlled_test", "validation")]
    chosen = select_validation(rows, 6, seed=4)
    assert all(r["split"] == "validation" for r in chosen)
    assert sum(r["family"] == "a" for r in chosen) == 3
    assert chosen == select_validation(rows, 6, seed=4)
    with pytest.raises(ValueError):
        select_validation(rows, 11, seed=4)
