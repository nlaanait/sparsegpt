"""Stage 1–4 harness smoke (CPU, synthetic)."""

from __future__ import annotations

from sparsifier.eval.tangent_stages import run_stages, summarize_stages


def test_run_stages_produces_all_four_sections():
    report = run_stages(seed=0, n_train=64, n_test=32, sparsity=0.25)
    assert "stage1_tangent_stability" in report
    assert "stage2_score_ranking" in report
    assert "stage3_compensated_methods" in report
    assert "stage4_bias_fold" in report
    assert "wanda" in report["stage3_compensated_methods"]
    assert "exact_diag_p" in report["stage3_compensated_methods"]
    summary = summarize_stages([report, run_stages(seed=1, n_train=64, n_test=32, sparsity=0.25)])
    assert summary["seed_count"] == 2
    assert "wanda_vs_heldout_direct_spearman" in summary["stage2_score_ranking"]
