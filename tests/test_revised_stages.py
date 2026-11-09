"""Revised note §6: Stages 0–4 as one runnable protocol on a synthetic layer."""

from __future__ import annotations

from sparsifier.eval.revised_protocol import (
    d1_decision,
    d3_decision,
    h1_disconfirmed,
    run_revised_protocol,
    stage4_within_tolerance,
)


def test_d1_decision_uses_the_kolmogorov_critical_value():
    assert d1_decision(ks=0.01, n=400) == "leverage_null_compatible"
    assert d1_decision(ks=0.5, n=400) == "leverage_null_different"


def test_d3_enrichment_is_screening_not_a_pruning_decision():
    assert d3_decision(kappa_tangent=1.1, kappa_presence=1.0) == "gradient_energy_screening_only"
    assert d3_decision(kappa_tangent=4.0, kappa_presence=1.0) == "gradient_energy_screening_only"


def test_h1_is_disconfirmed_without_a_held_out_gain():
    assert h1_disconfirmed(unstable=False, local_error=0.4, global_error=0.2) is True
    assert h1_disconfirmed(unstable=True, local_error=0.1, global_error=0.4) is True
    assert h1_disconfirmed(unstable=False, local_error=0.1, global_error=0.4) is False


def test_stage4_tolerance_is_relative_to_the_unadjusted_arm():
    baseline = {"ambient": 1.0, "presence": 1.0}
    assert stage4_within_tolerance({"ambient": 1.05, "presence": 1.05}, baseline, tolerance=0.10)
    assert not stage4_within_tolerance({"ambient": 1.2, "presence": 1.0}, baseline, tolerance=0.10)


def test_revised_protocol_runs_stages_zero_through_four():
    report = run_revised_protocol(seed=0, n_train=32, n_test=16, sparsity=0.5, dim=16)
    stage0 = report["stage0"]
    for diagnostic in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert "decision" in stage0[diagnostic]
    assert stage0["D1"]["decision"] in {"leverage_null_compatible", "leverage_null_different"}
    assert stage0["D3"]["decision"] == "gradient_energy_screening_only"
    assert stage0["D3"]["kappa_tangent"] > 2.0
    assert stage0["D6"]["decision"] in {"presence_required", "tangent_only"}

    circle = report["stage1"]["planar_circle"]
    harmonic = report["stage1"]["multi_harmonic"]
    assert circle["rank_reduction_possible"] is False
    assert harmonic["rank_reduction_possible"] is True
    assert circle["analytic_b"] == 2 and circle["analytic_r"] == 1
    assert harmonic["analytic_b"] == 4 and harmonic["analytic_r"] == 1
    assert "h1_disconfirmed" in harmonic
    assert harmonic["heldout_local_error"] >= 0.0
    assert harmonic["heldout_global_error"] >= 0.0

    scores = report["stage2"]["spearman_vs_compensated_tangent"]
    for name in ("magnitude", "wanda", "ambient_obs", "fisher", "geometric"):
        assert name in scores
    assert "editability_spearman" in report["stage2"]
    assert "probe_mse" in report["stage2"]["downstream"]
    assert "kl_to_dense" in report["stage2"]["downstream"]
    assert "selective" in report["stage2"]["selective_vs_broad"]
    assert "broad" in report["stage2"]["selective_vs_broad"]

    variant_a = report["stage3"]["variant_a"]
    alphas = [point["alpha_in"] for point in variant_a["unstructured"]]
    assert alphas == [0.0, 0.3, 1.0, 3.0]
    assert [point["alpha_in"] for point in variant_a["nm_2_4"]] == alphas
    for rung in (
        "ambient",
        "static_union",
        "global_bsf",
        "local_tangents",
        "tangents_and_presence",
        "fisher",
    ):
        assert rung in report["stage3"]["ladder"]
        assert "ambient" in report["stage3"]["ladder"][rung]
        assert "presence" in report["stage3"]["ladder"][rung]
        assert "per_concept" in report["stage3"]["ladder"][rung]
        assert "kl_to_dense" in report["stage3"]["ladder"][rung]["downstream"]
    assert "projected_cg" in report["stage3"]["exactness"]
    assert "cross_row" in report["stage3"]["exactness"]
    assert "up" in report["stage3"]["upstream"]
    assert "whole_neuron" in report["stage3"]["upstream"]
    assert report["stage3"]["variant_d"]["storage_used"] <= report["stage3"]["variant_d"]["storage_budget"] + 1e-6

    for arm in ("no_bias", "centered", "ones_column", "posthoc_global_mean"):
        row = report["stage4"]["arms"][arm]
        assert "presence_residual" in row
        assert "normal_residual" in row
        assert "within_tolerance" in row
        assert "input_dependent" in row
    assert 0.0 <= report["stage4"]["chart_assignment_agreement"] <= 1.0
    assert report["runtime_s"] >= 0.0
