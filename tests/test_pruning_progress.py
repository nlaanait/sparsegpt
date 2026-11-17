import json

import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.gram_dictionary import GramBlockDictionary
from sparsifier.prune.progress import Progress, compact_diagnostics
from sparsifier.prune.sparsegpt import SparseGPT
from sparsifier.prune.whole_neuron_metric import FrozenNeuronMetric


def test_dictionary_logging_preserves_fit_and_reports_actual_fixed_code_loss():
    class RecordingDictionary(GramBlockDictionary):
        def encode(self, *args, **kwargs):
            result = super().encode(*args, **kwargs)
            self.last_codes = result.copy()
            return result
    h = np.random.default_rng(2).normal(size=(6, 70))
    plain = RecordingDictionary(2, 2, 1, 3).fit(h, iterations=4)
    events = []
    logged = RecordingDictionary(2, 2, 1, 3).fit(h, iterations=4, progress=events.append)
    np.testing.assert_array_equal(plain.dictionary, logged.dictionary)
    x = h-logged.mean
    expected = np.square(x-logged.dictionary@logged.last_codes).sum()/np.square(x).sum()
    assert events[-1]["fixed_codes_relative_squared_error"] == pytest.approx(expected)
    assert [e["iteration"] for e in events] == [1, 2, 3, 4]
    assert logged.fit_diagnostics["stopping_rule"] == "fixed_iteration_budget"


def test_chart_logging_preserves_tangents_and_persists_rank_and_spectrum(tmp_path):
    theta = np.random.default_rng(3).uniform(-np.pi, np.pi, 100)
    y = np.column_stack([np.cos(theta), np.sin(theta)])
    def atlas(progress=None):
        a = FactorAtlas(1, 2, 1, 3, 1, 4)
        a.dictionary.dictionary, a.dictionary.mean = np.eye(2), np.zeros((2, 1))
        return a.fit_charts(y, progress=progress)
    plain, events = atlas(), []
    logged = atlas(events.append)
    for a, b in zip(plain.frames(y)[0], logged.frames(y)[0]):
        np.testing.assert_array_equal(a, b)
    charts = [e for e in events if e["event"] == "chart_fitted"]
    assert len(charts) == 3
    assert all(e["retained_singular_vectors"] == 1 and len(e["singular_values"]) == 2 for e in charts)
    assert all(0 < e["retained_variance_fraction"] <= 1 for e in charts)
    assert logged.diagnostics(y)["retained_singular_vectors"] == 3
    path = tmp_path/"atlas.npz"
    logged.save(path)
    assert FactorAtlas.load(path).fit_diagnostics == logged.fit_diagnostics


def test_sparsegpt_logged_loss_matches_explicit_damped_objective_and_weights():
    torch.manual_seed(11)
    x, weight = torch.randn(50, 7), torch.randn(3, 7)
    def prune(progress=None):
        layer = nn.Linear(7, 3, bias=False)
        layer.weight.data.copy_(weight)
        solver = SparseGPT(layer)
        solver.add_batch(x)
        h = solver.H.clone()
        solver.fasterprune(.5, blocksize=3, mask_weight=torch.ones_like(weight), progress=progress)
        return layer.weight.detach(), h, solver
    plain, _, _ = prune()
    events = []
    logged, h, solver = prune(events.append)
    assert torch.equal(plain, logged)
    residual = weight-logged
    h += solver.damping_diagnostics["damping"]*torch.eye(7)
    expected = float(((residual@h)*residual).sum()/2)
    assert solver.pruning_diagnostics["obs_surrogate_objective"] == pytest.approx(expected, rel=2e-5)
    assert [e["columns_done"] for e in events if e["event"] == "sparsegpt_block"] == [3, 6, 7]


@pytest.mark.parametrize("iterations", [1, 50])
def test_cg_logs_true_residual_and_nonconvergence_without_changing_solution(iterations):
    torch.manual_seed(21)
    x, w = torch.randn(20, 5, dtype=torch.double), torch.randn(2, 5, dtype=torch.double)
    h = x.T@x/len(x)
    metric = FrozenNeuronMetric(len(x), probes=[(torch.randn_like(w), 2.)])
    keep = torch.tensor([True, False, True, True, False])
    plain, _ = metric.refit(w, keep, h, .03, method="cg", rtol=1e-11, max_iterations=iterations)
    events = []
    logged, diagnostics = metric.refit(w, keep, h, .03, method="cg", rtol=1e-11,
                                      max_iterations=iterations, progress=events.append)
    assert torch.equal(plain, logged)
    gradient = metric.action(w-logged, h, .03)[:, keep]
    initial = metric.action(torch.where(keep, 0., w), h, .03)[:, keep]
    assert diagnostics["full_metric_relative_residual"] == pytest.approx(float(gradient.norm()/initial.norm()))
    assert events[-1]["event"] == "full_metric_completed"
    assert events[-1]["full_metric_converged"] == (iterations == 50)
    assert events[0]["event"] == "full_metric_started"
    assert next(e for e in events if e["event"] == "full_metric_cg")["residual_kind"] == "recomputed"


def test_whole_neuron_logging_keeps_mask_and_refit_identical():
    from test_whole_neuron import TinySwiGLU
    from sparsifier.prune.whole_neuron import SwiGLUTriplet, WholeNeuronConfig, prune_triplet
    torch.manual_seed(5)
    triplet = SwiGLUTriplet(TinySwiGLU().double())
    x = torch.randn(25, 7, dtype=torch.double)
    config = WholeNeuronConfig(.5, "ambient", blocksize=3)
    plain = prune_triplet(triplet, x, config)
    events = []
    logged = prune_triplet(triplet, x, config, progress=events.append)
    assert torch.equal(plain.keep, logged.keep) and torch.equal(plain.down, logged.down)
    assert events[-1]["surrogate_relative_stationarity"] < 1e-12
    assert events[-1]["surrogate_post_refit_objective"] <= events[-1]["surrogate_pre_refit_objective"]+1e-12
    assert events[-1]["removed_neurons"] == 3


def test_error_events_and_compaction_preserve_actionable_metrics(capsys):
    log = Progress(site=[3, "mlp.down_proj"])
    with pytest.raises(RuntimeError):
        with log.phase("output_atlas"):
            raise RuntimeError("degenerate chart")
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["event"] == "phase_failed" and record["error"] == "degenerate chart"
    assert record["site"] == [3, "mlp.down_proj"]
    compact = compact_diagnostics({"full_metric_iterations": 20, "kept_indices": list(range(100)),
                                   "full_metric_relative_residual": .01, "probe_ids": ["a"]})
    assert compact == {"full_metric_iterations": 20, "full_metric_relative_residual": .01}


def test_reader_recovers_actual_retained_ranks_from_historical_atlas(tmp_path):
    from sparsifier.eval.gsm8k_progress import site_record
    np.savez(tmp_path/"atlas-003.npz", **{"0_basis_0": np.ones((4, 1)), "1_basis_0": np.ones((4, 2))})
    record = site_record(tmp_path, 3, {"site": [1, "mlp.gate_proj"], "removed": 12,
        "diagnostics": {"atlas": {"relative_reconstruction_error": .02}, "scale_max": 4.}})
    assert record["metrics"]["atlas"] == {"relative_reconstruction_error": .02,
                                         "retained_singular_vectors": 3, "charts": 2}
    assert "dictionary_iterations" not in record["metrics"]["atlas"]
    assert record["removed_weights"] == 12
