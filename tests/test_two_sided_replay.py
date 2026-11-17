import json
import pytest
import torch
from torch import nn

from sparsifier.prune.curvature_replay import DiskFactorReplay, ReplayNeuronMetric
from sparsifier.prune.downstream import DownstreamTangentCalibrator, TangentAnchor
from sparsifier.prune.sample_metric import refit


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU allocation required"))])
@pytest.mark.parametrize("structure", ["independent_tokens", "shared_directions"])
def test_two_sided_weight_moments_match_explicit_sequence_response(structure, device):
    torch.manual_seed(81)
    target = nn.Linear(3, 2, bias=False, device=device).double()
    mixing = torch.randn(2, 2, 2, dtype=torch.double, device=device)
    class Anchor(nn.Module):
        def forward(self, value):
            return torch.einsum("btp,tpc->bc", value, mixing)
    anchor = Anchor()
    model = nn.Sequential(target, anchor).requires_grad_(False)
    x = torch.randn(2, 2, 3, dtype=torch.double, device=device)
    frames = torch.linalg.qr(torch.randn(2, 2, 3, 2, dtype=torch.double, device=device)).Q
    events = []
    calibrator = DownstreamTangentCalibrator({"target": target},
        {"anchor": TangentAnchor(anchor, torch.eye(2, dtype=torch.double))}, metric="weight_diagonal",
        num_probes=None, input_alpha=.7, alpha=.3, input_bases={"target": frames},
        input_structure=structure, input_probe_consumer=events.append)
    rng = torch.random.get_rng_state()
    calibrator.run(lambda: model(x))
    assert torch.equal(rng, torch.random.get_rng_state())
    expected = torch.zeros(2, 3, dtype=torch.double, device=device)
    for sample in range(2):
        for output in range(2):
            if structure == "independent_tokens":
                for token in range(2):
                    for direction in range(2):
                        z = torch.outer(mixing[token, :, output], frames[sample, token, :, direction])
                        expected += z.square()
            else:
                for direction in range(2):
                    z = mixing[:, :, output].T@frames[sample, :, :, direction]
                    expected += z.square()
    solver = calibrator.solvers["target"]
    torch.testing.assert_close(solver.input_geometry_energy, expected.cpu())
    solver.finalize_tangents()
    assert solver.tangent_diagnostics["achieved_input_to_ambient_ratio"] == pytest.approx(.7)
    assert events and all(e["term"] == "input" for e in events)
    assert not target._forward_hooks and not anchor._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def factors(tmp_path, structure):
    torch.manual_seed(7)
    replay = DiskFactorReplay(tmp_path, provenance={"seed": 7}, cache_budget_bytes=1)
    g, x = torch.randn(3, 2, dtype=torch.double), torch.randn(3, 3, dtype=torch.double)
    u = torch.linalg.qr(torch.randn(3, 3, 2, dtype=torch.double)).Q
    z = g.T@x
    replay.append("weight", {"gradient": g, "inputs": x}, weight=.5)
    base = [.5*torch.outer(z.flatten(), z.flatten())]
    q = torch.linalg.qr(torch.randn(2, 2, 1, dtype=torch.double)).Q
    local_x = torch.randn(2, 3, dtype=torch.double)
    replay.append("local", {"inputs": local_x, "basis": q}, weight=.25)
    for point, frame in zip(local_x, q):
        for column in frame.T:
            z = torch.outer(column, point).flatten()
            base.append(.25*torch.outer(z, z))
    inputs = []
    if structure == "independent_tokens":
        replay.append("independent", {"gradient": g, "basis": u}, term="input", weight=.4)
        for token in range(3):
            for direction in range(2):
                z = torch.outer(g[token], u[token, :, direction]).flatten()
                inputs.append(.4*torch.outer(z, z))
    else:
        for direction in range(2):
            replay.append("weight", {"gradient": g, "inputs": u[:, :, direction]}, term="input", weight=.4)
            z = (g.T@u[:, :, direction]).flatten()
            inputs.append(.4*torch.outer(z, z))
    replay.seal()
    return replay, sum(base), sum(inputs)


@pytest.mark.parametrize("structure", ["independent_tokens", "shared_directions"])
@pytest.mark.parametrize("columns", [True, False])
def test_replayed_coupled_operator_scores_and_refit_match_dense_reference(tmp_path, structure, columns):
    replay, base, inputs = factors(tmp_path, structure)
    metric = ReplayNeuronMetric(9, replay, gamma=1.3, eta=.8, chunk_size=1)
    h0 = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.double))
    ridge = torch.tensor([[.02, .03, .01], [.05, .04, .03]], dtype=torch.double)
    full = torch.kron(torch.eye(2, dtype=torch.double), h0)+1.3/9*base+.8/9*inputs+torch.diag(ridge.flatten())
    torch.testing.assert_close(metric.matrix((2, 3), h0, ridge), full)
    torch.testing.assert_close(metric.diagonal((2, 3)), base.diagonal().reshape(2, 3)/9)
    torch.testing.assert_close(metric.input_diagonal((2, 3)), inputs.diagonal().reshape(2, 3)/9)
    w = torch.randn(2, 3, dtype=torch.double)
    scores = []
    for column in range(3):
        delta = torch.zeros_like(w)
        delta[:, column] = w[:, column]
        scores.append(delta.flatten()@(full-torch.diag(ridge.flatten()))@delta.flatten())
    torch.testing.assert_close(metric.direct_scores(w, h0), torch.stack(scores))
    keep = torch.tensor([True, False, True]) if columns else torch.tensor([[True, False, True], [False, True, True]])
    actual, diagnostics = metric.refit(w, keep, h0, ridge, rtol=1e-11)
    expected = refit(w, keep.expand_as(w), full)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert diagnostics["full_metric_converged"] and diagnostics["full_metric_relative_stationarity"] < 1e-10
    assert replay.cache_bytes <= replay.cache_budget_bytes
    opened = DiskFactorReplay.open(tmp_path, provenance={"seed": 7}, cache_budget_bytes=0)
    restored = ReplayNeuronMetric(9, opened, gamma=1.3, eta=.8)
    torch.testing.assert_close(restored.action(w, h0, ridge), metric.action(w, h0, ridge))


def test_replay_provenance_disk_limit_sealing_and_corruption(tmp_path):
    replay, _, _ = factors(tmp_path, "independent_tokens")
    with pytest.raises(RuntimeError, match="frozen"):
        replay.append("weight", {})
    with pytest.raises(ValueError, match="provenance"):
        DiskFactorReplay.open(tmp_path, provenance={"seed": 8})
    with pytest.raises(ValueError, match="manifest hash"):
        DiskFactorReplay.open(tmp_path, expected_manifest_sha256="wrong")
    with pytest.raises(ValueError, match="disk budget"):
        DiskFactorReplay(tmp_path/"small", disk_budget_bytes=1).append("local", {"inputs": torch.ones(2, 3), "basis": torch.ones(2, 2, 1)})
    path = next(tmp_path.glob("*.pt"))
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(replay)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU allocation required"))])
def test_production_size_refit_never_materializes_parameter_hessian(tmp_path, monkeypatch, device):
    torch.manual_seed(17)
    shape = (33, 65)  # Above the explicit reference limit.
    replay = DiskFactorReplay(tmp_path, cache_budget_bytes=1024)
    replay.append("weight", {"gradient": torch.randn(2, shape[0]), "inputs": torch.randn(2, shape[1])})
    replay.seal()
    metric = ReplayNeuronMetric(100, replay)
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit Hessian is forbidden")
    monkeypatch.setattr(metric, "matrix", forbidden)
    weight = torch.randn(shape, device=device)
    keep = torch.ones(shape[1], dtype=torch.bool, device=device)
    keep[::3] = False
    candidate, diag = metric.refit(weight, keep, torch.eye(shape[1], device=device), .01, rtol=1e-5)
    assert diag["full_metric_converged"] and torch.isfinite(candidate).all()
    assert not candidate[:, ~keep].any()
    assert diag["full_metric_relative_stationarity"] < 2e-5
    assert sum(b["bytes"] for b in replay.blobs.values()) < weight.numel()*weight.element_size()
    assert replay.cache_bytes <= 1024


def test_two_sided_triplet_normalization_and_cg_failure_are_transactional(tmp_path):
    from test_whole_neuron import TinySwiGLU
    from sparsifier.prune.whole_neuron import SwiGLUTriplet, WholeNeuronConfig, prune_triplet
    torch.manual_seed(32)
    triplet = SwiGLUTriplet(TinySwiGLU(d=2, h=3).double())
    original = [value.clone() for value in triplet.weights]
    replay, _, _ = factors(tmp_path, "independent_tokens")
    metric = ReplayNeuronMetric(20, replay)
    ambient = torch.randn(20, 3, dtype=torch.double)
    moments = {"base": metric.diagonal((2, 3)), "input": metric.input_diagonal((2, 3))}
    cfg = WholeNeuronConfig(.34, "downstream_weight_diagonal", alpha=.3, input_alpha=.7,
                            compensation="full_metric_refit", full_metric_solver="cg", full_metric_rtol=1e-10)
    candidate = prune_triplet(triplet, ambient, cfg, geometric_diagonal=moments, full_metric=metric)
    assert candidate.diagnostics["input_to_ambient_ratio"] == pytest.approx(.7)
    assert candidate.diagnostics["added_to_ambient_ratio"] == pytest.approx(.3)
    assert candidate.diagnostics["full_metric_converged"]
    from dataclasses import replace
    with pytest.raises(RuntimeError, match="did not converge"):
        prune_triplet(triplet, ambient, replace(cfg, full_metric_max_iterations=1),
                      geometric_diagonal=moments, full_metric=metric)
    for actual, expected in zip(triplet.weights, original):
        torch.testing.assert_close(actual, expected)


def test_support_projection_and_manifest_mutation_are_explicit(tmp_path):
    replay, base, inputs = factors(tmp_path, "independent_tokens")
    metric = ReplayNeuronMetric(9, replay, gamma=1.3, eta=.8).with_support(torch.tensor([True, False, True]))
    h0 = torch.eye(3, dtype=torch.double)
    projected = torch.diag(torch.tensor([1., 0., 1., 1., 0., 1.], dtype=torch.double))
    expected = torch.eye(6, dtype=torch.double)+projected@(1.3/9*base+.8/9*inputs)@projected
    torch.testing.assert_close(metric.matrix((2, 3), h0), expected)
    replay.entries[0]["weight"] = 2
    with pytest.raises(ValueError, match="manifest changed"):
        metric.action(torch.ones(2, 3, dtype=torch.double), h0)


def test_sparsegpt_exposes_effective_ridge_after_factorization_retry(monkeypatch):
    from sparsifier.prune.sparsegpt import SparseGPT
    solver = SparseGPT(nn.Linear(3, 2, bias=False))
    native, calls = torch.linalg.cholesky, []
    def retry(matrix, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("synthetic Cholesky failure")
        return native(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, "cholesky", retry)
    solver._invert_hessian(torch.diag(torch.tensor([2., 3., 4.])), .01)
    assert solver.damping_diagnostics["effective_percdamp"] == .02
    assert solver.damping_diagnostics["damping"] == pytest.approx(.06)
    assert solver.damping_diagnostics["factorization_attempts"] == 2


def test_input_chart_fit_excludes_pruning_rows_and_aligns_shared_columns(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import sparsifier.eval.gsm8k_input_tangents as module
    fitted = []
    class Atlas:
        def __init__(self, *args):
            pass
        def fit(self, values):
            fitted.append(values.copy())
            self.dictionary.dictionary = np.eye(values.shape[1])
        def save(self, path):
            path.write_bytes(b"atlas")
        def frames(self, values, **kwargs):
            frames = [np.eye(values.shape[1])[:, :1]*(1 if point[0] >= 0 else -1) for point in values]
            return frames, None, None
    monkeypatch.setattr(module, "FactorAtlas", Atlas)
    monkeypatch.setattr(module, "GramBlockDictionary", lambda *a: SimpleNamespace())
    points = np.array([[1., 0., 0.], [2., 1., 0.], [-100., 50., 0.], [99., 2., 0.]])
    rows = [{"split": s} for s in ("geometry_fit", "geometry_fit", "pruning_calibration", "pruning_calibration")]
    provider = module.fit_input_tangents(points, rows, tmp_path/"atlas.npz", 0, "shared_directions")
    np.testing.assert_array_equal(fitted[0], points[:2])
    actual = provider(torch.tensor([[[-1., 2., 0.], [1., 0., 0.]]]))
    torch.testing.assert_close(actual[0, 0], actual[0, 1])
    assert actual[0, 0, 0, 0] == 1
    with pytest.raises(ValueError, match="calibration rows"):
        module.fit_input_tangents(points, [dict(split="validation")]*4, tmp_path/"bad.npz", 0)


@pytest.mark.parametrize("entrywise", [True, False])
def test_calibration_driver_two_sided_replay_and_refit(tmp_path, monkeypatch, entrywise):
    import numpy as np
    from test_whole_neuron import TinySwiGLU
    from test_gsm8k_whole_neuron import Tokenizer, rows
    import sparsifier.eval.gsm8k_whole_neuron as whole
    import sparsifier.eval.gsm8k_downstream as downstream
    import sparsifier.eval.gsm8k_input_tangents as input_module
    from sparsifier.prune.whole_neuron import WholeNeuronConfig, prune_triplet

    class Atlas:
        def __init__(self, *args):
            pass
        def fit(self, values):
            self.fitted = np.asarray(values).copy()
        def save(self, path):
            path.write_bytes(self.fitted.tobytes())
        def frames(self, values, **kwargs):
            return [np.eye(values.shape[-1])[:, :1] for _ in values], None, None
    monkeypatch.setattr(whole, "FactorAtlas", Atlas)
    monkeypatch.setattr(downstream, "FactorAtlas", Atlas)
    monkeypatch.setattr(input_module, "FactorAtlas", Atlas)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TinySwiGLU()
        def forward(self, x):
            return x+self.mlp(x)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])
            self.embed = nn.Embedding(10, 3)
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return x
    torch.manual_seed(28)
    model = Model().eval().requires_grad_(False)
    original = [p.clone() for p in model.parameters()]
    c4 = np.tile(np.arange(128) % 9, (32, 1))
    if entrywise:
        candidate, diag = downstream.prune_site(model, Tokenizer(), model.model.layers[0].mlp.down_proj,
            model.model.layers[1], rows(), c4, tmp_path, 0, metric="weight_diagonal", input_alpha=.5,
            full_metric_refit=True, full_metric_rtol=1e-4, num_probes=2, probe_count=2, replay_cache_budget_bytes=1)
        assert torch.isfinite(candidate).all() and diag["full_metric_converged"]
        assert diag["achieved_input_to_ambient_ratio"] == pytest.approx(.5)
    else:
        cfg = WholeNeuronConfig(.3, "downstream_weight_diagonal", alpha=.5, input_alpha=.5,
                                compensation="full_metric_refit", full_metric_solver="cg", full_metric_rtol=1e-4)
        triplet, ambient, moments, metric, metadata = whole.calibrate(model, Tokenizer(), 0, rows(), c4,
            cfg, tmp_path, num_probes=2, probe_count=2, cache_budget_bytes=1)
        assert metadata["replay_factors"] > 0 and metadata["input_fit_ids"] == [str(i) for i in range(65)]
        candidate = prune_triplet(triplet, ambient, cfg, geometric_diagonal=moments, full_metric=metric)
        assert candidate.diagnostics["full_metric_converged"]
    for parameter, expected in zip(model.parameters(), original):
        torch.testing.assert_close(parameter, expected)
    assert all(not module._forward_hooks for module in model.modules())
