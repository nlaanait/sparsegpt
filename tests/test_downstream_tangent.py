"""Numerical contracts for downstream activation geometry, not task gradients."""

import copy

import pytest
import torch
from torch import nn
import transformers

from sparsifier.prune.downstream import (
    ChartTangentBasis,
    DownstreamTangentCalibrator,
    DownstreamTangentSparseGPT,
    TangentAnchor,
    tangent_pullback_diagonal,
)
from sparsifier.prune.rescale import prune_with_rescaling
from sparsifier.prune.sparsegpt import SparseGPT


def test_exact_pullback_matches_explicit_jacobian_with_token_mixing():
    torch.manual_seed(1)
    y = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    mixing = torch.randn(4, 6, dtype=torch.double)
    transform = lambda value: torch.sin(mixing @ value.flatten()).reshape(2, 2)
    output = transform(y)
    # Different tangent direction at each downstream token.
    basis = torch.tensor([[[1.], [0.]], [[0.], [1.]]], dtype=torch.double)
    jacobian = torch.autograd.functional.jacobian(transform, y).reshape(4, 6)
    projector = torch.diag(torch.tensor([1., 0., 0., 1.], dtype=torch.double))
    expected = (jacobian.T @ projector @ jacobian).diagonal().reshape_as(y)
    result = tangent_pullback_diagonal(output, basis, {"y": y}, num_probes=None)
    torch.testing.assert_close(result["y"], expected)
    assert not result["y"].requires_grad
    # Finite perturbations converge to the predicted projected downstream cost.
    delta = torch.randn_like(y)
    eps = 1e-5
    actual = (transform(y.detach() + eps * delta) - transform(y.detach())).flatten()
    linear = jacobian @ delta.flatten()
    torch.testing.assert_close(actual @ projector @ actual / eps**2,
                               linear @ projector @ linear, atol=1e-4, rtol=1e-4)


def test_random_probes_converge_and_are_reproducible_without_parameter_grads():
    torch.manual_seed(3)
    layer = nn.Linear(2, 2).double()
    y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    # Mix tokens as well as output channels.
    forward = lambda: layer(y).sum(0, keepdim=True).expand(2, 2)
    basis = torch.eye(2, dtype=torch.double)
    expected = tangent_pullback_diagonal(forward(), basis, {"y": y}, num_probes=None)["y"]
    random_state = torch.random.get_rng_state()
    values = []
    for _ in range(2):
        values.append(tangent_pullback_diagonal(
            forward(), basis, {"y": y}, num_probes=512,
            generator=torch.Generator().manual_seed(4),
        )["y"])
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(values[0], expected, atol=0.03, rtol=0.15)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert y.grad is None
    assert all(parameter.grad is None for parameter in layer.parameters())


def test_disconnected_targets_and_zero_rank_charts_have_zero_metric():
    y = torch.randn(3, 2, requires_grad=True)
    other = torch.randn(3, 4, requires_grad=True)
    result = tangent_pullback_diagonal(y.square(), torch.eye(2), {"other": other}, num_probes=None)
    assert torch.count_nonzero(result["other"]) == 0
    zero = tangent_pullback_diagonal(y.square(), torch.empty(2, 0), {"y": y})
    assert torch.count_nonzero(zero["y"]) == 0
    # Padded empty charts do not invent tangent directions.
    padded = torch.zeros(3, 2, 2)
    padded[0, 0, 0] = 1
    exact = tangent_pullback_diagonal(y, padded, {"y": y}, num_probes=None)
    torch.testing.assert_close(exact["y"], padded.square().sum(-1))


@pytest.mark.parametrize("basis", [torch.ones(2, 2), torch.ones(3, 1), torch.full((2, 1), float("nan"))])
def test_invalid_frames_are_rejected(basis):
    y = torch.randn(3, 2, requires_grad=True)
    with pytest.raises(ValueError, match="basis"):
        tangent_pullback_diagonal(y, basis, {"y": y})


def test_calibrator_matches_sample_paired_curvature_and_preserves_frozen_model():
    torch.manual_seed(5)
    target = nn.Linear(3, 2, bias=False).double()
    downstream = nn.Linear(2, 2, bias=False).double()
    model = nn.Sequential(target, nn.Tanh(), downstream).requires_grad_(False).eval()
    x = torch.randn(2, 4, 3, dtype=torch.double)
    basis = torch.tensor([[1.], [0.]], dtype=torch.double)
    weights = [p.clone() for p in model.parameters()]
    calibrator = DownstreamTangentCalibrator(
        {"target": target}, {"anchor": TangentAnchor(downstream, basis)},
        alpha=0.7, num_probes=None,
    )
    # Even an outer no_grad scope is safe: calibration explicitly enables AD.
    with torch.no_grad():
        calibrator.run(lambda: model(x))
    solver = calibrator.solvers["target"]
    y = target(x)
    b = (1 - y.tanh().square()).square() * downstream.weight[0].square()
    expected_geo = b.reshape(-1, 2).T @ x.reshape(-1, 3).square()
    torch.testing.assert_close(solver.geometry_energy, expected_geo)
    torch.testing.assert_close(solver.H.double(), x.reshape(-1, 3).T @ x.reshape(-1, 3), atol=2e-6, rtol=2e-6)
    solver.finalize_tangents()
    ratio = (solver.scale.double().square() - 1) * solver.input_energy
    assert float(ratio.sum() / (2 * solver.input_energy.sum())) == pytest.approx(0.7, abs=2e-7)
    assert calibrator.batch_count == 1
    for before, parameter in zip(weights, model.parameters()):
        torch.testing.assert_close(before, parameter)
        assert not parameter.requires_grad and parameter.grad is None
    assert not target._forward_hooks and not downstream._forward_hooks


def test_multiple_anchors_sum_metrics_and_activation_basis_callback_is_detached():
    target = nn.Linear(2, 2, bias=False).double()
    middle = nn.Linear(2, 2, bias=False).double()
    model = nn.Sequential(target, middle).requires_grad_(False)
    with torch.no_grad():
        middle.weight.copy_(torch.tensor([[2., 3.], [-1., 4.]], dtype=torch.double))
    x = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.double)

    def frames(output):
        assert not output.requires_grad
        return torch.eye(2, dtype=torch.double)

    calibrator = DownstreamTangentCalibrator(
        {"target": target, "middle": middle},
        {"local": TangentAnchor(target, frames, weight=0.5),
         "far": TangentAnchor(middle, torch.eye(2), weight=2.0)},
        num_probes=None,
    )
    calibrator.run(lambda: model(x))
    b = 0.5 + 2 * middle.weight.square().sum(0)
    expected = b[:, None] * x.square().sum(0)
    torch.testing.assert_close(calibrator.solvers["target"].geometry_energy, expected)
    torch.testing.assert_close(calibrator.solvers["middle"].geometry_energy,
                               2 * target(x).square().sum(0).expand(2, -1))


def test_input_tangents_and_unobserved_features_are_handled_explicitly():
    layer = nn.Linear(3, 2, bias=False)
    x = torch.tensor([[1., 2., 0.], [2., 1., 0.]])
    b = torch.tensor([[1., 0.], [0., 2.]])
    frames = torch.zeros(2, 3, 1)
    frames[0, 0, 0] = 1
    frames[1, 1, 0] = 1
    solver = DownstreamTangentSparseGPT(layer, alpha=0.5, input_alpha=0.25)
    solver.add_pullback_batch(x, b, input_basis=frames)
    torch.testing.assert_close(solver.input_geometry_energy,
                               (b.T @ frames.square().sum(-1)).double())
    solver.finalize_tangents()
    assert torch.equal(solver.scale[:, 2], torch.ones(2))
    assert solver.tangent_diagnostics["unsupported_inputs"] == 1
    ratio = (solver.scale.double().square() - 1) * solver.input_energy
    assert float(ratio.sum() / (2 * solver.input_energy.sum())) == pytest.approx(0.75, abs=2e-7)
    # Pairing matters: globally averaged channel sensitivity loses it.
    assert solver.scale[0, 0] != solver.scale[0, 1]


@pytest.mark.parametrize("nm", [False, True])
def test_pruning_matches_existing_rescaled_compensation(nm):
    torch.manual_seed(8)
    layer = nn.Linear(8, 3, bias=False)
    reference = copy.deepcopy(layer)
    x = torch.randn(4, 16, 8)
    b = torch.rand(4, 16, 3)
    solver = DownstreamTangentSparseGPT(layer)
    solver.add_pullback_batch(x, b)
    solver.finalize_tangents()
    ambient = SparseGPT(reference)
    ambient.add_batch(x)
    kwargs = {"zeros_per_group": 2, "group_size": 4} if nm else {}
    prune_with_rescaling(ambient, solver.scale, 0.5, blocksize=4, **kwargs)
    solver.fasterprune(0.5, blocksize=4, **kwargs)
    torch.testing.assert_close(layer.weight, reference.weight)
    assert torch.count_nonzero(layer.weight == 0) == layer.weight.numel() // 2
    if nm:
        assert ((layer.weight == 0).reshape(3, 2, 4).sum(-1) == 2).all()
    with pytest.raises(RuntimeError, match="already"):
        solver.fasterprune(0.5)


def test_zero_geometry_keeps_ambient_scale_and_multiple_batches_accumulate():
    layer = nn.Linear(2, 2, bias=False)
    calibrator = DownstreamTangentCalibrator(
        {"target": layer}, {"anchor": TangentAnchor(layer, torch.empty(2, 0))},
    )
    calibrator.run(lambda: layer(torch.ones(3, 2)))
    calibrator.run(lambda: layer(torch.ones(4, 2)))
    solver = calibrator.solvers["target"]
    solver.finalize_tangents()
    assert solver.vector_count == 7
    torch.testing.assert_close(solver.input_energy, torch.full((2,), 7., dtype=torch.double))
    assert torch.equal(solver.scale, torch.ones(2, 2))
    assert solver.tangent_diagnostics["gamma"] == 0


def test_conv1d_layout_and_tuple_anchor_selector():
    target = transformers.Conv1D(3, 2)

    class TupleAnchor(nn.Module):
        def forward(self, y):
            return y.square(), None

    anchor = TupleAnchor()
    model = nn.Sequential(target, anchor).requires_grad_(False)
    calibrator = DownstreamTangentCalibrator(
        {"target": target},
        {"anchor": TangentAnchor(anchor, torch.eye(3), select=lambda out: out[0])},
        num_probes=None,
    )
    x = torch.randn(2, 4, 2)
    calibrator.run(lambda: model(x))
    solver = calibrator.solvers["target"]
    y = target(x)
    torch.testing.assert_close(solver.geometry_energy,
                               4 * y.reshape(-1, 3).double().square().T @ x.reshape(-1, 2).double().square())
    solver.fasterprune(0.5, blocksize=2)
    assert target.weight.shape == (2, 3)
    assert int((target.weight == 0).sum()) == 3


def test_hooks_cleaned_after_failure_and_existing_parameter_gradients_untouched():
    layer = nn.Linear(2, 2)
    for parameter in layer.parameters():
        parameter.grad = torch.ones_like(parameter)
    calibrator = DownstreamTangentCalibrator(
        {"target": layer}, {"anchor": TangentAnchor(layer, torch.eye(2))}, num_probes=None,
    )
    x = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="more than once"):
        calibrator.run(lambda: layer(layer(x)))
    assert not layer._forward_hooks
    assert calibrator.batch_count == 0
    assert calibrator.solvers["target"].vector_count == 0
    calibrator.run(lambda: layer(x))
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in layer.parameters())
    assert not layer._forward_hooks
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        calibrator.run(lambda: layer(x))


def test_missing_input_geometry_and_missing_targets_are_rejected():
    target, anchor = nn.Linear(2, 2), nn.Identity()
    with pytest.raises(ValueError, match="input basis"):
        DownstreamTangentCalibrator({"x": target}, {"a": TangentAnchor(anchor, torch.eye(2))}, input_alpha=1)
    calibrator = DownstreamTangentCalibrator({"x": target}, {"a": TangentAnchor(anchor, torch.eye(2))})
    with pytest.raises(RuntimeError, match="every target"):
        calibrator.run(lambda: target(torch.ones(2, 2)))
    assert not target._forward_hooks and not anchor._forward_hooks


def test_frozen_chart_assignment_uses_fit_geometry_on_new_examples():
    torch.manual_seed(11)
    first = torch.randn(32, 3) * torch.tensor([1., .01, .01]) + torch.tensor([0., -10., 0.])
    second = torch.randn(32, 3) * torch.tensor([.01, .01, 1.]) + torch.tensor([0., 10., 0.])
    atlas = ChartTangentBasis.fit(torch.cat((first, second)), charts=2, rank=1)
    centers_before, bases_before = atlas.centers.clone(), atlas.bases.clone()
    query = torch.tensor([[[0., -10., 0.], [0., 10., 0.]]], requires_grad=True)
    frames = atlas(query)
    assert frames.shape == (1, 2, 3, 1) and not frames.requires_grad
    assert float(frames[0, 0, 0].square()) > .99
    assert float(frames[0, 1, 2].square()) > .99
    torch.testing.assert_close(atlas.centers, centers_before)
    torch.testing.assert_close(atlas.bases, bases_before)
    target = nn.Linear(2, 3)
    anchor = nn.Identity()
    model = nn.Sequential(target, anchor).requires_grad_(False)
    calibrator = DownstreamTangentCalibrator(
        {"target": target}, {"anchor": TangentAnchor(anchor, atlas)}, num_probes=None,
    )
    calibrator.run(lambda: model(torch.randn(4, 2)))
    calibrator.solvers["target"].finalize_tangents()
    assert torch.isfinite(calibrator.solvers["target"].scale).all()


def test_keyword_input_and_default_probe_generator_preserve_rng_state():
    layer = nn.Linear(2, 2).requires_grad_(False)
    calibrator = DownstreamTangentCalibrator(
        {"target": layer}, {"anchor": TangentAnchor(layer, torch.eye(2))}, num_probes=4,
    )
    rng_before = torch.random.get_rng_state()
    calibrator.run(lambda: layer(input=torch.ones(2, 2)))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    y = torch.ones(2, 2, requires_grad=True)
    tangent_pullback_diagonal(y.square(), torch.eye(2), {"y": y}, num_probes=4)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    solver = calibrator.solvers["target"]
    solver.free()
    with pytest.raises(RuntimeError, match="freed"):
        solver.finalize_tangents()
    assert solver.geometry_energy.numel() == 0
