import pytest
import torch
from torch import nn

from sparsifier.prune.downstream import tangent_pullback_diagonal, DownstreamTangentCalibrator, TangentAnchor


@pytest.mark.parametrize("sign,expected", [(-1, 0.), (1, 4.)])
def test_token_cancellation_and_reinforcement(sign, expected):
    y = torch.ones(1, 2, 1, dtype=torch.double, requires_grad=True)
    out = y[:, 0] + sign * y[:, 1]
    k = tangent_pullback_diagonal(out, torch.ones(1, 1), {"y": y},
                                 weight_inputs={"y": torch.ones_like(y)}, num_probes=None)
    torch.testing.assert_close(k["y"], torch.tensor([[expected]], dtype=torch.double))


def test_rank_reversal_and_shared_weight_jacobian_diagonal():
    x = torch.tensor([[[1., .5], [1., -.5]]], dtype=torch.double)
    w = torch.tensor([[.7, -.3]], dtype=torch.double, requires_grad=True)
    y = x @ w.T
    out = y[:, 0] - y[:, 1]
    k = tangent_pullback_diagonal(out, torch.ones(1, 1), {"y": y},
                                 weight_inputs={"y": x}, num_probes=None, retain_graph=True)["y"]
    direct, = torch.autograd.grad(out.sum(), w)
    torch.testing.assert_close(k, direct.square())
    torch.testing.assert_close(k, torch.tensor([[0., 1.]], dtype=torch.double))
    act = tangent_pullback_diagonal(out, torch.ones(1, 1), {"y": y}, num_probes=None)["y"]
    old = act.reshape(-1, 1).T @ x.reshape(-1, 2).square()
    torch.testing.assert_close(old, torch.tensor([[2., .5]], dtype=torch.double))


def test_independent_sequences_are_never_summed_before_squaring():
    x = torch.tensor([[[1.], [1.]], [[-1.], [-1.]]], dtype=torch.double)
    y = torch.ones_like(x, requires_grad=True)
    # Coupled anchor makes per-sequence weight gradients opposite.
    output = y.sum().reshape(1, 1)
    k = tangent_pullback_diagonal(output, torch.ones(1, 1), {"y": y},
                                 weight_inputs={"y": x}, num_probes=None)["y"]
    torch.testing.assert_close(k, torch.tensor([[8.]], dtype=torch.double))


def test_nonlinear_weight_diagonal_matches_full_quadratic_and_refit():
    from sparsifier.prune.sample_metric import refit
    torch.manual_seed(2)
    x = torch.randn(1, 3, 2, dtype=torch.double)
    w = torch.randn(2, 2, dtype=torch.double)
    mix = torch.randn(2, 6, dtype=torch.double)
    def f(weight):
        return torch.sin(mix @ (x @ weight.T).flatten())
    y = (x @ w.T).requires_grad_(True)
    output = torch.sin(mix @ y.flatten()).reshape(1, 2)
    k = tangent_pullback_diagonal(output, torch.eye(2), {"y": y},
                                 weight_inputs={"y": x}, num_probes=None)["y"]
    jac = torch.autograd.functional.jacobian(f, w).reshape(2, 4)
    h = jac.T @ jac + .1 * torch.eye(4, dtype=torch.double)
    torch.testing.assert_close(k.flatten(), (jac.T @ jac).diagonal())
    keep = torch.tensor([[True, False], [False, True]])
    candidate = refit(w, keep, h)
    torch.testing.assert_close((h @ (w - candidate).flatten())[keep.flatten()],
                               torch.zeros(2, dtype=torch.double), atol=1e-10, rtol=0)


def test_weight_mode_multiple_anchors_and_supported_trace_guard_statistics():
    layer = nn.Linear(2, 1, bias=False).double().requires_grad_(False)
    calibration = DownstreamTangentCalibrator(
        {"target": layer}, {"a": TangentAnchor(layer, torch.ones(1, 1), weight=2.)},
        metric="weight_diagonal", num_probes=None)
    solver = calibration.solvers["target"]
    solver.add_ambient_batch(torch.tensor([[1., 0.]], dtype=torch.double))
    calibration.run(lambda: layer(torch.tensor([[[0., 1.], [0., 1.]]], dtype=torch.double)), collect_ambient=False)
    solver.finalize_tangents()
    assert solver.tangent_diagnostics["raw_geometry_trace"] == 4. # Independent anchor positions.
    assert solver.tangent_diagnostics["excluded_geometry_trace"] == 4.
    assert solver.tangent_diagnostics["geometry_trace"] == 0
    assert solver.tangent_diagnostics["achieved_geometry_to_ambient_ratio"] == 0
    assert not layer._forward_hooks
