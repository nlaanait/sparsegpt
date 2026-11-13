import pytest
import torch
from sparsifier.prune.whole_neuron_metric import FrozenNeuronMetric, NeuronProbeConsumer
from sparsifier.prune.downstream import tangent_pullback_diagonal


def test_local_full_direction_cancellation_is_not_weight_diagonal():
    q = torch.tensor([[1.], [-1.]], dtype=torch.double)/2**.5
    w = torch.ones(2, 1, dtype=torch.double)
    metric = FrozenNeuronMetric(1, local_inputs=torch.ones(1, 1, dtype=torch.double), frames=[q])
    h0 = torch.zeros(1, 1, dtype=torch.double)
    assert float(metric.direct_scores(w, h0)) == pytest.approx(0, abs=1e-12)
    assert float((metric.diagonal(w.shape)*w.square()).sum()) == pytest.approx(1)


@pytest.mark.parametrize("sign,expected", [(1, 4.), (-1, 0.)])
def test_streamed_full_group_scores_sum_output_coordinates_before_squaring(sign, expected):
    w = torch.ones(2, 1, dtype=torch.double)
    x = torch.ones(1, 1, 1, dtype=torch.double)
    y = torch.ones(1, 1, 2, dtype=torch.double, requires_grad=True)
    out = (y[..., 0]+sign*y[..., 1]).reshape(1, 1)
    consumer = NeuronProbeConsumer(w, cache_budget_bytes=1024)
    diagonal = tangent_pullback_diagonal(out, torch.ones(1, 1), {"down": y}, num_probes=None,
        weight_inputs={"down": x}, probe_consumer=consumer)["down"]
    assert float(consumer.group_scores[0]) == expected
    torch.testing.assert_close(consumer.diagonal, diagonal)
    full = FrozenNeuronMetric(1, probes=consumer.probes)
    torch.testing.assert_close(full.diagonal(w.shape), diagonal)
    assert float(full.direct_scores(w, torch.zeros(1, 1, dtype=torch.double))) == expected


def test_full_local_action_explicit_matrix_and_cg_refit():
    from sparsifier.prune.sample_metric import quadratic_hessian
    torch.manual_seed(3)
    x, z = torch.randn(30, 3, dtype=torch.double), torch.randn(4, 3, dtype=torch.double)
    frames = [torch.linalg.qr(torch.randn(2, 1, dtype=torch.double)).Q for _ in z]
    metric = FrozenNeuronMetric(len(x), gamma=2, local_inputs=z, frames=frames)
    h0 = x.T@x/len(x)
    matrix = metric.matrix((2, 3), h0, .03)
    expected = quadratic_hessian(x, z, torch.stack([q@q.T for q in frames]), 2, .03)
    torch.testing.assert_close(matrix, expected)
    w = torch.randn(2, 3, dtype=torch.double)
    keep = torch.tensor([True, False, True])
    direct, diagnostics = metric.refit(w, keep, h0, .03)
    cg, cg_diagnostics = metric.refit(w, keep, h0, .03, method="cg", rtol=1e-11)
    torch.testing.assert_close(cg, direct)
    assert diagnostics["full_metric_relative_stationarity"] < 1e-10
    assert cg_diagnostics["full_metric_converged"]
    _, failed = metric.refit(w, keep, h0, .03, method="cg", max_iterations=0)
    assert not failed["full_metric_converged"]


def test_probe_cache_limit_and_full_reference_size_limit():
    with pytest.raises(ValueError, match="cache budget"):
        FrozenNeuronMetric(1, probes=[(torch.ones(4, 4), 1)], cache_budget_bytes=1)
    with pytest.raises(ValueError, match="2048"):
        FrozenNeuronMetric(1).matrix((100, 100), torch.eye(100))


def test_downstream_full_matrix_and_equal_down_coordinate_bridge():
    from sparsifier.prune.whole_neuron_metric import down_only_bridge
    torch.manual_seed(10)
    x, w = torch.randn(30, 3, dtype=torch.double), torch.randn(2, 3, dtype=torch.double)
    z = torch.randn(2, 3, dtype=torch.double)
    metric = FrozenNeuronMetric(len(x), gamma=2., probes=[(z, .5)])
    h0 = x.T@x/len(x)
    matrix = metric.matrix(w.shape, h0, .03)
    expected = torch.kron(torch.eye(2, dtype=torch.double), h0+.03*torch.eye(3, dtype=torch.double))
    expected += 2*.5/len(x)*torch.outer(z.flatten(), z.flatten())
    torch.testing.assert_close(matrix, expected)
    columns = torch.tensor([True, False, True])
    entries = torch.tensor([[True, False, True], [True, True, False]])
    result = down_only_bridge(w, x, entries, columns, ridge=.03, full_metric=metric)
    assert result["column"]["free_gradient_norm"] < 1e-10
    assert result["entrywise"]["free_gradient_norm"] < 1e-10
    with pytest.raises(ValueError, match="equal"):
        down_only_bridge(w, x, torch.ones_like(entries), columns)
