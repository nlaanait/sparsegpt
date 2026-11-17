"""Review regressions for contracts stated in whole-neuron-pruning.md."""
import pytest
import torch
from torch import nn
from sparsifier.prune.whole_neuron import SwiGLUTriplet, WholeNeuronConfig, prune_triplet, geometric_scales
from sparsifier.prune.whole_neuron_metric import FrozenNeuronMetric, StreamedGroupMetric


class MLP(nn.Module):
    def __init__(self, d=2, h=3):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_alpha_zero_direct_full_metric_matches_ambient_with_supplied_metric():
    mlp = MLP(1, 2).double()
    with torch.no_grad():
        mlp.down_proj.weight.copy_(torch.tensor([[1., 2.]]))
    triplet = SwiGLUTriplet(mlp)
    x = torch.eye(2, dtype=torch.double)
    cfg = WholeNeuronConfig(.5, 'local', alpha=0, selector='direct_full_metric')
    baseline = prune_triplet(triplet, x, cfg)
    actual = prune_triplet(triplet, x, cfg,
        full_metric=StreamedGroupMetric(torch.tensor([100., 0.]), len(x)))
    assert torch.equal(actual.keep, baseline.keep)
    assert torch.equal(actual.down, baseline.down)


def test_architecture_rejects_extra_hook_transformation():
    mlp = MLP()
    handle = mlp.down_proj.register_forward_hook(lambda module, args, output: output + 1)
    try:
        with pytest.raises(ValueError):
            SwiGLUTriplet(mlp)
    finally:
        handle.remove()


def test_full_refit_preserves_selection_ridge_in_original_coordinates():
    torch.manual_seed(9)
    triplet = SwiGLUTriplet(MLP().double())
    x = torch.randn(20, 3, dtype=torch.double)
    z = torch.tensor([[3., 1., .2], [.1, 2., 1.]], dtype=torch.double)
    q = [torch.tensor([[1.], [0.]], dtype=torch.double)] * 2
    metric = FrozenNeuronMetric(len(x), local_inputs=z, frames=q)
    k = metric.diagonal(triplet.down.weight.shape)
    cfg = WholeNeuronConfig(.34, 'local', alpha=2, percdamp=.5, compensation='full_metric_refit')
    candidate = prune_triplet(triplet, x, cfg, geometric_diagonal=k, full_metric=metric)
    h0 = x.T @ x / len(x)
    scale, _, info = geometric_scales(h0, k, cfg.alpha, 2)
    full = metric.with_gamma(info['gamma'])
    error = triplet.down.weight.detach() - candidate.down
    delta = candidate.diagnostics['damping']
    # Distinguish the old isotropic objective from the selection's ridge.
    isotropic = full.action(error, h0, delta)[:, candidate.keep]
    assert isotropic.norm() > 1e-6
    # The selection's scaled-coordinate ridge is delta * scale**2.
    selected_ridge = full.action(error, h0, delta * scale.square())[:, candidate.keep]
    assert selected_ridge.norm() < 1e-10
    assert candidate.diagnostics['full_metric_ridge'] == 'anisotropic'
    assert 'surrogate_relative_stationarity' in candidate.diagnostics
    assert 'relative_stationarity' not in candidate.diagnostics


@pytest.mark.parametrize('site', ['root', 'gate', 'activation'])
@pytest.mark.parametrize('pre', [False, True])
def test_hook_rejection_covers_root_and_all_relevant_submodules(site, pre):
    mlp = MLP()
    module = {'root': mlp, 'gate': mlp.gate_proj, 'activation': mlp.act_fn}[site]
    handle = (module.register_forward_pre_hook(lambda module, args: args)
              if pre else module.register_forward_hook(lambda module, args, output: output))
    try:
        with pytest.raises(ValueError, match='hooks'):
            SwiGLUTriplet(mlp)
    finally:
        handle.remove()


def test_full_refit_factor_failure_restarts_selection_and_keeps_requested_damping(monkeypatch):
    torch.manual_seed(12)
    triplet = SwiGLUTriplet(MLP().double())
    x = torch.randn(20, 3, dtype=torch.double)
    q = [torch.tensor([[1.], [0.]], dtype=torch.double)]
    metric = FrozenNeuronMetric(len(x), local_inputs=torch.ones(1, 3, dtype=torch.double), frames=q)
    k = metric.diagonal(triplet.down.weight.shape)
    native = FrozenNeuronMetric.refit
    calls = []
    def transient(self, weight, keep, h0, ridge, **kwargs):
        calls.append(ridge.clone())
        if len(calls) == 1:
            raise torch.linalg.LinAlgError('synthetic full factor failure')
        return native(self, weight, keep, h0, ridge, **kwargs)
    monkeypatch.setattr(FrozenNeuronMetric, 'refit', transient)
    config = WholeNeuronConfig(.34, 'local', alpha=1., compensation='full_metric_refit')
    candidate = prune_triplet(triplet, x, config, geometric_diagonal=k, full_metric=metric)
    assert len(calls) == 2
    torch.testing.assert_close(calls[1], 2*calls[0])
    assert candidate.diagnostics['selection_restarts_for_refit'] == 1
    assert candidate.diagnostics['requested_percdamp'] == .01
    assert candidate.diagnostics['effective_percdamp'] == .02
    assert candidate.diagnostics['full_metric_relative_stationarity'] < 1e-10


@pytest.mark.parametrize('metric', ['ambient', 'local', 'downstream_weight_diagonal'])
def test_alpha_zero_does_not_call_reused_metric(metric):
    class ForbiddenMetric:
        def direct_scores(self, *args):
            raise AssertionError('geometric object must be ignored')
    cfg = WholeNeuronConfig(.34, metric, alpha=0., selector='direct_full_metric')
    triplet = SwiGLUTriplet(MLP().double())
    x = torch.randn(20, 3, dtype=torch.double)
    candidate = prune_triplet(triplet, x, cfg, full_metric=ForbiddenMetric())
    baseline = prune_triplet(triplet, x, cfg)
    assert torch.equal(candidate.keep, baseline.keep)
    assert torch.equal(candidate.down, baseline.down)
