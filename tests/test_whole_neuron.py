import copy
import itertools
import pytest
import torch
from torch import nn
from sparsifier.prune.whole_neuron import (
    SwiGLUTriplet, WholeNeuronConfig, prune_triplet, refit_columns, select_columns,
    apply_candidate, damped_curvature, greedy_columns, geometric_scales,
)


class TinySwiGLU(nn.Module):
    def __init__(self, d=3, h=7):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


def test_refit_sign_stationarity_and_enumerated_masks():
    torch.manual_seed(0)
    x, w = torch.randn(40, 5, dtype=torch.double), torch.randn(3, 5, dtype=torch.double)
    h = x.T@x/len(x)+.03*torch.eye(5, dtype=torch.double)
    for deleted in itertools.combinations(range(5), 2):
        keep = torch.ones(5, dtype=torch.bool)
        keep[list(deleted)] = False
        candidate = refit_columns(w, h, keep, row_chunk=1)
        residual = w-candidate
        torch.testing.assert_close((residual@h)[:, keep], torch.zeros(3, 3, dtype=torch.double), atol=1e-12, rtol=0)
        naive = w-w*keep
        assert (residual@h*residual).sum() <= (naive@h*naive).sum()+1e-12
        # Independent constrained dense solve in row-major coordinates.
        from sparsifier.prune.sample_metric import refit
        explicit = refit(w, keep[None].expand_as(w), torch.kron(torch.eye(3, dtype=torch.double), h))
        torch.testing.assert_close(candidate, explicit)


@pytest.mark.parametrize("width,block,count", [(7, 3, 3), (9, 4, 8), (5, 128, 0)])
def test_shared_block_quotas_and_deterministic_ties(width, block, count):
    w = torch.ones(2, width, dtype=torch.double)
    h = torch.eye(width, dtype=torch.double)
    keep, _, _, diagnostics = select_columns(w, h, count=count, blocksize=block)
    assert int((~keep).sum()) == count
    assert sum(diagnostics["block_quotas"]) == count
    for b in range(0, width, block):
        e = min(width, b+block)
        quota = count*e//width-count*b//width
        assert torch.equal(torch.where(~keep[b:e])[0], torch.arange(quota))


def test_mask_support_and_scaled_independent_row_refits():
    torch.manual_seed(2)
    mlp = TinySwiGLU().double()
    triplet = SwiGLUTriplet(mlp)
    x = torch.randn(40, 7, dtype=torch.double)
    x[:, -1] = 0
    k = torch.rand(3, 7, dtype=torch.double)
    k[:, -1] = 0
    config = WholeNeuronConfig(.5, "local", alpha=1, blocksize=3)
    candidate = prune_triplet(triplet, x, config, geometric_diagonal=k)
    assert candidate.keep[-1] and int((~candidate.keep).sum()) == 3
    h0 = x.T@x/len(x)
    scale, _, _ = geometric_scales(h0, k, 1, 3)
    h, _, _ = damped_curvature(h0, .01)
    reference = triplet.down.weight.detach()
    for row in range(3):
        curvature = scale[row, :, None]*h*scale[row, None, :]
        explicit = refit_columns(reference[row:row+1], curvature, candidate.keep)
        torch.testing.assert_close(candidate.down[row:row+1], explicit)
    before_g, before_u, _ = (w.clone() for w in triplet.weights)
    apply_candidate(triplet, candidate)
    assert torch.equal(mlp.gate_proj.weight[candidate.keep], before_g[candidate.keep])
    assert torch.equal(mlp.up_proj.weight[candidate.keep], before_u[candidate.keep])
    assert not mlp.gate_proj.weight[~candidate.keep].any()
    assert not mlp.up_proj.weight[~candidate.keep].any()
    assert not mlp.down_proj.weight[:, ~candidate.keep].any()
    assert candidate.diagnostics["surrogate_post_refit_objective"] <= candidate.diagnostics["surrogate_pre_refit_objective"]+1e-12


def test_alpha_zero_bypasses_geometry_and_zero_sparsity_is_bitwise():
    mlp = TinySwiGLU()
    triplet = SwiGLUTriplet(mlp)
    x = torch.randn(30, 7)
    ambient = prune_triplet(triplet, x, WholeNeuronConfig(.4))
    for metric in ("local", "downstream_activation_diagonal", "downstream_weight_diagonal"):
        geo = prune_triplet(triplet, x, WholeNeuronConfig(.4, metric, alpha=0))
        assert torch.equal(ambient.keep, geo.keep)
        assert torch.equal(ambient.down, geo.down)
    dense = prune_triplet(triplet, x, WholeNeuronConfig(0))
    assert torch.equal(dense.down, mlp.down_proj.weight)
    assert torch.equal(dense.gate, mlp.gate_proj.weight)


def test_unsupported_geometry_and_cardinality_fail_closed():
    mlp = TinySwiGLU()
    x = torch.zeros(10, 7)
    x[:, 0] = 1
    with pytest.raises(ValueError, match="eligible"):
        prune_triplet(SwiGLUTriplet(mlp), x, WholeNeuronConfig(.5))
    k = torch.zeros(3, 7)
    k[:, 1] = 1
    with pytest.raises(ValueError, match="align calibration"):
        geometric_scales(x.T@x/len(x), k, 1, 3)


def test_architecture_rejects_bias_ties_other_activations_and_extra_transforms():
    mlp = TinySwiGLU()
    mlp.act_fn = nn.ReLU()
    with pytest.raises(ValueError, match="computation"):
        SwiGLUTriplet(mlp)
    mlp = TinySwiGLU()
    mlp.up_proj.weight = mlp.gate_proj.weight
    with pytest.raises(ValueError, match="tied"):
        SwiGLUTriplet(mlp)
    mlp = TinySwiGLU()
    mlp.gate_proj = nn.Linear(3, 7)
    with pytest.raises(ValueError, match="bias-free"):
        SwiGLUTriplet(mlp)


def test_greedy_first_cost_matches_exact_constrained_cost():
    torch.manual_seed(3)
    x, w = torch.randn(30, 6, dtype=torch.double), torch.randn(3, 6, dtype=torch.double)
    h = x.T@x/len(x)+.02*torch.eye(6, dtype=torch.double)
    losses = []
    for q in range(6):
        keep = torch.ones(6, dtype=torch.bool); keep[q] = False
        error = w-refit_columns(w, h, keep)
        losses.append(float((error@h*error).sum()))
    keep = greedy_columns(w, h, 1, torch.ones(6, dtype=torch.bool))
    assert int(torch.where(~keep)[0]) == min(range(6), key=losses.__getitem__)


@pytest.mark.parametrize("kwargs", [{"alpha": 1}, {"blocksize": 1.5}, {"metric": "bad"}, {"sparsity": 1}])
def test_invalid_config(kwargs):
    args = {"sparsity": .5} | kwargs
    with pytest.raises(ValueError):
        WholeNeuronConfig(**args)


def test_refit_failure_restarts_selection_with_one_stronger_ridge(monkeypatch):
    import sparsifier.prune.whole_neuron as module
    original = module.refit_columns
    calls = []
    def transient(weight, h, keep, **kwargs):
        calls.append(h.clone())
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("synthetic retained factor failure")
        return original(weight, h, keep, **kwargs)
    monkeypatch.setattr(module, "refit_columns", transient)
    candidate = prune_triplet(SwiGLUTriplet(TinySwiGLU()), torch.randn(40, 7), WholeNeuronConfig(.5))
    assert len(calls) == 2
    assert (calls[1].diagonal() > calls[0].diagonal()).all()
    assert candidate.diagnostics["selection_restarts_for_refit"] == 1
    assert candidate.diagnostics["requested_percdamp"] == .01
    assert candidate.diagnostics["effective_percdamp"] == .02


def test_failed_apply_rolls_back_all_three_original_tensors(monkeypatch):
    triplet = SwiGLUTriplet(TinySwiGLU())
    candidate = prune_triplet(triplet, torch.randn(40, 7), WholeNeuronConfig(.5))
    originals = tuple(t.clone() for t in triplet.weights)
    native = torch.Tensor.copy_
    failed = False
    def interrupted(tensor, source, *args, **kwargs):
        nonlocal failed
        if tensor is triplet.up.weight and not failed:
            failed = True
            raise RuntimeError("synthetic copy interruption")
        return native(tensor, source, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        apply_candidate(triplet, candidate)
    assert failed
    assert all(torch.equal(old, new) for old, new in zip(originals, triplet.weights))


def test_block_update_matches_independent_columnwise_elimination():
    torch.manual_seed(27)
    x, w = torch.randn(40, 9, dtype=torch.double), torch.randn(3, 9, dtype=torch.double)
    h0 = x.T@x/len(x)
    keep, blocked, h, _ = select_columns(w, h0, count=5, blocksize=4)
    c = torch.linalg.cholesky(torch.linalg.inv(h), upper=True)
    unblocked = w.clone()
    for q in range(9):
        old = unblocked[:, q].clone()
        new = old if keep[q] else torch.zeros_like(old)
        unblocked[:, q:] -= ((old-new)/c[q,q])[:,None]*c[q,q:]
        unblocked[:, q] = new
    torch.testing.assert_close(unblocked, blocked)
