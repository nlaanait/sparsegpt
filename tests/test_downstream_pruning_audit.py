"""CPU review witnesses: distinguish documented surrogates from exact curvature."""
import pytest
import torch
from torch import nn

from sparsifier.prune.downstream import DownstreamTangentSparseGPT, tangent_pullback_diagonal


@pytest.mark.parametrize('sign,expected', [(-1., 0.), (1., 4.)])
def test_token_diagonal_drops_cancellation_and_reinforcement(sign, expected):
    x = torch.ones(2, 1, dtype=torch.double)
    y = x.clone().requires_grad_()
    anchor = y[0] + sign * y[1]
    b = tangent_pullback_diagonal(anchor, torch.ones(1, 1), {'y': y}, num_probes=None)['y']
    solver = DownstreamTangentSparseGPT(nn.Linear(1, 1, bias=False))
    solver.add_pullback_batch(x, b)
    assert solver.geometry_energy.item() == 2.
    exact_weight_diagonal = (x[0] + sign * x[1]).square().item()
    assert exact_weight_diagonal == expected


def test_activation_diagonal_can_reverse_shared_weight_importance():
    x = torch.tensor([[1., .5], [1., -.5]], dtype=torch.double)
    y = torch.zeros(2, 1, dtype=torch.double, requires_grad=True)
    b = tangent_pullback_diagonal(y[0] - y[1], torch.ones(1, 1), {'y': y}, num_probes=None)['y']
    surrogate = (b.T @ x.square()).flatten()
    exact = (x[0] - x[1]).square()
    torch.testing.assert_close(surrogate, torch.tensor([2., .5], dtype=torch.double))
    torch.testing.assert_close(exact, torch.tensor([0., 1.], dtype=torch.double))
    assert surrogate.argmax() != exact.argmax()


def test_raw_geometry_guard_can_pass_when_all_geometry_is_discarded():
    solver = DownstreamTangentSparseGPT(nn.Linear(2, 1, bias=False), alpha=1.)
    solver.add_ambient_batch(torch.tensor([[1., 0.]]))
    solver.add_pullback_batch(torch.tensor([[0., 1.]]), torch.ones(1, 1), collect_ambient=False)
    # This is the experiment driver's pre-finalization acceptance condition.
    assert float(solver.geometry_energy.sum()) > 0
    solver.finalize_tangents()
    assert solver.tangent_diagnostics['geometry_trace'] == 0
    assert solver.tangent_diagnostics['gamma'] == 0
    torch.testing.assert_close(solver.scale, torch.ones(1, 2))


def test_separate_budgets_keep_requested_supported_trace_ratio():
    solver = DownstreamTangentSparseGPT(nn.Linear(2, 2, bias=False), alpha=.7)
    solver.add_ambient_batch(torch.tensor([[1., 2.], [3., 4.], [2., 1.]]))
    solver.add_pullback_batch(torch.tensor([[2., 3.]]), torch.tensor([[1., 4.]]), collect_ambient=False)
    solver.finalize_tangents()
    added = ((solver.scale.double().square() - 1) * solver.input_energy).sum()
    assert float(added / (2 * solver.input_energy.sum())) == pytest.approx(.7, abs=2e-7)


@pytest.mark.parametrize('anchor_type', ['linear_attention', 'full_attention'])
def test_tiny_qwen_next_block_vjp_matches_weight_finite_difference(anchor_type, monkeypatch):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from sparsifier.eval.gsm8k_downstream import AnchorPrefix

    # Test-only bypass: current helper misses kernels.layer Func wrappers.
    # Never execute optional FLA/Triton kernels in this CPU audit.
    import inspect
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    for name in _PYTORCH_ATTENTION:
        function = getattr(modeling, name)
        if isinstance(function, nn.Module):
            function = inspect.getclosurevars(function.forward).nonlocals['func']
        monkeypatch.setattr(modeling, name, inspect.unwrap(function))
    torch.manual_seed(41)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        layer_types=['linear_attention', anchor_type],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .5},
    )
    config._attn_implementation = 'sdpa'
    model = Qwen3_5TextModel(config).eval().requires_grad_(False)
    target, anchor = model.layers[0].mlp.down_proj, model.layers[1]
    prefix = AnchorPrefix(model, anchor)
    tokens = torch.tensor([[1, 4, 7, 2]])
    captured = {}

    def target_hook(module, args, output):
        captured['x'] = args[0].detach()
        output = output.detach().requires_grad_(True)
        captured['y'] = output
        return output

    def anchor_hook(module, args, output):
        captured['h'] = output[0] if isinstance(output, tuple) else output

    handles = [target.register_forward_hook(target_hook), anchor.register_forward_hook(anchor_hook)]
    try:
        prefix(input_ids=tokens, use_cache=False)
        g, = torch.autograd.grad(captured['h'][0, -1, 0], captured['y'])
        direction = torch.randn_like(target.weight)
        direction /= direction.norm()
        predicted = (g * (captured['x'] @ direction.T)).sum()
        original = target.weight.clone()
        values = []
        eps = 1e-3
        for sign in (1, -1):
            with torch.no_grad():
                target.weight.copy_(original + sign * eps * direction)
            prefix(input_ids=tokens, use_cache=False)
            values.append(captured['h'][0, -1, 0].detach())
        numerical = (values[0] - values[1]) / (2 * eps)
        torch.testing.assert_close(predicted, numerical, atol=2e-5, rtol=.02)
        assert predicted.abs() > 1e-6
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.xfail(strict=True, reason="force_pytorch_attention does not unwrap kernels.layer Func modules")
def test_reference_backend_helper_removes_optional_kernel_dispatch(monkeypatch):
    from sparsifier.models.qwen import force_pytorch_attention, _PYTORCH_ATTENTION
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    force_pytorch_attention()
    assert all(not isinstance(getattr(modeling, name), nn.Module) for name in _PYTORCH_ATTENTION)
