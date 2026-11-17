import pytest
import torch

from sparsifier.prune.downstream import _weight_probe_diagonal_, tangent_pullback_diagonal


@pytest.mark.parametrize('chunk', [1, 3, 64])
def test_chunked_probe_preserves_unsquared_factor_and_diagonal(chunk):
    torch.manual_seed(10)
    g, x = torch.randn(5, 7, dtype=torch.double), torch.randn(5, 11, dtype=torch.double)
    expected = g.T@x
    moment = torch.ones_like(expected)
    factor = _weight_probe_diagonal_(g, x, moment, consumer=True, chunk_rows=chunk)
    torch.testing.assert_close(factor, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(moment, 1+expected.square(), atol=1e-12, rtol=1e-12)
    assert not factor.requires_grad


def test_invalid_scratch_and_chunk_settings_fail_before_contraction():
    g, x = torch.ones(2, 1), torch.ones(2, 3)
    with pytest.raises(ValueError, match='positive integer'):
        _weight_probe_diagonal_(g, x, None, chunk_rows=0)
    with pytest.raises(ValueError, match='FP64 moment'):
        _weight_probe_diagonal_(g, x, torch.zeros(1, 3))
    with pytest.raises(ValueError, match='matching sequence'):
        _weight_probe_diagonal_(g, torch.ones(3, 3), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('consume', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float64])
def test_cuda_pullback_matches_cpu_reference_probes_consumers_and_rng(consume, dtype):
    torch.manual_seed(12)
    y = torch.randn(2, 5, 7, device='cuda', dtype=dtype).requires_grad_(True)
    x = torch.randn(2, 5, 11, device='cuda', dtype=dtype)
    basis = torch.linalg.qr(torch.randn(7, 2, dtype=torch.double)).Q.to(
        device='cuda', dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    def run(backend):
        events = []
        generator = torch.Generator().manual_seed(9)
        result = tangent_pullback_diagonal(y.sin(), basis, {'site': y}, weight_inputs={'site': x},
            num_probes=7, generator=generator, contraction_backend=backend, contraction_chunk_rows=3,
            probe_consumer=events.append if consume else None)['site']
        return result, events, generator.get_state()
    cpu, old, old_rng = run('cpu')
    gpu, new, new_rng = run('auto')
    assert gpu.device.type == 'cpu' and gpu.dtype == torch.float64
    torch.testing.assert_close(gpu, cpu, atol=1e-10, rtol=1e-12)
    assert torch.equal(old_rng, new_rng)
    assert len(new) == len(old) == (14 if consume else 0)
    for a, b in zip(old, new):
        for key in ('gradient', 'inputs', 'contraction'):
            assert b[key].device.type == 'cpu'
            torch.testing.assert_close(a[key], b[key], atol=1e-10, rtol=1e-12)
        assert a['probe'] == b['probe'] and a['sequence'] == b['sequence']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_keeps_token_cancellation_and_independent_sequences():
    y = torch.ones(2, 2, 1, device='cuda', dtype=torch.double, requires_grad=True)
    x = torch.tensor([[[1.], [1.]], [[-1.], [-1.]]], device='cuda', dtype=torch.double)
    result = tangent_pullback_diagonal(y.sum().reshape(1, 1), torch.ones(1, 1, device='cuda'),
        {'site':y}, weight_inputs={'site':x}, num_probes=None)['site']
    torch.testing.assert_close(result, torch.tensor([[8.]], dtype=torch.double))
    y = torch.ones(1, 2, 1, device='cuda', dtype=torch.double, requires_grad=True)
    result = tangent_pullback_diagonal((y[:,0]-y[:,1]), torch.ones(1, 1, device='cuda'),
        {'site':y}, weight_inputs={'site':torch.ones_like(y)}, num_probes=None)['site']
    assert float(result) == 0.
