"""Independent contracts from tangent-pruning-methods sections 5.1–5.2."""
import numpy as np
import pytest
import torch
from sparsifier.eval.gsm8k_pruning import prune_candidate
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION, local_row_scales, row_scales


def test_local_zero_and_compensation_match_regularized_surrogate():
    w=torch.tensor([[.001,2.],[3.,4.]])
    x=torch.tensor([[1.,.2],[.3,2.],[1.,1.],[-.5,1.]])
    pd=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    baseline,_=prune_candidate(w,x,arm='sparsegpt_task',sparsity=.25,device='cpu')
    zero,_=prune_candidate(w,x,arm='local',sparsity=.25,alpha=0,readout_inputs=x,pdiag=pd,device='cpu')
    torch.testing.assert_close(zero,baseline,rtol=0,atol=0)
    got,d=prune_candidate(w,x,arm='local',sparsity=.25,alpha=1,readout_inputs=x,pdiag=pd,device='cpu')
    h=x.T@x/len(x);h=h+.01*h.diag().mean()*torch.eye(2)
    coupling=torch.tensor(pd,dtype=x.dtype).T@x.square()/x.square().sum(0)
    scales=(1+d['gamma']*coupling).sqrt()
    effective=torch.diag(scales[0])@h@torch.diag(scales[0])
    expected=w.clone();expected[0,0]=0
    expected[0,1]+=w[0,0]*effective[1,0]/effective[1,1]
    torch.testing.assert_close(got,expected,rtol=1e-5,atol=1e-6)


def test_general_shared_curvature_matches_documented_diagonal():
    h0=torch.tensor([2.,8.]);hp=torch.tensor([4.,4.]);g=torch.tensor([[.2,.4]])
    actual=row_scales(g,3.,h0_diag=h0,hprime_diag=hp)
    expected=((h0/hp)[None,:]*(1+3*g)).sqrt()
    torch.testing.assert_close(actual,expected)


def test_alpha_zero_matches_local_baseline():
    torch.manual_seed(12)
    w, x = torch.randn(3, 4), torch.randn(20, 4)
    baseline, _ = prune_candidate(w, x, arm="sparsegpt_task", device="cpu")
    got, diag = prune_candidate(w, x, arm="local", alpha=0, readout_inputs=x[:2],
        pdiag=torch.eye(3)[:2], device="cpu")
    torch.testing.assert_close(got, baseline, rtol=0, atol=0)
    assert diag["added_curvature_trace_ratio"] == 0


def test_default_curvature_normalization_matches_independent_quadratic():
    # Geometry protects the high-energy readout only. Rank averaging misses
    # this concentration; compare against the trace of an explicit quadratic.
    from sparsifier.prune.sample_metric import quadratic_hessian
    x = torch.tensor([[10., 1.], [1., 1.], [1., 1.]])
    z = x[:1]
    pd = torch.tensor([[1., 0.]])
    metrics = torch.diag_embed(pd)
    ambient_h = quadratic_hessian(x, z, metrics, 0)
    scale, diag = local_row_scales(x, z, pd, .7)
    h = quadratic_hessian(x, z, metrics, diag["gamma"])
    ratio = float((h - ambient_h).trace() / ambient_h.trace())
    assert diag["normalization"] == LOCAL_PRUNING_VERSION
    assert diag["added_curvature_trace_ratio"] == pytest.approx(ratio)
    assert ratio == pytest.approx(.7)
    torch.testing.assert_close(scale.double().square().flatten() * ambient_h.diag(),
                               h.diag(), rtol=1e-6, atol=1e-6)


def test_trace_scales_preserve_pairing_and_ignore_uniform_activation_units():
    x = torch.tensor([[4., 1.], [1., 3.], [1., 1.]])
    pd = torch.eye(2)
    scale, diag = local_row_scales(x, x[:2], pd, 1)
    units, _ = local_row_scales(x * 7, x[:2] * 7, pd, 1)
    torch.testing.assert_close(scale, units)
    shuffled, other = local_row_scales(x, x[:2], pd.flip(0), 1)
    assert not torch.allclose(scale, shuffled)
    assert diag["added_curvature_trace_ratio"] == pytest.approx(other["added_curvature_trace_ratio"])


def test_trace_normalization_reports_unsupported_and_zero_energy():
    x = torch.tensor([[1., 0.], [2., 0.]])
    scale, diag = local_row_scales(x, torch.tensor([[1., 3.]]), torch.ones(1, 2), 1)
    torch.testing.assert_close(scale[:, 1], torch.ones(2))
    assert diag["unsupported_input_coordinates"] == 1
    assert diag["excluded_geometric_trace_sum"] == 18
    assert diag["added_curvature_trace_ratio"] == pytest.approx(1)
    scale, diag = local_row_scales(x, torch.zeros(1, 2), torch.ones(1, 2), 1)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    assert diag["gamma"] == 0
    assert diag["zero_geometric_trace"]


def test_default_candidate_matches_independent_trace_normalized_solve():
    from sparsifier.prune.sparsegpt import SparseGPT
    torch.manual_seed(52)
    w, x = torch.randn(3, 4), torch.randn(20, 4)
    z, pd = x[:5], torch.softmax(torch.randn(5, 3), dim=1)
    # Compute gamma from explicit weight-curvature traces rather than ranks.
    geo = pd.T @ z.square()
    gamma = .7 * w.shape[0] * float(x.square().sum()) / float(geo.sum())
    scale = (1 + gamma * (geo / x.square().sum(0)[None, :])).sqrt()
    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(w * scale)
    solver = SparseGPT(layer)
    solver.add_batch(x)
    solver.fasterprune(.5, percdamp=.01, blocksize=128, mask_weight=torch.ones_like(w))
    expected = layer.weight.detach() / scale
    got, _ = prune_candidate(w, x, arm="local", alpha=.7,
                             readout_inputs=z, pdiag=pd, device="cpu")
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)
