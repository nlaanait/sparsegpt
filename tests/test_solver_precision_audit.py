"""Validate the precision reference before using it on full recorded matrices."""
import pytest
import torch

from scripts.audit_gsm8k_solver_precision import source_run,reference_run,reconstruction


@pytest.mark.parametrize('sparsity',[.5,.7])
@pytest.mark.parametrize('rank_deficient',[False,True])
def test_fp64_reference_matches_ordered_fp32_on_supported_small_system(sparsity,rank_deficient):
    torch.manual_seed(708)
    weight = torch.randn(11,24)
    x = torch.randn(12 if rank_deficient else 80,24)
    source,sd = source_run(weight,x,sparsity,blocksize=8)
    reference,rd = reference_run(weight,x,sparsity,blocksize=8)
    torch.testing.assert_close(source.double(),reference,rtol=1e-3,atol=2e-5)
    assert torch.equal(source==0,reference==0)
    assert sd['inverse_probe_residual']['max']<1e-3
    assert rd['inverse_probe_residual']['max']<1e-10
    assert (source==0).sum()==3*int(11*8*sparsity)


def test_common_objective_matches_explicit_damped_quadratic():
    torch.manual_seed(709)
    w,c,x = torch.randn(3,4),torch.randn(3,4),torch.randn(10,4)
    result = reconstruction(w,c,x,.03)
    delta = w.double()-c.double()
    h = x.double().T @ x.double()/len(x)+.03*torch.eye(4,dtype=torch.double)
    expected = .5*torch.einsum('ij,jk,ik->',delta,h,delta)
    assert result['common_scaled_surrogate_half_quadratic']==pytest.approx(float(expected),rel=1e-12)
