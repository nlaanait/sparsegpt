import pytest
import torch

from sparsifier.prune.sample_metric import quadratic_hessian, refit, objective


def test_quadratic_matches_direct_sample_dependent_loss_and_gradient():
    torch.manual_seed(1)
    x, z = torch.randn(21, 4, dtype=torch.double), torch.randn(7, 4, dtype=torch.double)
    q = torch.randn(7, 3, 1, dtype=torch.double)
    q = q / q.norm(dim=1, keepdim=True)
    metrics = q @ q.transpose(1, 2)
    gamma, damping = 2., .03
    h = quadratic_hessian(x, z, metrics, gamma, damping)
    a = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    r = z @ a.T
    direct = (x @ a.T).square().sum()/len(x) + gamma*torch.einsum('na,nab,nb->',r,metrics,r)/len(x) + damping*a.square().sum()
    quadratic = a.flatten() @ h @ a.flatten()
    torch.testing.assert_close(quadratic, direct)
    torch.testing.assert_close(torch.autograd.grad(direct, a)[0].flatten(), 2*h@a.detach().flatten())


def test_exact_refit_has_stationary_free_entries_and_beats_feasible_candidates():
    torch.manual_seed(2)
    w = torch.randn(3, 4, dtype=torch.double)
    x, z = torch.randn(30, 4), torch.randn(8, 4)
    q = torch.randn(8, 3, 1)
    h = quadratic_hessian(x, z, q @ q.transpose(1, 2), 1, .01)
    keep = torch.rand_like(w) > .5
    best = refit(w, keep, h)
    assert (best[~keep] == 0).all()
    gradient = h @ (w-best).flatten()
    torch.testing.assert_close(gradient[keep.flatten()], torch.zeros_like(gradient[keep.flatten()]), atol=1e-10, rtol=0)
    assert objective(w, best, h) <= objective(w, w*keep, h)
    assert objective(w, best, h) <= objective(w, (w+torch.randn_like(w))*keep, h)


def test_negative_metric_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        quadratic_hessian(torch.ones(3,2), torch.ones(1,2), -torch.eye(2)[None], 1)
