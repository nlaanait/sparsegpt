"""Independent ordered-OBS references at the observed failure frontier."""
import pytest
import torch

from sparsifier.eval.gsm8k_pruning import prune_candidate
from sparsifier.prune.sparsegpt import SparseGPT


def ordered_reference(weight, inputs, sparsity, blocksize):
    """Use explicit trailing-system solves, without inverse Cholesky updates.

    Masks are selected at each block boundary, as in SparseGPT. Once a
    coordinate is committed, its error is optimally redistributed over the
    remaining coordinates by solving the quadratic normal equations.
    """
    h = inputs.double().T @ inputs.double()
    h += .01 * h.diag().mean() * torch.eye(h.shape[0], dtype=h.dtype)
    w = weight.double().clone()
    for start in range(0, w.shape[1], blocksize):
        stop = min(start + blocksize, w.shape[1])
        variances = torch.stack([
            torch.linalg.inv(h[j:, j:])[0, 0] for j in range(start, stop)
        ])
        scores = w[:, start:stop].square() / variances
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.flatten()[scores.flatten().argsort()[:int(scores.numel()*sparsity)]] = True
        for j in range(start, stop):
            error = w[:, j] * mask[:, j-start]
            w[:, j] -= error
            if j+1 < w.shape[1]:
                redistribution = torch.linalg.solve(h[j+1:, j+1:], h[j+1:, j])
                w[:, j+1:] += error[:, None] * redistribution
    return w.float()


@pytest.mark.parametrize("sparsity", [.5, .7, .8, .9, .99])
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_high_sparsity_compensation_matches_explicit_normal_equations(sparsity, rank_deficient):
    torch.manual_seed(701)
    weight = torch.randn(11, 24)
    inputs = torch.randn(12 if rank_deficient else 80, 24)
    layer = torch.nn.Linear(24, 11, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    solver = SparseGPT(layer)
    solver.add_batch(inputs)
    solver.fasterprune(sparsity, blocksize=8, mask_weight=torch.ones_like(weight))
    expected = ordered_reference(weight, inputs, sparsity, 8)
    torch.testing.assert_close(layer.weight, expected, rtol=1e-3, atol=2e-5)
    assert torch.equal(layer.weight == 0, expected == 0)
    assert (layer.weight == 0).sum() == 3 * int(11*8*sparsity)


@pytest.mark.parametrize("sparsity", [.7, .8, .9, .99])
def test_geometry_alpha_zero_has_exact_high_sparsity_ambient_parity(sparsity):
    torch.manual_seed(702)
    weight, inputs = torch.randn(11, 24), torch.randn(12, 24)
    baseline, _ = prune_candidate(weight, inputs, arm="sparsegpt_task",
                                   sparsity=sparsity, device="cpu")
    geometry, diagnostics = prune_candidate(
        weight, inputs, arm="local", sparsity=sparsity, alpha=0,
        readout_inputs=inputs[:4], pdiag=torch.ones(4, 11), device="cpu")
    torch.testing.assert_close(geometry, baseline, rtol=0, atol=0)
    assert diagnostics["added_curvature_trace_ratio"] == 0
