"""Contracts for aligned gradients and actual vocabulary-space evaluation."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sparsifier.eval.natural_layer import (
    knockout_metrics, language_metrics, local_metrics, prune, record_aligned, replaced_weight, vocabulary_reference,
)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.layer = nn.Linear(4, 6, bias=False)
        self.head = nn.Linear(6, 11, bias=False)
        self.requires_grad_(False)
        self.layer.weight.requires_grad_(True)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.head(self.layer(self.embed(input_ids))))


def test_recorded_gradients_are_from_the_same_token_forward():
    torch.manual_seed(7)
    model = TinyLM()
    windows = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[8, 5, 2, 9]])]
    phi, gradients = record_aligned(model, model.layer, windows)
    expected_x, expected_g = [], []
    for tokens in windows:
        x = model.embed(tokens)
        y = model.layer(x)
        logits = model.head(y)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 11), tokens[:, 1:].flatten(), reduction='sum')
        g = torch.autograd.grad(loss, y)[0]
        expected_x.append(x[:, :-1].flatten(0, 1))
        expected_g.append(g[:, :-1].flatten(0, 1))
    torch.testing.assert_close(phi, torch.cat(expected_x))
    torch.testing.assert_close(gradients, torch.cat(expected_g))


def test_language_kl_and_nll_use_model_vocabulary_and_restore_weights():
    torch.manual_seed(1)
    model = TinyLM()
    windows = [torch.tensor([[1, 2, 3, 4]])]
    original = model.layer.weight.detach().clone()
    refs = vocabulary_reference(model, windows, 'cpu')
    assert language_metrics(model, windows, refs, 'cpu')['vocabulary_kl']['mean'] == pytest.approx(0.)
    with replaced_weight(model.layer, torch.zeros_like(original)):
        measured = language_metrics(model, windows, refs, 'cpu')
        assert measured['nll']['mean'] == pytest.approx(torch.log(torch.tensor(11.)).item())
        assert measured['vocabulary_kl']['mean'] > 0
    torch.testing.assert_close(model.layer.weight, original)
    with pytest.raises(RuntimeError), replaced_weight(model.layer, torch.zeros_like(original)):
        raise RuntimeError('interrupted evaluation')
    torch.testing.assert_close(model.layer.weight, original)


def test_obs_arms_share_exact_half_sparsity():
    torch.manual_seed(2)
    weight, phi = torch.randn(8, 8), torch.randn(32, 8)
    for method in ('sparsegpt', 'wanda', 'input_side'):
        hat = prune(weight, phi, method=method, deltas=torch.randn(4, 8), labels=[0, 1, 2, 3], alpha=1.)
        assert int((hat == 0).sum()) == 32


def test_rank_one_metrics_match_full_residual_metrics():
    torch.manual_seed(3)
    weight, phi = torch.randn(6, 4), torch.randn(10, 4)
    concepts = {'deltas': torch.randn(9, 4), 'presence': torch.randn(9, 4), 'labels': list(range(9))}
    basis, _ = torch.linalg.qr(torch.randn(6, 2), mode='reduced')
    delta = torch.randn(4)
    hat = weight.clone()
    hat[2] -= delta
    base = {field: (concepts[field] @ weight.T).square().sum() for field in ('deltas', 'presence')}
    fast = knockout_metrics(delta, 2, phi, concepts, basis, base)
    full = local_metrics(weight, hat, phi, concepts, basis)
    assert fast['ambient_mse'] == pytest.approx(full['ambient_mse'], rel=1e-6)
    assert fast['heldout_span_mse'] == pytest.approx(full['heldout_span_mse'], rel=1e-6)
    for field in ('deltas', 'presence'):
        assert fast[field]['mse'] == pytest.approx(full[field]['mse'], rel=1e-6)
        assert fast[field]['relative_energy'] == pytest.approx(full[field]['relative_energy'], rel=1e-6)
