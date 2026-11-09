from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from sparsifier.eval.gsm8k_sensitivity import answer_gradient, balanced_split


class TinyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.layer = nn.Linear(4, 6, bias=False)
        self.head = nn.Linear(6, 11, bias=False)
        self.requires_grad_(False)

    def forward(self, input_ids, **kwargs):
        y = self.layer(self.embed(input_ids))
        logits = self.head(y.cumsum(1))
        return SimpleNamespace(logits=logits[:, -kwargs['logits_to_keep']:])


def test_aligned_answer_gradient_matches_explicit_autograd():
    torch.manual_seed(5)
    model = TinyCausal()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    x, g, loss = answer_gradient(model, model.layer, tokens, 3)
    expected_x = model.embed(tokens)
    y = model.layer(expected_x).detach().requires_grad_(True)
    logits = model.head(y.cumsum(1))
    expected_loss = F.cross_entropy(logits[0, 2:4], tokens[0, 3:])
    expected_g = torch.autograd.grad(expected_loss, y)[0][0, 2]
    torch.testing.assert_close(x, expected_x[0, 2])
    torch.testing.assert_close(g, expected_g)
    assert not model.layer._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_selector_preserves_split_and_balances_families():
    rows = [{"id": f"{s}-{f}-{i}", "family": f, "split": s}
            for s in ("pruning_calibration", "stage2_holdout") for f in ("a", "b") for i in range(5)]
    selected = balanced_split(rows, "pruning_calibration", 4)
    assert all(r["split"] == "pruning_calibration" for r in selected)
    assert sum(r["family"] == "a" for r in selected) == 2
