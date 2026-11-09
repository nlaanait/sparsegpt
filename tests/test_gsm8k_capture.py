from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.eval.gsm8k_capture import capture_one, check_competence


class CaptureLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(12, 4)
        self.layer = nn.Linear(4, 6, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.layer(self.embed(input_ids)))


def test_readout_uses_final_causal_position_and_aligned_inputs_outputs():
    torch.manual_seed(4)
    model = CaptureLM()
    tokens = torch.tensor([[1, 3, 5, 7]])
    recorded = capture_one(model, model.layer, tokens, token_samples=2)
    expected_x = model.embed(tokens)[0].detach().numpy()
    np.testing.assert_allclose(recorded["x"], expected_x[-1])
    np.testing.assert_allclose(recorded["token_x"], expected_x[recorded["positions"]])
    np.testing.assert_allclose(recorded["y"], recorded["x"] @ model.layer.weight.detach().numpy().T, atol=1e-6)
    assert int(recorded["readout_position"]) == 3
    assert not model.layer._forward_hooks


def test_capture_hook_restored_on_failure():
    model = CaptureLM()
    def fail(**kwargs):
        raise RuntimeError("failed forward")
    model.forward = fail
    with pytest.raises(RuntimeError):
        capture_one(model, model.layer, torch.tensor([[1]]))
    assert not model.layer._forward_hooks


def test_competence_gate_requires_complete_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"complete": False, "competence_floor_passed": True}))
    with pytest.raises(RuntimeError):
        check_competence(report)
    report.write_text(json.dumps({"complete": True, "competence_floor_passed": True}))
    assert len(check_competence(report)) == 64
