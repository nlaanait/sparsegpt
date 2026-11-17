"""Check the diagnostic itself on a real tiny Qwen cache implementation."""
import pytest
import torch

from scripts.diagnose_gsm8k_pruning_frontier import cache_comparison, teacher_forcing
from sparsifier.eval.gsm8k_pruning import prune_candidate


@pytest.mark.parametrize('sparsity', [0., .7])
def test_fixed_history_cache_comparison_includes_same_positions(sparsity, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from sparsifier.eval.gsm8k_batching import reference_backend
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    reference_backend()
    torch.manual_seed(703)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':.5})
    config._attn_implementation = 'sdpa'
    model = modeling.Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    if sparsity:
        for layer in model.model.layers:
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                target = getattr(layer.mlp, name)
                candidate, _ = prune_candidate(target.weight, torch.randn(12, target.in_features),
                    arm='sparsegpt_task', sparsity=sparsity, device='cpu')
                with torch.no_grad():
                    target.weight.copy_(candidate)
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return 'prompt'
        def encode(self, text, **kwargs):
            return [1, 2, 3, 4] if text == 'prompt' else [5, 6, 7, 8]
    row, reference = {'id':'a', 'question':'q'}, {'a':{'text':'answer'}}
    comparison = cache_comparison(model, Tokenizer(), row, reference)
    assert comparison['positions'] == 5
    assert comparison['next_token_agreement'] == 1
    assert comparison['max_logit_difference'] < 1e-6
    scored = teacher_forcing(model, Tokenizer(), [row], reference)[0]
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1,2,3,4,5,6,7,8]]), use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(logits[0,3:7], torch.tensor([5,6,7,8]))
    assert scored['target_tokens'] == 4
    assert scored['nll'] == pytest.approx(float(expected), abs=1e-6)
