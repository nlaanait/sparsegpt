"""Batched Qwen layer forwards match the one-window loop, and the SM121 kernel binds."""

from __future__ import annotations

import torch
import torch.nn as nn

from sparsegpt.qwen import (
    _enable_gdn_kernel,
    _leading_example,
    _run_layer,
    _run_layer_batched,
    forward_batch_size,
    gdn_backend,
)
from sparsegpt.sparsegpt import SparseGPT


def test_forward_batch_size_fits_27b_and_keeps_small_models_whole():
    # Qwen3.6-27B bf16: two (2048, 17408) activations per window.
    assert forward_batch_size(128, 2048, 17408, 2) == 30
    # Qwen3.5-0.8B bf16 activations are small enough for the whole calibration set.
    assert forward_batch_size(128, 2048, 3584, 2) == 128
    assert forward_batch_size(1, 2048, 17408, 2) == 1


def test_leading_example_slices_batch_not_mrope_axis():
    rope = torch.zeros(8, 4, 16)
    # Batch 4 is the Qwen3.5-9B C4 remainder (256 % 42). Float RoPE must keep
    # the batch-first slice; only integer mRoPE position ids use axis 0.
    rope_batch4 = torch.zeros(4, 2048, 64)
    ids = torch.zeros(4, 8, 4, dtype=torch.long)
    cos, sin = _leading_example((rope, rope))
    assert cos.shape == (1, 4, 16)
    assert sin.shape == (1, 4, 16)
    assert _leading_example(rope_batch4).shape == (1, 2048, 64)
    assert _leading_example(ids).shape == (4, 1, 4)


def test_chunked_hessian_matches_one_window_at_a_time():
    layer = nn.Linear(8, 4, bias=False)
    online = SparseGPT(layer)
    chunked = SparseGPT(layer)
    windows = [torch.randn(1, 16, 8) for _ in range(4)]
    for window in windows:
        online.add_batch(window, window)
    chunked.add_batch(torch.cat(windows[:2], dim=0), torch.cat(windows[:2], dim=0))
    chunked.add_batch(torch.cat(windows[2:], dim=0), torch.cat(windows[2:], dim=0))
    assert torch.allclose(online.H, chunked.H)


def _text_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "/home/nlaanait/sparsegpt/sparse_opt/qwen3.5-0.8b",
        trust_remote_code=True,
    )
    return getattr(config, "text_config", config)


def _layer_inputs(config, batch, seqlen, device):
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    hidden = torch.randn(batch, seqlen, config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seqlen, device=device).view(1, 1, -1).expand(3, 1, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config).to(device)
    position_embeddings = rotary(hidden[:1], position_ids)
    text_position_ids = torch.arange(seqlen, device=device).view(1, -1)
    mask_kwargs = {
        "config": config,
        "inputs_embeds": hidden[:1],
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }
    return hidden, masks, position_embeddings, text_position_ids


def _assert_batched_matches(
    layer, hidden, mask, position_embeddings, position_ids, *, parallel=False, atol=0, rtol=0
):
    layer.eval()
    single = torch.empty_like(hidden)
    with torch.no_grad():
        for row in range(hidden.shape[0]):
            single[row : row + 1] = _run_layer(
                layer,
                hidden[row : row + 1],
                attention_mask=mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
        batched = torch.empty_like(hidden)
        _run_layer_batched(
            layer,
            hidden,
            batched,
            attention_mask=mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            batch=hidden.shape[0],
            parallel=parallel,
        )
    torch.testing.assert_close(batched, single, atol=atol, rtol=rtol)


def test_batched_layer_matches_one_window_on_cuda():
    if not torch.cuda.is_available():
        return
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    config = _text_config()
    device = "cuda"
    hidden, masks, position_embeddings, position_ids = _layer_inputs(config, 4, 32, device)
    for index, kind in ((0, "linear_attention"), (3, "full_attention")):
        assert config.layer_types[index] == kind
        layer = Qwen3_5DecoderLayer(config, index).to(device=device, dtype=torch.bfloat16)
        _assert_batched_matches(
            layer,
            hidden,
            masks[kind],
            position_embeddings,
            position_ids,
            atol=2e-2,
            rtol=1e-2,
        )


def test_gdn_kernel_binds_on_sm121_and_runs_a_batch():
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor != 121:
        return
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = _text_config()
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    config.vocab_size = 128
    model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    before = gdn_backend(model)
    _enable_gdn_kernel(model)
    after = gdn_backend(model)
    assert "gdn" in after, f"Gated DeltaNet forward was not the Atlas kernel ({after})"
    assert next(model.parameters()).device.type == "cpu"

    device = "cuda"
    layer = model.model.layers[0].to(device)
    hidden, masks, position_embeddings, position_ids = _layer_inputs(config, 4, 32, device)
    _assert_batched_matches(
        layer,
        hidden,
        masks["linear_attention"],
        position_embeddings,
        position_ids,
        parallel=True,
    )
    with torch.no_grad():
        correct = torch.empty_like(hidden)
        for row in range(hidden.shape[0]):
            correct[row : row + 1] = _run_layer(
                layer,
                hidden[row : row + 1],
                attention_mask=masks["linear_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
        leaked = _run_layer(
            layer,
            hidden,
            attention_mask=masks["linear_attention"],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
    row0 = (leaked[:1] - correct[:1]).abs().max()
    later = (leaked[1:] - correct[1:]).abs().max()
    assert row0 < 0.05
    assert later > 0.05, "gdn_prefill no longer leaks later rows; a real batch is safe"
