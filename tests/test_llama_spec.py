"""SweepSpec checks for the Llama 3.1 / 3.2 family (no GPU)."""

from __future__ import annotations

import torch.nn as nn

from analysis.plots import _family_sizes
from experiments.llama import (
    LLAMA3,
    LLAMA3_WIKITEXT2_CAL,
    MODELS,
    PUBLISHED_STORED_B,
    SMOKE_DATASET,
    SMOKE_PPL_RANGE,
    stored_params,
)
from sparsegpt.datautils import uses_legacy_llama_tokenizer
from sparsegpt.llama import is_cross_attention_layer


def test_llama3_is_one_family_with_full_ids():
    assert LLAMA3.name == "llama3"
    assert LLAMA3.models == MODELS
    assert len(MODELS) == 6
    assert len({name.split("/")[-1].lower() for name in MODELS}) == 6
    assert LLAMA3.calibration == "c4"
    assert LLAMA3.eval_datasets == ("wikitext2", "ptb")
    assert LLAMA3.runner == "sparsegpt.llama"
    assert LLAMA3.result_dir.name == "llama3"
    assert SMOKE_DATASET == "wikitext2"
    low, high = SMOKE_PPL_RANGE
    assert 0 < low < high
    assert "meta-llama/Llama-3.2-1B" in MODELS
    assert "meta-llama/Llama-3.1-8B" in MODELS
    assert "meta-llama/Llama-3.2-11B-Vision" in MODELS
    assert "meta-llama/Llama-3.1-70B" in MODELS
    assert "meta-llama/Llama-3.2-90B-Vision" in MODELS
    assert LLAMA3_WIKITEXT2_CAL.models == MODELS
    assert LLAMA3_WIKITEXT2_CAL.calibration == "wikitext2"
    assert LLAMA3_WIKITEXT2_CAL.eval_datasets == ("c4", "ptb")
    assert LLAMA3_WIKITEXT2_CAL.result_dir != LLAMA3.result_dir
    assert _family_sizes(LLAMA3_WIKITEXT2_CAL)[MODELS[0]] == PUBLISHED_STORED_B[MODELS[0]]


def test_stored_params_follow_the_full_ids():
    sizes = stored_params()
    assert list(sizes) == list(MODELS)
    assert sizes == {model: PUBLISHED_STORED_B[model] for model in MODELS}
    ordered = [sizes[model] for model in MODELS]
    assert ordered == sorted(ordered)
    assert _family_sizes(LLAMA3)[MODELS[-1]] == sizes[MODELS[-1]]


def test_llama3_tokenizer_is_not_the_llama2_hack():
    assert uses_legacy_llama_tokenizer("meta-llama/Llama-2-7b-hf")
    assert not uses_legacy_llama_tokenizer("meta-llama/Llama-3.2-1B")
    assert not uses_legacy_llama_tokenizer("meta-llama/Llama-3.2-90B-Vision")
    assert not uses_legacy_llama_tokenizer("allenai/OLMo-2-1124-7B-Instruct")


def test_cross_attention_layers_are_skipped():
    class MllamaCrossAttentionDecoderLayer(nn.Module):
        pass

    class LlamaDecoderLayer(nn.Module):
        pass

    assert is_cross_attention_layer(MllamaCrossAttentionDecoderLayer())
    assert not is_cross_attention_layer(LlamaDecoderLayer())
