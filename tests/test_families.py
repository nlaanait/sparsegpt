"""The four measurement families implement the layer-loop protocols."""

from __future__ import annotations

from sparsifier.models.families import DecoderFamily, OwnLoopFamily
from sparsifier.models.gemma import GEMMA
from sparsifier.models.llama import LLAMA
from sparsifier.models.olmo import OLMO
from sparsifier.models.qwen import QWEN

_DECODER = (
    "matches",
    "load",
    "prepare",
    "capture",
    "forward_layer",
    "prunable",
    "sequential_groups",
    "likelihood",
)
_OWN = ("matches", "load", "prune_model", "likelihood")


def test_decoder_families_implement_the_protocol():
    for family, model_id in (
        (QWEN, "Qwen/Qwen3.5-2B"),
        (OLMO, "allenai/OLMo-2-1124-7B-Instruct"),
        (LLAMA, "meta-llama/Llama-3.1-8B"),
    ):
        assert isinstance(family, DecoderFamily)
        assert family.matches(model_id)
        assert not family.matches("google/gemma-4-E2B")
        for name in _DECODER:
            assert callable(getattr(family, name))


def test_gemma_owns_its_loop():
    assert isinstance(GEMMA, OwnLoopFamily)
    assert GEMMA.matches("google/gemma-4-E2B")
    assert not GEMMA.matches("Qwen/Qwen3.5-2B")
    for name in _OWN:
        assert callable(getattr(GEMMA, name))
