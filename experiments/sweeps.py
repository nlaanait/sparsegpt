"""Every sweep spec, named so a plan or a figure rewrite can find it."""

from __future__ import annotations

from experiments.gemma import GEMMA
from experiments.llama import LLAMA3, LLAMA3_WIKITEXT2_CAL
from experiments.olmo import OLMO2, OLMO2_WIKITEXT2_CAL
from experiments.protocol import SweepSpec
from experiments.qwen import (
    QWEN_35,
    QWEN_35_WIKITEXT2_CAL,
    QWEN_36_38,
    QWEN_WIKITEXT2_CAL,
)

_SWEEPS = (
    QWEN_35,
    QWEN_35_WIKITEXT2_CAL,
    QWEN_36_38,
    QWEN_WIKITEXT2_CAL,
    GEMMA,
    OLMO2,
    OLMO2_WIKITEXT2_CAL,
    LLAMA3,
    LLAMA3_WIKITEXT2_CAL,
)


def sweeps() -> tuple[SweepSpec, ...]:
    return _SWEEPS
