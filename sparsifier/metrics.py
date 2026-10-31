"""Scores computed from a summed negative log-likelihood.

Perplexity is the metric the sweeps record today. Bits per byte uses the
same nats once an eval also counts bytes. The historical log line for
perplexity stays `Perplexity:` so `experiments.protocol.PPL_RE` still matches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Likelihood:
    """Nats already reduced the way the current eval loops reduce them.

    ``nats`` is the sum of per-window ``loss * seqlen`` values. ``tokens`` is
    ``nsamples * seqlen``, the denominator in ``exp(nats / tokens)``. That is
    the printed perplexity, including the off-by-one from shifted labels.
    ``bytes`` is unset until an eval counts UTF-8 bytes of the same windows.
    """

    nats: float
    tokens: int
    bytes: int | None = None

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("tokens must be positive")
        if not math.isfinite(self.nats):
            raise ValueError("nats must be finite")
        if self.bytes is not None and self.bytes <= 0:
            raise ValueError("bytes must be positive when set")


@dataclass(frozen=True)
class Score:
    name: str
    value: float


class Metric(Protocol):
    name: str

    def score(self, likelihood: Likelihood) -> Score: ...


class Perplexity:
    name = "perplexity"

    def score(self, likelihood: Likelihood) -> Score:
        return Score(self.name, math.exp(likelihood.nats / likelihood.tokens))


class BitsPerByte:
    """Nats converted to bits, divided by the byte length of the windows."""

    name = "bits_per_byte"

    def score(self, likelihood: Likelihood) -> Score:
        if likelihood.bytes is None:
            raise ValueError("bits per byte needs likelihood.bytes")
        bits = likelihood.nats / math.log(2)
        return Score(self.name, bits / likelihood.bytes)


_METRICS: dict[str, Metric] = {
    Perplexity.name: Perplexity(),
    BitsPerByte.name: BitsPerByte(),
}


def get_metric(name: str) -> Metric:
    try:
        return _METRICS[name]
    except KeyError:
        known = ", ".join(sorted(_METRICS))
        raise KeyError(f"unknown metric {name!r}; known: {known}") from None


def format_result(score: Score) -> str:
    """One log line. Perplexity keeps the spelling the sweep parser expects."""
    if score.name == Perplexity.name:
        return f"Perplexity: {score.value:3f}"
    return f"{score.name}: {score.value:.6f}"
