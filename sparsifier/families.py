"""Family protocols for the layer loop, once it moves out of ``sparsegpt``.

A decoder family supplies capture, the layer forward, and which linears are
prunable. A family whose block is not that shape implements ``prune_model``
and ``likelihood`` itself. Gemma is the second kind. Model lookup is
``sparsifier.catalog.resolve``. Nothing here is wired to a loop yet.
"""

from __future__ import annotations

from typing import Protocol

from sparsifier.metrics import Likelihood
from sparsifier.pruners import Pruner
from sparsifier.run import RunConfig


class DecoderFamily(Protocol):
    name: str

    def matches(self, model_id: str) -> bool: ...

    def load(self, model_id: str): ...

    def prepare(self, model) -> None: ...

    def capture(self, model, batches, config: RunConfig): ...

    def forward_layer(self, layer, hidden, captured): ...

    def prunable(self, layer) -> dict: ...

    def likelihood(self, model, encoded, config: RunConfig) -> Likelihood: ...


class OwnLoopFamily(Protocol):
    """Owns the layer loop. The shared run only loads, caches, and scores."""

    name: str

    def matches(self, model_id: str) -> bool: ...

    def load(self, model_id: str): ...

    def prune_model(self, model, batches, config: RunConfig, pruner: Pruner) -> None: ...

    def likelihood(self, model, encoded, config: RunConfig) -> Likelihood: ...
