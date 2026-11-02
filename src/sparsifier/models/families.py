"""Family protocols for the layer loop, once it moves out of ``sparsegpt``.

A decoder family supplies capture, the layer forward, and which linears are
prunable. A family whose block is not that shape implements ``prune_model``
and ``likelihood`` itself. Gemma is the second kind. Model lookup is
``sparsifier.models.catalog.resolve``. Nothing here is wired to a loop yet.
"""

from __future__ import annotations

from typing import Protocol

from sparsifier.metrics import Likelihood
from sparsifier.prune import Pruner
from sparsifier.sweep import RunConfig


class DecoderFamily(Protocol):
    """Shared layer loop: capture inputs, forward one block, score."""

    name: str

    def matches(self, model_id: str) -> bool:
        """Whether this family owns ``model_id``."""

    def load(self, model_id: str):
        """Load the checkpoint. The caller owns the returned model."""

    def prepare(self, model) -> None:
        """Eval mode and any family-specific flags before capture."""

    def capture(self, model, batches, config: RunConfig):
        """Calibration activations for each prunable linear."""

    def forward_layer(self, layer, hidden, captured):
        """One decoder block. Returns the next hidden state."""

    def prunable(self, layer) -> dict:
        """Name to linear module for the linears SparseGPT may touch."""

    def likelihood(self, model, encoded, config: RunConfig) -> Likelihood:
        """Summed negative log-likelihood on ``encoded``."""


class OwnLoopFamily(Protocol):
    """Owns the layer loop. The shared run only loads, caches, and scores."""

    name: str

    def matches(self, model_id: str) -> bool:
        """Whether this family owns ``model_id``."""

    def load(self, model_id: str):
        """Load the checkpoint. The caller owns the returned model."""

    def prune_model(self, model, batches, config: RunConfig, pruner: Pruner) -> None:
        """Prune in place. The family walks its own blocks."""

    def likelihood(self, model, encoded, config: RunConfig) -> Likelihood:
        """Summed negative log-likelihood on ``encoded``."""
