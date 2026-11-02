"""Model table and family protocols.

``catalog`` is the registered checkpoints. ``families`` is the layer-loop
protocol once loops leave ``sparsegpt``.
"""

from sparsifier.models.catalog import ModelRecord, models, resolve, stored_count

__all__ = [
    "ModelRecord",
    "models",
    "resolve",
    "stored_count",
]
