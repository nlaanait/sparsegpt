"""Model table and the family loops.

``catalog`` is the registered checkpoints. Qwen, OLMo, and Llama implement
``DecoderFamily``. Gemma implements ``OwnLoopFamily``. wandb is optional;
the runs call ``init_wandb`` and ``log_perplexity``.
"""

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from sparsifier.models.catalog import ModelRecord, models, resolve, stored_count

has_wandb = _wandb is not None


def init_wandb(config) -> None:
    """Start a run when ``config.log_wandb`` is set."""
    if not getattr(config, "log_wandb", False):
        return
    if _wandb is None:
        raise RuntimeError("wandb is not installed")
    payload = vars(config) if hasattr(config, "__dict__") else config
    _wandb.init(config=payload)


def log_perplexity(dataset: str, perplexity: float, enabled: bool) -> None:
    """Record one dataset's perplexity. No-op when logging is off."""
    if not enabled:
        return
    if _wandb is None:
        raise RuntimeError("wandb is not installed")
    _wandb.log({f"{dataset}/perplexity": perplexity})


__all__ = [
    "ModelRecord",
    "has_wandb",
    "init_wandb",
    "log_perplexity",
    "models",
    "resolve",
    "stored_count",
]
