"""Held-out tangent validation stages (note §6)."""

# Install new-process defaults without changing the v1 source file used by
# workers that are already running and checking their saved answer hashes.
from sparsifier.eval import gsm8k_batch_defaults as _gsm8k_batch_defaults

from sparsifier.eval.tangent_stages import run_stages, summarize_stages

__all__ = ["run_stages", "summarize_stages"]
