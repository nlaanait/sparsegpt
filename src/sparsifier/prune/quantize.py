"""Quantization steps applied to a weight tensor after the sparse solve.

A step is a function from a weight tensor to a weight tensor. The chain is
empty at 16 bits. Below that, the only step raises ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Quantize = Callable[[torch.Tensor], torch.Tensor]


def _not_implemented(weights: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        f"quantization below 16 bits is not implemented (tensor shape {tuple(weights.shape)})"
    )


def quantization_chain(wbits: int) -> tuple[Quantize, ...]:
    """Steps to run, in order. Empty when ``wbits`` is at least 16."""
    if wbits >= 16:
        return ()
    return (_not_implemented,)


def apply_quantization(weights: torch.Tensor, wbits: int) -> torch.Tensor:
    """Run the chain. Does not catch ``NotImplementedError``."""
    for step in quantization_chain(wbits):
        weights = step(weights)
    return weights
