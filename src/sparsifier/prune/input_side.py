"""Revised note §3.3 / §4.4: Variant A input-side Hessian augmentation.

Paired concept differences add a second shared Hessian. SparseGPT itself
is unchanged after ``H ← H + γ_in H_Δ``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sparsifier.prune.sparsegpt import SparseGPT


def difference_covariance(
    deltas: torch.Tensor,
    knob_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """H_Δ = (1/K) Σ δφ_k δφ_k^T, optionally trace-normalized per knob."""
    d = deltas.float()
    if d.ndim != 2 or d.shape[0] < 1 or d.shape[1] < 1:
        raise ValueError("deltas must have shape [pairs, input_dim]")
    if knob_ids is None:
        return d.T @ d / d.shape[0]
    ids = torch.as_tensor(knob_ids).reshape(-1)
    if ids.shape[0] != d.shape[0]:
        raise ValueError("knob_ids must have one entry per pair")
    covs: list[torch.Tensor] = []
    for knob in ids.unique(sorted=True):
        chunk = d[ids == knob]
        if chunk.shape[0] == 0:
            continue
        cov = chunk.T @ chunk / chunk.shape[0]
        trace = torch.trace(cov)
        if float(trace) <= 0.0:
            continue
        covs.append(cov / trace)
    if not covs:
        raise ValueError("all knobs have zero difference energy")
    return torch.stack(covs, dim=0).mean(dim=0)


def pair_mix_scale(
    hessian: torch.Tensor,
    h_delta: torch.Tensor,
    alpha_in: float,
) -> float:
    """γ_in = α_in tr(H) / tr(H_Δ). α_in is the difference term's curvature share."""
    if not math.isfinite(alpha_in) or alpha_in < 0.0:
        raise ValueError("alpha_in must be finite and nonnegative")
    if alpha_in == 0.0:
        return 0.0
    trace_h = float(torch.trace(hessian))
    trace_delta = float(torch.trace(h_delta))
    if not math.isfinite(trace_delta) or trace_delta <= 0.0:
        raise ValueError("H_delta must have positive trace")
    return float(alpha_in) * trace_h / trace_delta


def augment_hessian(
    hessian: torch.Tensor,
    deltas: torch.Tensor,
    alpha_in: float,
    knob_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """H' = H + γ_in H_Δ. α_in = 0 leaves H unchanged."""
    if not math.isfinite(alpha_in) or alpha_in < 0.0:
        raise ValueError("alpha_in must be finite and nonnegative")
    if alpha_in == 0.0:
        return hessian
    h_delta = difference_covariance(deltas, knob_ids=knob_ids)
    gamma_in = pair_mix_scale(hessian, h_delta, alpha_in)
    return hessian + gamma_in * h_delta.to(device=hessian.device, dtype=hessian.dtype)


class InputSideSparseGPT(SparseGPT):
    """Ambient SparseGPT Hessian plus paired-input difference curvature."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__(layer)
        self._pair_chunks: list[torch.Tensor] = []
        self._knob_chunks: list[torch.Tensor] = []
        self.pairs_finalized = False

    def add_pairs(
        self,
        deltas: torch.Tensor,
        knob_ids: torch.Tensor | None = None,
    ) -> None:
        if self.pairs_finalized:
            raise RuntimeError("Cannot add pairs after finalize_pairs")
        d = deltas.detach().float()
        if d.ndim != 2 or d.shape[1] != self.columns:
            raise ValueError("deltas must have shape [pairs, input_dim]")
        if d.shape[0] < 1:
            raise ValueError("at least one paired difference is required")
        self._pair_chunks.append(d.cpu())
        if knob_ids is not None:
            ids = torch.as_tensor(knob_ids).reshape(-1)
            if ids.shape[0] != d.shape[0]:
                raise ValueError("knob_ids must have one entry per pair")
            self._knob_chunks.append(ids.cpu())

    def finalize_pairs(self, alpha_in: float) -> None:
        if self.pairs_finalized:
            return
        if not math.isfinite(alpha_in) or alpha_in < 0.0:
            raise ValueError("alpha_in must be finite and nonnegative")
        if alpha_in == 0.0:
            self.pairs_finalized = True
            return
        if not self._pair_chunks:
            raise RuntimeError("Collect paired input differences before finalize_pairs")
        deltas = torch.cat(self._pair_chunks, dim=0)
        knob_ids = None
        if self._knob_chunks:
            if len(self._knob_chunks) != len(self._pair_chunks):
                raise RuntimeError("knob_ids must be supplied for every add_pairs call")
            knob_ids = torch.cat(self._knob_chunks, dim=0)
        self.H = augment_hessian(
            self.H, deltas.to(device=self.H.device), alpha_in, knob_ids=knob_ids
        )
        self.pairs_finalized = True

    def fasterprune(self, *args, **kwargs) -> None:
        if not self.pairs_finalized:
            raise RuntimeError("Call finalize_pairs before pruning")
        super().fasterprune(*args, **kwargs)

    def free(self) -> None:
        self._pair_chunks = []
        self._knob_chunks = []
        super().free()
