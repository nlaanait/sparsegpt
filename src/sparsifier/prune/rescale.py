"""Revised note §4.5: Variant B rescaling trick.

Approximate each row Hessian by a congruence of the shared Hessian that
matches its diagonal, then run unmodified SparseGPT on ``W ⊙ Λ``.
"""

from __future__ import annotations

import math

import torch
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix


LOCAL_PRUNING_VERSION = "curvature-trace-v2"


def local_row_scales(inputs, readout_inputs, pdiag, alpha):
    """Pair readout projector diagonals with input squares before normalizing.

    Both sums use the ambient token count as the averaging denominator; the
    output metric is zero away from readouts. Alpha matches supported
    weight-curvature traces, as in the downstream accumulator. Unsupported features keep unit
    scale and their excluded geometric energy is reported.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    x = inputs.float()
    z = readout_inputs.to(device=x.device, dtype=torch.float32)
    pd = torch.as_tensor(pdiag, device=x.device, dtype=torch.float32)
    if (x.ndim != 2 or z.ndim != 2 or pd.ndim != 2 or not len(x)
            or not len(z) or not x.shape[1] or not pd.shape[1]
            or z.shape[1] != x.shape[1] or len(z) != len(pd)):
        raise ValueError("expected ambient/readout inputs and paired projector diagonals")
    if not all(torch.isfinite(t).all() for t in (x, z, pd)) or (pd < 0).any():
        raise ValueError("inputs and nonnegative projector diagonals must be finite")
    ranks = pd.sum(1)
    rank_sum = float(pd.sum())
    if rank_sum <= 0:
        raise ValueError("no supported local geometry; refuse geometric arm")
    ambient = x.square().sum(0)
    support = ambient > 1e-12
    geo = pd.T @ z.square()
    ambient_trace = pd.shape[1] * float(ambient[support].sum())
    geo_trace = float(geo[:, support].sum())
    if ambient_trace <= 0:
        raise ValueError("no supported ambient curvature")
    gamma = alpha * ambient_trace / geo_trace if geo_trace > 0 else 0.0
    scale = row_scales(geo / ambient.clamp_min(1e-12)[None, :], gamma, support=support)
    # Bound diagnostic cost on full MLP matrices; extrema use all coordinates.
    flat = scale.flatten()
    sample = flat[::max(1, math.ceil(flat.numel() / 65536))]
    quantiles = torch.quantile(sample, sample.new_tensor([.25, .5, .75, .9, .99])).tolist()
    return scale, {
        "normalization": LOCAL_PRUNING_VERSION, "alpha": float(alpha), "gamma": gamma,
        "averaging_convention": "ambient_token_mean_geometry_zero_outside_readouts",
        "ambient_tokens": len(x), "readout_tokens": len(z),
        "mean_readout_rank": rank_sum / len(pd),
        "readout_rank_min": float(ranks.min()), "readout_rank_max": float(ranks.max()),
        "supported_readout_fraction": float((ranks > 0).float().mean()),
        "output_metric_trace_ratio": gamma * rank_sum / (pd.shape[1] * len(x)),
        "added_curvature_trace_ratio": gamma * geo_trace / ambient_trace,
        "ambient_curvature_trace_sum": ambient_trace,
        "geometric_curvature_trace_sum": geo_trace,
        "unsupported_input_coordinates": int((~support).sum()),
        "excluded_geometric_trace_sum": float(geo[:, ~support].sum()),
        "zero_geometric_trace": geo_trace == 0,
        "scale_min": float(flat.min()), "scale_max": float(flat.max()),
        "scale_quantile_probabilities": [.25, .5, .75, .9, .99],
        "scale_quantiles": quantiles, "scale_quantile_samples": sample.numel(),
    }


def row_scales(
    coupling: torch.Tensor,
    gamma: float,
    *,
    h0_diag: torch.Tensor | None = None,
    hprime_diag: torch.Tensor | None = None,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """λ_{pq} = sqrt((H_0)_{qq} / H'_{qq} * (1 + γ G_{pq})).

    With a shared Hessian ``H' = H_0``, this is ``sqrt(1 + γ G)``. Features
    with no calibration support get λ = 1.
    """
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")
    g = coupling.float()
    if g.ndim != 2:
        raise ValueError("coupling G must have shape [outputs, inputs]")
    squared = 1.0 + float(gamma) * g
    if h0_diag is not None or hprime_diag is not None:
        if h0_diag is None or hprime_diag is None:
            raise ValueError("h0_diag and hprime_diag must be supplied together")
        if h0_diag.shape != hprime_diag.shape or h0_diag.numel() != g.shape[1]:
            raise ValueError("Hessian diagonals must match the input dimension")
        ratio = h0_diag.float() / hprime_diag.float().clamp_min(1e-12)
        squared = squared * ratio.reshape(1, -1)
    scale = torch.sqrt(squared.clamp_min(0.0))
    if support is not None:
        mask = support.reshape(-1).to(dtype=torch.bool)
        if mask.numel() != g.shape[1]:
            raise ValueError("support must have one flag per input feature")
        scale = scale.clone()
        scale[:, ~mask] = 1.0
    return scale


def heuristic_score_weight(coupling: torch.Tensor, gamma: float) -> torch.Tensor:
    """Score-only ablation: ``1 + γ G`` without rescaled compensation.

    This is the labeled heuristic. Do not multiply scores by raw G.
    """
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")
    return 1.0 + float(gamma) * coupling.float()


def prune_with_rescaling(
    solver: SparseGPT,
    scale: torch.Tensor,
    *prune_args,
    **prune_kwargs,
) -> None:
    """Prune ``W ⊙ Λ`` with ambient SparseGPT, then divide the result by Λ."""
    weights = _weight_matrix(solver.layer)
    lam = scale.to(device=weights.device, dtype=weights.dtype)
    if lam.shape != weights.shape:
        raise ValueError("scale must match the SparseGPT weight layout")
    if (lam <= 0).any() or not torch.isfinite(lam).all():
        raise ValueError("rescaling factors must be finite and positive")
    scaled = weights * lam
    _write_weight_matrix(solver.layer, scaled)
    kwargs = dict(prune_kwargs)
    # Use the top-k mask path so unstructured selection matches 1+γG scores.
    if (
        int(kwargs.get("zeros_per_group", 0) or 0) == 0
        and kwargs.get("mask_weight") is None
        and kwargs.get("score_matrix") is None
    ):
        kwargs["mask_weight"] = torch.ones_like(scaled, device="cpu")
    SparseGPT.fasterprune(solver, *prune_args, **kwargs)
    hat = _weight_matrix(solver.layer)
    _write_weight_matrix(solver.layer, hat / lam)


def _write_weight_matrix(layer, weights: torch.Tensor) -> None:
    data = weights
    if isinstance(layer, transformers.Conv1D):
        data = data.t()
    layer.weight.data = data.reshape(layer.weight.shape).to(dtype=layer.weight.dtype)


def _mean_geo_rank(diagnostics: dict) -> float:
    ranks = diagnostics.get("chart_ranks") or []
    counts = diagnostics.get("chart_counts") or []
    sample_count = int(diagnostics.get("sample_count") or 0)
    if ranks and counts and sample_count:
        return float(sum(int(r) * int(c) for r, c in zip(ranks, counts)) / sample_count)
    diags = diagnostics.get("sample_projector_diagonals")
    if isinstance(diags, torch.Tensor) and diags.numel():
        return float(diags.sum(dim=1).mean())
    raise RuntimeError("tangent diagnostics do not contain a geometric rank")


class RescaleSparseGPT:
    """Variant B: SparseGPT on W ⊙ Λ with Λ = sqrt(1 + γ G).

    ``heuristic=True`` keeps the 1+γG score-only ablation (no rescaled
    compensation). ``geo_alpha`` is the unbounded α of the revised note.
    """

    def __init__(
        self,
        layer,
        *,
        alpha: float = 1.0,
        charts: int = 8,
        rank: int = 2,
        calibration_samples: int = 512,
        seed: int = 0,
        bias_fold: bool = False,
        heuristic: bool = False,
        anchor_eta: float = 0.0,
    ) -> None:
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("rescale alpha must be finite and nonnegative")
        from sparsifier.prune.tangent import TangentAwareSparseGPT

        self._solver = TangentAwareSparseGPT(
            layer,
            alpha=1.0,
            charts=charts,
            rank=rank,
            calibration_samples=calibration_samples,
            seed=seed,
            bias_fold=bias_fold,
            anchor_eta=anchor_eta,
        )
        self.geo_alpha = float(alpha)
        self.heuristic = bool(heuristic)
        self.scale: torch.Tensor | None = None
        self.gamma: float = 0.0
        self.layer = layer

    def __getattr__(self, name: str):
        return getattr(self._solver, name)

    def add_batch(self, inputs, out=None, blocksize: int = 1024) -> None:
        self._solver.add_batch(inputs, out, blocksize)

    def add_tangent_batch(self, inputs, out=None, blocksize: int = 1024) -> None:
        self._solver.add_tangent_batch(inputs, out, blocksize)

    def finalize_tangents(self) -> None:
        from sparsifier.prune.tangent import estimate_output_tangent_coupling

        solver = self._solver
        if solver.finalized:
            return
        if solver._sample_inputs is None or solver._sample_outputs is None:
            raise RuntimeError("Collect calibration inputs and outputs before pruning")
        coupling, diagnostics = estimate_output_tangent_coupling(
            solver._sample_inputs.to(solver.device),
            solver._sample_outputs.to(solver.device),
            charts=solver.charts,
            rank=solver.rank,
            alpha=1.0,
            seed=solver.seed,
            anchor_eta=solver.anchor_eta,
        )
        expected = _weight_matrix(solver.layer).shape
        if coupling.shape != expected:
            if isinstance(solver.layer, transformers.Conv1D):
                coupling = coupling.T.contiguous()
            if coupling.shape != expected:
                raise RuntimeError("Tangent score shape does not match SparseGPT weight layout")
        ambient_diag = solver.H.diag()
        support = (solver._sample_inputs.square().sum(dim=0).to(solver.device) > 1e-12)
        support = support[:coupling.shape[1]] & (ambient_diag > 1e-12)
        r_bar = _mean_geo_rank(diagnostics)
        ambient_trace = coupling.shape[0] * float(ambient_diag[support].sum())
        geometric_diag = coupling * ambient_diag[None, :]
        geo_trace = float(geometric_diag[:, support].sum())
        self.gamma = self.geo_alpha * ambient_trace / geo_trace if geo_trace > 0 else 0.0
        self.scale = row_scales(coupling, self.gamma, support=support)
        solver.mask_weight = heuristic_score_weight(coupling, self.gamma)
        diagnostics = dict(diagnostics)
        diagnostics["gamma"] = self.gamma
        diagnostics["mean_geo_rank"] = r_bar
        diagnostics["normalization"] = LOCAL_PRUNING_VERSION
        diagnostics["added_curvature_trace_ratio"] = self.gamma * geo_trace / ambient_trace if ambient_trace > 0 else 0.0
        diagnostics["unsupported_input_coordinates"] = int((~support).sum())
        diagnostics["excluded_geometric_trace"] = float(geometric_diag[:, ~support].sum())
        diagnostics["heuristic"] = self.heuristic
        solver.tangent_diagnostics = diagnostics
        solver._dense_weight = _weight_matrix(solver.layer).detach().cpu().clone()
        diags = diagnostics.get("sample_projector_diagonals")
        solver.sample_projector_diagonals = (
            None if diags is None else diags.detach().cpu().float()
        )
        solver.finalized = True
        if not solver.bias_fold:
            solver._sample_inputs = None
            solver._sample_outputs = None
            solver._sample_keys = None

    def fasterprune(self, *args, **kwargs) -> None:
        from sparsifier.prune.tangent_common import apply_bias_fold_from_samples

        self.finalize_tangents()
        solver = self._solver
        if self.heuristic:
            SparseGPT.fasterprune(solver, *args, mask_weight=solver.mask_weight, **kwargs)
        else:
            if self.scale is None:
                raise RuntimeError("rescaling factors were not finalized")
            prune_with_rescaling(solver, self.scale, *args, **kwargs)
        if solver.bias_fold:
            apply_bias_fold_from_samples(
                solver.layer,
                dense_weight=solver._dense_weight,
                sample_inputs=solver._sample_inputs,
                sample_outputs=solver._sample_outputs,
                projector_diagonals=solver.sample_projector_diagonals,
            )
            solver._sample_inputs = None
            solver._sample_outputs = None
            solver._sample_keys = None

    def free(self) -> None:
        self.scale = None
        self._solver.free()

    @property
    def tangent_diagnostics(self):
        return self._solver.tangent_diagnostics

    @property
    def H(self):
        return self._solver.H

    @H.setter
    def H(self, value) -> None:
        self._solver.H = value

