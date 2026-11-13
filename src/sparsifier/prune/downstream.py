"""Downstream tangent metric calibration (revised research note, section 4.9).

Anchor bases are fitted separately and held fixed during VJPs. Calibration
supports an activation-diagonal ablation and a shared-weight diagonal
that sums token contributions before squaring. Pruning uses Variant B's matched
rescaling and compensation; it is an approximation to the full pullback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
import transformers

from sparsifier.prune.rescale import prune_with_rescaling
from sparsifier.prune.sparsegpt import SparseGPT


Basis = torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _frames(basis: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """Return [positions, channels, rank] orthonormal frames, zeros allowed."""
    dtype = torch.float64 if activation.dtype == torch.float64 else torch.float32
    q = basis.detach().to(device=activation.device, dtype=dtype)
    if q.ndim < 2:
        raise ValueError("basis must have at least channel and rank dimensions")
    if q.ndim == 2:
        if q.shape[0] != activation.shape[-1]:
            raise ValueError("basis channel dimension does not match activation")
        q = q.expand(*activation.shape[:-1], *q.shape)
    if q.shape[:-2] != activation.shape[:-1] or q.shape[-2] != activation.shape[-1]:
        raise ValueError("basis must be [channels, rank] or [*positions, channels, rank]")
    if q.shape[-1] > q.shape[-2] or not torch.isfinite(q).all():
        raise ValueError("basis must be finite with rank <= channels")
    q = q.reshape(math.prod(activation.shape[:-1]), activation.shape[-1], q.shape[-1])
    gram = q.transpose(-1, -2) @ q
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    # Zero columns support empty charts and padded variable-rank frames.
    expected = torch.diag_embed((diagonal > 0.5).to(q.dtype))
    if not torch.allclose(gram, expected, atol=2e-4, rtol=2e-4):
        raise ValueError("basis columns must be orthonormal (or zero padding)")
    return q


@dataclass(frozen=True)
class ChartTangentBasis:
    """Frozen local PCA atlas fitted on separate activation samples.

    Use ``fit(samples, charts=..., rank=...)`` once, then pass the resulting
    callable to ``TangentAnchor`` or ``input_bases``. Assignment uses the
    nearest fitted center; zero padding preserves variable chart ranks.
    This is the existing k-means/PCA estimator, not a concept-factor model.
    """

    centers: torch.Tensor
    bases: torch.Tensor  # [charts, channels, max_rank], zero-padded

    @classmethod
    @torch.no_grad()
    def fit(cls, samples: torch.Tensor, *, charts: int = 8, rank: int = 2, seed: int = 0):
        from sparsifier.prune.tangent_common import fit_output_chart_atlas

        if samples.ndim < 2 or samples.shape[-1] == 0 or not torch.isfinite(samples).all():
            raise ValueError("chart fitting requires finite sample-major activations")
        atlas = fit_output_chart_atlas(
            samples.detach().reshape(-1, samples.shape[-1]),
            charts=charts, rank=rank, seed=seed,
        )
        bases = atlas.centers.new_zeros(len(atlas.bases), samples.shape[-1], max(atlas.ranks))
        for index, basis in enumerate(atlas.bases):
            bases[index, :, :basis.shape[1]] = basis
        return cls(atlas.centers.detach().clone(), bases.detach())

    @torch.no_grad()
    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.centers.shape[-1] or not torch.isfinite(activation).all():
            raise ValueError("finite activations must match fitted chart channels")
        values = activation.detach().reshape(-1, activation.shape[-1]).float()
        centers = self.centers.to(device=values.device, dtype=values.dtype)
        assignment = torch.cdist(values, centers).argmin(-1)
        selected = self.bases.to(device=values.device)[assignment]
        return selected.reshape(*activation.shape[:-1], *selected.shape[-2:])


def tangent_pullback_diagonal(
    output: torch.Tensor,
    basis: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    *,
    num_probes: int | None = 8,
    generator: torch.Generator | None = None,
    retain_graph: bool = False,
    weight_inputs: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate diag(J.T P J) at targets by backpropagating tangent probes.

    ``basis`` is global [channels, rank] or per-position
    [*output.shape[:-1], channels, rank]. It is detached, as are all results.
    Independent Rademacher coefficients at every output position estimate
    the block-diagonal anchor projector without assuming token independence
    in the intervening network. ``num_probes=None`` enumerates every anchor
    position and basis column exactly (for small numerical references).

    A CPU generator controls probes without touching the global RNG; absent
    a generator, a private generator seeded with zero is used.
    Disconnected targets get zero sensitivity; existing parameter .grad
    buffers are never changed. The graph is consumed unless retained.

    With ``weight_inputs``, return shared-weight diagonals instead: sum
    gradient/input outer products over tokens BEFORE squaring, separately
    for each independent sequence, then sum sequences and average probes.
    Two-dimensional targets represent one sequence; three-dimensional
    targets have an independent leading batch axis.
    """
    if num_probes is not None and (
        isinstance(num_probes, bool) or not isinstance(num_probes, int) or num_probes < 1
    ):
        raise ValueError("num_probes must be a positive integer or None for exact VJPs")
    if not output.is_floating_point() or not output.requires_grad:
        raise ValueError("anchor output must require gradients; run a grad-enabled forward")
    if not targets or any(not value.requires_grad for value in targets.values()):
        raise ValueError("nonempty target activations must require gradients")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    frames = _frames(basis, output)
    positions, _, rank = frames.shape
    if weight_inputs is not None:
        if weight_inputs.keys() != targets.keys():
            raise ValueError("weight inputs must match target names")
        for name, target in targets.items():
            x = weight_inputs[name]
            if target.ndim not in (2, 3) or x.shape[:-1] != target.shape[:-1]:
                raise ValueError("weight inputs must match target sequence axes")
            if not torch.isfinite(x).all():
                raise ValueError("weight inputs must be finite")
    result = {
        name: torch.zeros_like(value, dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
        for name, value in targets.items()
    } if weight_inputs is None else {
        name: torch.zeros(value.shape[-1], weight_inputs[name].shape[-1], dtype=torch.float64)
        for name, value in targets.items()
    }
    if rank == 0 or not frames.any():
        return result
    count = positions * rank if num_probes is None else num_probes
    for index in range(count):
        if num_probes is None:
            seed = torch.zeros_like(frames[:, :, 0])
            position, column = divmod(index, rank)
            seed[position] = frames[position, :, column]
        else:
            coefficients = torch.randint(2, (positions, rank), generator=generator, device="cpu")
            coefficients = coefficients.to(device=frames.device, dtype=frames.dtype) * 2 - 1
            seed = torch.einsum("pdr,pr->pd", frames, coefficients)
        gradients = torch.autograd.grad(
            output,
            tuple(targets.values()),
            grad_outputs=seed.reshape_as(output).to(output.dtype),
            retain_graph=retain_graph or index + 1 < count,
            create_graph=False,
            allow_unused=True,
        )
        for name, gradient in zip(targets, gradients):
            if gradient is not None:
                if weight_inputs is None:
                    result[name].add_(gradient.detach().to(result[name].dtype).square())
                else:
                    g = gradient.detach().to(device="cpu", dtype=torch.float64)
                    x = weight_inputs[name].detach().to(device="cpu", dtype=torch.float64)
                    if g.ndim == 2:
                        g, x = g.unsqueeze(0), x.unsqueeze(0)
                    # Shared weights aggregate tokens, never independent examples.
                    for sequence_g, sequence_x in zip(g, x):
                        result[name].add_((sequence_g.T @ sequence_x).square())
    if num_probes is not None:
        for diagonal in result.values():
            diagonal.div_(count)
    if any(not torch.isfinite(diagonal).all() for diagonal in result.values()):
        raise ValueError("nonfinite tangent pullback; check activations and backward precision")
    return result


def _tensor_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("anchor must return a tensor, or supply an output selector")


@dataclass(frozen=True)
class TangentAnchor:
    """A downstream activation site and its separately fitted tangent basis.

    A basis callback receives detached activations in their original shape
    and returns orthonormal frames in the formats accepted by
    ``tangent_pullback_diagonal``. Tangent-plus-presence unions work too.
    For blocks returning tuples, pass ``select=lambda out: out[0]``.
    """

    module: nn.Module
    basis: Basis
    weight: float = 1.0
    select: Callable = _tensor_output


class DownstreamTangentSparseGPT(SparseGPT):
    """SparseGPT supplied with sample-paired downstream sensitivity moments.

    ``alpha`` sets total geometric diagonal curvature relative to ambient
    curvature, normalized by their traces over all weight coordinates.
    ``input_alpha`` similarly weights an optional local input-tangent term.
    Both keep ambient reconstruction and disable zero-energy terms.
    """

    def __init__(self, layer: nn.Module, *, alpha: float = 1.0, input_alpha: float = 0.0):
        if not isinstance(layer, (nn.Linear, transformers.Conv1D)):
            raise TypeError("downstream tangent pruning supports Linear and Conv1D")
        _finite_nonnegative(alpha, "alpha")
        _finite_nonnegative(input_alpha, "input_alpha")
        super().__init__(layer)
        self.alpha, self.input_alpha = float(alpha), float(input_alpha)
        rows = layer.out_features if isinstance(layer, nn.Linear) else layer.nf
        self.input_energy = torch.zeros(self.columns, dtype=torch.float64)
        self.geometry_energy = torch.zeros(rows, self.columns, dtype=torch.float64)
        self.input_geometry_energy = torch.zeros_like(self.geometry_energy)
        self.vector_count = 0
        self.geometry_vector_count = 0
        self.metric = "activation_diagonal"
        self.scale: torch.Tensor | None = None
        self.tangent_diagnostics: dict = {}

    def add_batch(self, inputs, out=None, blocksize=1024) -> None:
        raise RuntimeError("use DownstreamTangentCalibrator.run or add_pullback_batch")

    @torch.no_grad()
    def add_pullback_batch(self, inputs, sensitivity, *, input_basis=None, collect_ambient=True) -> None:
        if self.scale is not None or self.H is None:
            raise RuntimeError("cannot add calibration after finalization or pruning")
        if inputs.ndim not in (2, 3) or inputs.shape[-1] != self.columns or inputs.numel() == 0:
            raise ValueError("inputs must be nonempty [positions, inputs] or [batch, sequence, inputs]")
        if sensitivity.shape != (*inputs.shape[:-1], self.geometry_energy.shape[0]):
            raise ValueError("sensitivity positions/output channels must match local inputs/layer")
        x = inputs.detach().reshape(-1, self.columns).to(device="cpu", dtype=torch.float64)
        b = sensitivity.detach().reshape(len(x), -1).to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(x).all() or not torch.isfinite(b).all() or (b < 0).any():
            raise ValueError("inputs must be finite and sensitivities finite/nonnegative")
        local_diagonal = None
        if input_basis is not None:
            frames = _frames(input_basis, inputs)
            local_diagonal = frames.square().sum(-1).to(device="cpu", dtype=torch.float64)
        elif self.input_alpha:
            raise ValueError("input_alpha > 0 requires a local input tangent basis on every batch")
        if collect_ambient:
            self.add_ambient_batch(inputs)
        self.geometry_energy.add_(b.T @ x.square())
        if local_diagonal is not None:
            self.input_geometry_energy.add_(b.T @ local_diagonal)
        self.geometry_vector_count += len(x)

    @torch.no_grad()
    def add_ambient_batch(self, inputs: torch.Tensor) -> None:
        """Add matched ambient calibration independently of backward probes.

        Use with ``collect_ambient=False`` when task-gradient sequences have
        a different token budget from the ambient SparseGPT reference.
        """
        if self.scale is not None or self.H is None:
            raise RuntimeError("cannot add calibration after finalization or pruning")
        if inputs.ndim not in (2, 3) or inputs.shape[-1] != self.columns or not inputs.numel():
            raise ValueError("ambient inputs must have nonempty matching input channels")
        x = inputs.detach().reshape(-1, self.columns).to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError("ambient inputs must be finite")
        SparseGPT.add_batch(self, inputs.detach().to(self.device))
        self.input_energy.add_(x.square().sum(0))
        self.vector_count += len(x)

    @torch.no_grad()
    def add_weight_diagonal_batch(self, inputs, diagonal, *, collect_ambient=True):
        """Accumulate an already probe-averaged, sequence-summed weight metric."""
        if self.scale is not None or self.H is None:
            raise RuntimeError("cannot add calibration after finalization or pruning")
        if self.input_alpha:
            raise ValueError("weight-diagonal calibration does not support input_alpha")
        value = diagonal.detach().to(device="cpu", dtype=torch.float64)
        if value.shape != self.geometry_energy.shape or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("weight diagonal must be finite, nonnegative and match layer weights")
        if collect_ambient:
            self.add_ambient_batch(inputs)
        self.geometry_energy.add_(value)
        self.metric = "weight_diagonal"
        self.geometry_vector_count += inputs.numel() // self.columns

    def finalize_tangents(self) -> None:
        if self.scale is not None:
            return
        if self.H is None:
            raise RuntimeError("solver has already been pruned or freed")
        if not self.vector_count:
            raise RuntimeError("collect downstream tangent calibration before pruning")
        support = self.input_energy > 0
        ambient_trace = float(self.input_energy.sum()) * self.geometry_energy.shape[0]
        # Exclude unsupported entries; they cannot be represented by rescaling.
        geo = self.geometry_energy.clone()
        inp = self.input_geometry_energy.clone()
        geo[:, ~support] = 0
        inp[:, ~support] = 0
        geo_trace, input_trace = float(geo.sum()), float(inp.sum())
        raw_trace = float(self.geometry_energy.sum())
        gamma = self.alpha * ambient_trace / geo_trace if geo_trace > 0 else 0.0
        eta = self.input_alpha * ambient_trace / input_trace if input_trace > 0 else 0.0
        extra = gamma * geo + eta * inp
        denominator = torch.where(support, self.input_energy, 1.0)
        self.scale = torch.sqrt(1.0 + extra / denominator).float()
        if not torch.isfinite(self.scale).all():
            self.scale = None
            raise ValueError("nonfinite geometry scales")
        self.tangent_diagnostics = {
            "method": "downstream_tangent_pullback",
            "sample_count": self.vector_count,
            "geometry_sample_count": self.geometry_vector_count,
            "alpha": self.alpha,
            "input_alpha": self.input_alpha,
            "gamma": gamma,
            "eta": eta,
            "ambient_trace": ambient_trace / self.vector_count,
            "geometry_trace": geo_trace / self.vector_count,
            "raw_geometry_trace": raw_trace / self.vector_count,
            "excluded_geometry_trace": (raw_trace - geo_trace) / self.vector_count,
            "achieved_geometry_to_ambient_ratio": gamma * geo_trace / ambient_trace if ambient_trace else 0.,
            "input_geometry_trace": input_trace / self.vector_count,
            "unsupported_inputs": int((~support).sum()),
            "metric": self.metric,
            "diagonal_output_and_token_approximation": self.metric == "activation_diagonal",
            "shared_weight_diagonal_approximation": self.metric == "weight_diagonal",
            "congruence_approximation": True,
        }

    @torch.no_grad()
    def fasterprune(self, *args, **kwargs) -> None:
        if self.H is None:
            raise RuntimeError("solver has already been pruned or freed")
        self.finalize_tangents()
        prune_with_rescaling(self, self.scale, *args, **kwargs)

    def free(self) -> None:
        super().free()
        self.input_energy = torch.empty(0, dtype=torch.float64)
        self.geometry_energy = torch.empty(0, dtype=torch.float64)
        self.input_geometry_energy = torch.empty(0, dtype=torch.float64)
        self.scale = None


class DownstreamTangentCalibrator:
    """Run grad-enabled network forwards and prepare target SparseGPT solvers.

    Hooks exist only inside ``run(forward)`` and are removed on failure too.
    Frozen target outputs are made activation-gradient leaves, so a fully
    frozen model works without changing parameter requires_grad flags or
    .grad buffers. Target modules and anchor modules must each execute once
    per forward (shared/recurrent modules are rejected).

    The caller controls model eval mode, checkpointing, masks and use_cache.
    Target input bases are optional detached callbacks with the same frame
    convention as anchors. Anchor bases/assignments must be fitted separately.
    """

    def __init__(
        self,
        targets: Mapping[str, nn.Module],
        anchors: Mapping[str, TangentAnchor],
        *,
        alpha: float = 1.0,
        input_alpha: float = 0.0,
        input_bases: Mapping[str, Basis] | None = None,
        num_probes: int | None = 8,
        seed: int = 0,
        metric: str = "activation_diagonal",
    ):
        if not targets or not anchors:
            raise ValueError("at least one target and one anchor are required")
        if metric not in ("activation_diagonal", "weight_diagonal"):
            raise ValueError("unknown downstream metric")
        if metric == "weight_diagonal" and (input_alpha or input_bases):
            raise ValueError("weight_diagonal does not support local input tangent coupling")
        self.metric = metric
        if len({id(module) for module in targets.values()}) != len(targets):
            raise ValueError("target modules must be unique")
        if num_probes is not None and (
            isinstance(num_probes, bool) or not isinstance(num_probes, int) or num_probes < 1
        ):
            raise ValueError("num_probes must be a positive integer or None")
        for anchor in anchors.values():
            _finite_nonnegative(anchor.weight, "anchor weight")
        self.targets, self.anchors = dict(targets), dict(anchors)
        self.input_bases = dict(input_bases or {})
        if self.input_bases.keys() - self.targets.keys():
            raise ValueError("input bases must refer to target names")
        if input_alpha and self.input_bases.keys() != self.targets.keys():
            raise ValueError("input_alpha > 0 requires an input basis for every target")
        self.solvers = {
            name: DownstreamTangentSparseGPT(module, alpha=alpha, input_alpha=input_alpha)
            for name, module in targets.items()
        }
        self.num_probes = num_probes
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.batch_count = 0

    def run(self, forward: Callable[[], object], *, collect_ambient: bool = True) -> None:
        """Collect one batch; e.g. ``run(lambda: model(**batch, use_cache=False))``."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("calibration cannot run in inference_mode; use a normal forward")
        if any(solver.scale is not None or solver.H is None for solver in self.solvers.values()):
            raise RuntimeError("cannot calibrate after a target is finalized, pruned or freed")
        target_inputs, target_outputs, anchor_outputs = {}, {}, {}
        handles = []

        def target_hook(name):
            def capture(module, args, kwargs, output):
                if name in target_outputs:
                    raise RuntimeError(f"target {name!r} executed more than once")
                if not isinstance(output, torch.Tensor):
                    raise TypeError("target linear must return a tensor")
                inputs = args[0] if args else kwargs.get("input", kwargs.get("x"))
                if not isinstance(inputs, torch.Tensor):
                    raise TypeError("target linear must receive a tensor input")
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                target_inputs[name] = inputs.detach().clone()
                target_outputs[name] = output
                return output
            return capture

        def anchor_hook(name, anchor):
            def capture(module, args, output):
                if name in anchor_outputs:
                    raise RuntimeError(f"anchor {name!r} executed more than once")
                anchor_outputs[name] = anchor.select(output)
            return capture

        try:
            for name, module in self.targets.items():
                handles.append(module.register_forward_hook(target_hook(name), with_kwargs=True))
            for name, anchor in self.anchors.items():
                handles.append(anchor.module.register_forward_hook(anchor_hook(name, anchor)))
            with torch.enable_grad():
                forward()
                if target_outputs.keys() != self.targets.keys() or anchor_outputs.keys() != self.anchors.keys():
                    raise RuntimeError("forward did not execute every target and anchor")
                sensitivities = {
                    name: torch.zeros_like(
                        out, dtype=torch.float64 if out.dtype == torch.float64 else torch.float32,
                    )
                    for name, out in target_outputs.items()
                } if self.metric == "activation_diagonal" else {
                    name: torch.zeros_like(solver.geometry_energy) for name, solver in self.solvers.items()
                }
                for index, (name, anchor) in enumerate(self.anchors.items()):
                    if anchor.weight == 0:
                        continue
                    output = anchor_outputs[name]
                    basis = anchor.basis(output.detach()) if callable(anchor.basis) else anchor.basis
                    diagonal = tangent_pullback_diagonal(
                        output, basis, target_outputs,
                        num_probes=self.num_probes, generator=self.generator,
                        retain_graph=index + 1 < len(self.anchors),
                        weight_inputs=target_inputs if self.metric == "weight_diagonal" else None,
                    )
                    for target in sensitivities:
                        sensitivities[target].add_(diagonal[target], alpha=anchor.weight)
                input_frames = {}
                for name, basis in self.input_bases.items():
                    inputs = target_inputs[name]
                    input_frames[name] = basis(inputs) if callable(basis) else basis
                for name, solver in self.solvers.items():
                    if self.metric == "weight_diagonal":
                        solver.add_weight_diagonal_batch(target_inputs[name], sensitivities[name],
                                                        collect_ambient=collect_ambient)
                    else:
                        solver.add_pullback_batch(
                            target_inputs[name], sensitivities[name], input_basis=input_frames.get(name),
                            collect_ambient=collect_ambient,
                        )
                self.batch_count += 1
        finally:
            for handle in handles:
                handle.remove()
