"""Frozen full-direction neuron metrics and bounded coupled references.

Probe factors Z represent rank-one PSD weight-curvature terms, not squared
weight diagonals. All geometric sums share the ambient token denominator.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import torch
from sparsifier.prune.sample_metric import refit as explicit_refit


@dataclass
class FrozenNeuronMetric:
    ambient_count: int
    gamma: float = 1.
    local_inputs: torch.Tensor | None = None
    frames: list[torch.Tensor] | None = None
    probes: list[tuple[torch.Tensor, float]] | None = None
    cache_budget_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if self.ambient_count < 1 or not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("positive ambient token count and nonnegative gamma required")
        retained = [z for z, _ in (self.probes or [])] + (self.frames or [])
        if self.local_inputs is not None:
            retained.append(self.local_inputs)
        if self.cache_budget_bytes < 0 or sum(t.numel()*t.element_size() for t in retained) > self.cache_budget_bytes:
            raise ValueError("full-metric cache budget exceeded; replay or reduce the reference")
        if (self.local_inputs is None) != (self.frames is None):
            raise ValueError("local frames require paired geometric inputs")
        if self.local_inputs is not None:
            if self.local_inputs.ndim != 2 or len(self.local_inputs) != len(self.frames):
                raise ValueError("local geometry pairs do not align")
            self.local_inputs = self.local_inputs.detach().clone()
            self.frames = [q.detach().clone() for q in self.frames]
            if not torch.isfinite(self.local_inputs).all():
                raise ValueError("nonfinite local inputs")
            for q in self.frames:
                if q.ndim != 2 or not torch.isfinite(q).all() or not torch.allclose(
                    q.T @ q, torch.eye(q.shape[1], device=q.device, dtype=q.dtype), atol=2e-5, rtol=2e-5):
                    raise ValueError("finite orthonormal frames required")
        self.probes = [(z.detach().clone(), float(w)) for z, w in (self.probes or [])]
        if any(z.ndim != 2 or not torch.isfinite(z).all() or not math.isfinite(w) or w < 0
               for z, w in self.probes):
            raise ValueError("invalid frozen curvature probe")
        tensors = [z for z, _ in self.probes] + (self.frames or [])
        if self.local_inputs is not None:
            tensors.append(self.local_inputs)
        if sum(t.numel()*t.element_size() for t in tensors) > self.cache_budget_bytes:
            raise ValueError("full-metric cache budget exceeded; use deterministic replay or a smaller reference")

    def with_gamma(self, gamma):
        # Parameters are frozen by construction; no need to copy the cache.
        import copy
        result = copy.copy(self)
        result.gamma = gamma
        return result

    def action(self, a, h0, ridge=0.):
        result = a @ h0 + torch.as_tensor(ridge, device=a.device, dtype=a.dtype) * a
        if self.local_inputs is not None:
            for z, q in zip(self.local_inputs, self.frames):
                z, q = z.to(a), q.to(a)
                result = result + self.gamma/self.ambient_count * torch.outer(q @ (q.T @ (a @ z)), z)
        for z, weight in self.probes:
            z = z.to(a)
            if z.shape != a.shape:
                raise ValueError("probe/weight shape mismatch")
            result = result + self.gamma*weight/self.ambient_count * z * (z*a).sum()
        return result

    def diagonal(self, shape, *, device=None, dtype=torch.float64):
        k = torch.zeros(shape, device=device, dtype=dtype)
        if self.local_inputs is not None:
            for z, q in zip(self.local_inputs, self.frames):
                k += torch.outer(q.to(k).square().sum(1), z.to(k).square()) / self.ambient_count
        for z, weight in self.probes:
            if z.shape != shape:
                raise ValueError("probe shape mismatch")
            k += weight * z.to(k).square() / self.ambient_count
        return k

    def direct_scores(self, weight, h0):
        scores = weight.square().sum(0) * h0.diagonal()
        if self.local_inputs is not None:
            for z, q in zip(self.local_inputs, self.frames):
                scores += self.gamma/self.ambient_count * z.to(weight).square() * (q.to(weight).T @ weight).square().sum(0)
        for z, probability in self.probes:
            scores += self.gamma*probability/self.ambient_count * (weight*z.to(weight)).sum(0).square()
        return scores

    def matrix(self, shape, h0, ridge=0.):
        size = math.prod(shape)
        if size > 2048:
            raise ValueError("explicit full reference limited to 2048 weight coordinates")
        eye = torch.eye(size, device=h0.device, dtype=torch.float64)
        return torch.stack([self.action(row.reshape(shape), h0.double(), ridge).flatten() for row in eye], 1)

    def refit(self, weight, keep, h0, ridge, *, method="direct", rtol=1e-5, max_iterations=200):
        mask = keep[None, :].expand_as(weight)
        if method == "direct":
            h = self.matrix(weight.shape, h0, ridge)
            candidate = explicit_refit(weight, mask, h).to(weight)
            iterations, converged = 1, True
        elif method == "cg":
            residual = torch.where(mask, torch.zeros_like(weight), weight)
            rhs = -self.action(residual, h0, ridge)[mask]
            def apply(v):
                a = torch.zeros_like(weight)
                a[mask] = v
                return self.action(a, h0, ridge)[mask]
            diagonal = h0.diagonal()[None, :].expand_as(weight) + self.gamma*self.diagonal(
                weight.shape, device=weight.device, dtype=weight.dtype) + torch.as_tensor(ridge, device=weight.device)
            preconditioner = diagonal[mask]
            if (preconditioner <= 0).any():
                raise ValueError("positive full-metric preconditioner required")
            solution = torch.zeros_like(rhs)
            r = rhs.clone()
            z = r/preconditioner
            p = z.clone()
            rz = torch.dot(r, z)
            target = rtol * rhs.norm().clamp_min(1e-30)
            converged = bool(r.norm() <= target)
            iterations = 0
            for iterations in range(1, max_iterations+1):
                if converged:
                    break
                ap = apply(p)
                denom = torch.dot(p, ap)
                if denom <= 0:
                    raise RuntimeError("non-positive CG curvature")
                step = rz/denom
                solution += step*p
                r -= step*ap
                converged = bool(r.norm() <= target)
                z = r/preconditioner
                next_rz = torch.dot(r, z)
                p = z + (next_rz/rz)*p
                rz = next_rz
            residual[mask] = solution
            candidate = weight-residual
        else:
            raise ValueError("unknown full refit method")
        error = weight-candidate
        gradient = self.action(error, h0, ridge)
        stationarity = float(gradient[mask].norm()/self.action(weight, h0, ridge).norm().clamp_min(1e-30))
        return candidate, {"full_metric_converged": converged, "full_metric_iterations": iterations,
                           "full_metric_relative_stationarity": stationarity,
                           "full_metric_objective": float((error*gradient).sum()),
                           "full_metric_ridge": "isotropic" if not isinstance(ridge, torch.Tensor) or ridge.ndim == 0 else "anisotropic"}


class NeuronProbeConsumer:
    """Stream unsquared contractions; optionally cache bounded full factors."""
    def __init__(self, reference_down, *, cache_budget_bytes=0):
        self.reference = reference_down.detach().to(device="cpu", dtype=torch.float64)
        self.diagonal = torch.zeros_like(self.reference, dtype=torch.float64, device="cpu")
        self.group_scores = torch.zeros(self.reference.shape[1], dtype=torch.float64)
        self.cache_budget_bytes = cache_budget_bytes
        self.cache_bytes = 0
        self.probes = []
        self.count = 0

    def __call__(self, probe):
        # VJPs retain sequence/token axes. Independent sequences stay separate.
        z = probe["contraction"].detach().to(device="cpu", dtype=torch.float64)
        coefficient = probe["coefficient"]
        self.diagonal += coefficient*z.square()
        self.group_scores += coefficient*(z*self.reference).sum(0).square()
        self.count += 1
        if self.cache_budget_bytes:
            size = z.numel()*z.element_size()
            if self.cache_bytes+size > self.cache_budget_bytes:
                raise ValueError("probe cache budget exceeded; replay or reduce the reference")
            self.probes.append((z.clone(), coefficient))
            self.cache_bytes += size


@dataclass
class StreamedGroupMetric:
    """Full-direction direct scores without a retained full refit operator."""
    group_energy: torch.Tensor
    ambient_count: int
    gamma: float = 1.

    def with_gamma(self, gamma):
        return replace(self, gamma=gamma)

    def direct_scores(self, weight, h0):
        return weight.square().sum(0)*h0.diagonal()+self.gamma*self.group_energy.to(weight)/self.ambient_count

    def refit(self, *args, **kwargs):
        raise ValueError("full-metric refit needs cached factors or deterministic replay, not only group scores")


def down_only_bridge(weight, ambient, entrywise_keep, column_keep, *, ridge=.01, full_metric=None):
    """Bounded structure-control experiment with identical reconstruction/ridge.

    Equal deleted down coordinates, independent converged fixed-mask solves.
    This deliberately does not match architectural triplet storage savings.
    """
    if weight.numel() > 2048:
        raise ValueError("bridge reference limited to 2048 down coordinates")
    if entrywise_keep.dtype != torch.bool or entrywise_keep.shape != weight.shape:
        raise ValueError("invalid entrywise bridge mask")
    column_mask = column_keep[None, :].expand_as(weight)
    if column_keep.dtype != torch.bool or int((~column_mask).sum()) != int((~entrywise_keep).sum()):
        raise ValueError("bridge masks must delete equal numbers of down coordinates")
    x = ambient.double()
    h0 = x.T@x/len(x)
    h = (full_metric or FrozenNeuronMetric(len(x), gamma=0)).matrix(weight.shape, h0, ridge)
    outcomes = {}
    for name, keep in (("entrywise", entrywise_keep), ("column", column_mask)):
        candidate = explicit_refit(weight, keep, h)
        error = (weight.double()-candidate).flatten()
        outcomes[name] = {"candidate": candidate, "objective": float(error@h@error),
                          "free_gradient_norm": float((h@error)[keep.flatten()].norm())}
    return outcomes
