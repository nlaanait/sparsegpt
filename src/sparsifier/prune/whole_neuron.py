"""Static SwiGLU triplet deletion with down-only compensation.

One authoritative column mask; survivors of gate/up are never optimized.
Curvatures use ambient token means. This is independent of entrywise masks.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import math
import operator
import time
import torch
from torch import nn
from torch.nn import functional as F

VERSION = "whole-neuron-v2"
METRICS = ("ambient", "local", "downstream_activation_diagonal", "downstream_weight_diagonal")
SELECTORS = ("block_obs", "direct_write", "direct_full_metric", "random", "greedy_reference")


@dataclass(frozen=True)
class WholeNeuronConfig:
    sparsity: float
    metric: str = "ambient"
    alpha: float = 0.
    selector: str = "block_obs"
    compensation: str = "surrogate_refit"
    blocksize: int = 128
    percdamp: float = .01
    unsupported_policy: str = "protect"
    support_threshold: float = 1e-12
    seed: int = 0
    input_alpha: float = 0.
    input_structure: str = "independent_tokens"
    full_metric_solver: str = "auto"
    full_metric_rtol: float = 1e-5
    full_metric_max_iterations: int = 200

    def __post_init__(self):
        if not math.isfinite(self.sparsity) or not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must be in [0,1)")
        if self.metric not in METRICS or self.selector not in SELECTORS:
            raise ValueError("unknown metric or selector")
        if self.compensation not in ("surrogate_refit", "none", "full_metric_refit"):
            raise ValueError("unknown compensation")
        if self.unsupported_policy not in ("protect", "allow"):
            raise ValueError("unsupported_policy must be protect or allow")
        if isinstance(self.blocksize, bool) or not isinstance(self.blocksize, int) or self.blocksize < 1:
            raise ValueError("blocksize must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (self.alpha, self.input_alpha, self.percdamp, self.support_threshold)):
            raise ValueError("alpha, percdamp and support threshold must be finite/nonnegative")
        if self.metric == "ambient" and self.alpha != 0:
            raise ValueError("ambient requires alpha=0")
        from sparsifier.prune.two_sided import STRUCTURES
        if self.input_structure not in STRUCTURES or (self.input_alpha and not self.metric.startswith("downstream_")):
            raise ValueError("two-sided input coupling requires a downstream metric and declared structure")
        if self.metric == "downstream_activation_diagonal" and self.input_structure != "independent_tokens":
            raise ValueError("activation-diagonal coupling requires independent tokens")
        if self.full_metric_solver not in ("auto", "direct", "cg"):
            raise ValueError("unknown full-metric solver")
        if not math.isfinite(self.full_metric_rtol) or self.full_metric_rtol <= 0 or self.full_metric_max_iterations < 1:
            raise ValueError("positive full-metric tolerance and iteration limit required")


class SwiGLUTriplet:
    """Validate the actual FX computation, not just projection attribute names."""
    def __init__(self, mlp: nn.Module):
        self.mlp = mlp
        if any(m._forward_hooks or m._forward_pre_hooks for m in mlp.modules()):
            raise ValueError("forward/pre-forward hooks are unsupported on a SwiGLU triplet")
        from torch.nn.modules import module as module_hooks
        if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
            raise ValueError("global forward/pre-forward hooks prevent architecture validation")
        modules = [getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")]
        if any(type(m) is not nn.Linear or m.bias is not None for m in modules):
            raise ValueError("requires distinct bias-free dense Linear gate/up/down projections")
        self.gate, self.up, self.down = modules
        h, d = self.gate.weight.shape
        if min(h, d) < 1 or self.up.weight.shape != (h, d) or self.down.weight.shape != (d, h):
            raise ValueError("incompatible SwiGLU triplet shapes")
        if len({m.weight.untyped_storage().data_ptr() for m in modules}) != 3:
            raise ValueError("tied or aliased triplet parameters unsupported")
        if len({(m.weight.device, m.weight.dtype) for m in modules}) != 1:
            raise ValueError("triplet devices and dtypes must agree")
        if any(not m.weight.is_floating_point() or not torch.isfinite(m.weight).all() for m in modules):
            raise ValueError("finite floating-point weights required")
        try:
            graph = torch.fx.symbolic_trace(mlp)
            nodes = list(graph.graph.nodes)
            x, = [n for n in nodes if n.op == "placeholder"]
            gate, = [n for n in nodes if n.op == "call_module" and n.target == "gate_proj"]
            up, = [n for n in nodes if n.op == "call_module" and n.target == "up_proj"]
            down, = [n for n in nodes if n.op == "call_module" and n.target == "down_proj"]
            act, = [n for n in nodes if (n.op == "call_function" and n.target is F.silu)
                     or (n.op == "call_module" and type(graph.get_submodule(n.target)) is nn.SiLU)]
            mul, = [n for n in nodes if n.op == "call_function" and n.target is operator.mul]
            out, = [n for n in nodes if n.op == "output"]
            if (len(nodes) != 7 or gate.args != (x,) or up.args != (x,) or act.args != (gate,)
                    or set(mul.args) != {act, up} or down.args != (mul,) or out.args != (down,)):
                raise ValueError("extra intermediate transformations")
            if act.kwargs.get("inplace", False):
                raise ValueError("inplace activation unsupported")
        except Exception as exc:
            raise ValueError("unsupported MLP computation; expected down(silu(gate(x))*up(x))") from exc

    @property
    def weights(self):
        return tuple(m.weight.detach() for m in (self.gate, self.up, self.down))


@dataclass
class WholeNeuronCandidate:
    keep: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    diagnostics: dict


def _curvature(h0):
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1] or not torch.isfinite(h0).all():
        raise ValueError("finite square curvature required")
    if not torch.allclose(h0, h0.T, atol=1e-6, rtol=1e-5):
        raise ValueError("curvature must be symmetric")


def damped_curvature(h0, percdamp):
    """One damping owner: return H, inverse Cholesky and effective ridge."""
    start = time.monotonic()
    _curvature(h0)
    mean = h0.diagonal().mean()
    if mean <= 0:
        raise ValueError("zero ambient energy")
    rho = percdamp
    for attempt in range(20):
        delta = rho * mean
        h = (h0 + h0.T) * .5 + delta * torch.eye(len(h0), device=h0.device, dtype=h0.dtype)
        try:
            chol = torch.linalg.cholesky(h)
            c = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
            return h, c, {"requested_percdamp": percdamp, "effective_percdamp": rho,
                           "damping": float(delta), "damping_retries": attempt,
                           "factorization_seconds": time.monotonic()-start,
                           "minimum_diagonal": float(h.diagonal().min())}
        except torch.linalg.LinAlgError:
            rho = rho * 2 if rho else 1e-8
    raise RuntimeError("whole-neuron curvature factorization failed after 20 attempts")


def refit_columns(reference_weight, h_damped, keep, *, row_chunk=256):
    """Exact fixed-mask solve; original reference weights and same ridge."""
    d, width = reference_weight.shape
    if keep.dtype != torch.bool or keep.shape != (width,) or not keep.any():
        raise ValueError("keep must be a boolean column mask retaining at least one neuron")
    keep = keep.to(reference_weight.device)
    _curvature(h_damped)
    if h_damped.shape != (width, width):
        raise ValueError("curvature width mismatch")
    out = reference_weight.clone()
    if keep.all():
        return out
    out[:, ~keep] = 0
    factor = torch.linalg.cholesky(h_damped[keep][:, keep])
    cross = h_damped[keep][:, ~keep]
    for begin in range(0, d, row_chunk):
        stop = min(begin + row_chunk, d)
        rhs = cross @ reference_weight[begin:stop, ~keep].T
        out[begin:stop, keep] += torch.cholesky_solve(rhs, factor).T
    return out


def select_columns(weight, h0, *, count, blocksize=128, damping=.01, eligible=None, scale=None, progress=None):
    """Shared-mask block OBS; deterministic ties in original neuron order."""
    start = time.monotonic()
    h, c, diagnostics = damped_curvature(h0, damping)
    if progress is not None:
        progress({"event": "curvature_factorized", **diagnostics})
    v = weight.clone() if scale is None else weight * scale
    width = v.shape[1]
    eligible = torch.ones(width, dtype=torch.bool, device=v.device) if eligible is None else eligible.to(v.device)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < width or count > int(eligible.sum()):
        raise ValueError("deletion count exceeds eligible neurons or removes every neuron")
    if not isinstance(blocksize, int) or isinstance(blocksize, bool) or blocksize < 1:
        raise ValueError("positive integer blocksize required")
    keep = torch.ones(width, dtype=torch.bool, device=v.device)
    total, previous = int(eligible.sum()), 0
    quotas = []
    for b in range(0, width, blocksize):
        e = min(b + blocksize, width)
        cumulative = previous + int(eligible[b:e].sum())
        quota = count * cumulative // total - count * previous // total if total else 0
        indices = torch.where(eligible[b:e])[0] + b
        scores = v[:, indices].square().sum(0) / c.diagonal()[indices].square()
        chosen = indices[torch.argsort(scores, stable=True)[:quota]]
        keep[chosen] = False
        errors = v.new_zeros(v.shape[0], e - b)
        for q in range(b, e):
            old = v[:, q].clone()
            new = old.clone() if keep[q] else torch.zeros_like(old)
            err = (old - new) / c[q, q]
            v[:, q:e] -= err[:, None] * c[q, q:e]
            v[:, q] = new
            errors[:, q - b] = err
        v[:, e:] -= errors @ c[b:e, e:]
        quotas.append(quota)
        previous = cumulative
        if progress is not None:
            progress({"event": "neuron_selection_block", "columns_done": e, "columns_total": width,
                "blocks_done": b//blocksize+1, "blocks_total": math.ceil(width/blocksize),
                "removed_neurons": int((~keep).sum()), "target_deletions": count,
                "block_obs_surrogate_objective": float(errors.square().sum()/2),
                "objective_coordinates": "scaled_weights_half_damped_quadratic"})
    return keep, v, h, dict(diagnostics, block_quotas=quotas,
                           column_selection_seconds=time.monotonic()-start-diagnostics["factorization_seconds"])


def greedy_columns(weight, h, count, eligible):
    """Bounded unrestricted greedy reference, not the production selector."""
    if weight.shape[1] > 256:
        raise ValueError("greedy reference limited to 256 neurons")
    active = torch.arange(weight.shape[1], device=weight.device)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    v = weight.clone()
    for _ in range(count):
        scores = v.square().sum(0) / inverse.diagonal()
        scores[~eligible[active]] = float("inf")
        j = int(torch.argmin(scores))
        free = torch.arange(len(active), device=active.device) != j
        v = v[:, free] - v[:, j:j+1] * inverse[j, free] / inverse[j, j]
        inverse = inverse[free][:, free] - inverse[free, j:j+1] * inverse[j, free] / inverse[j, j]
        active = active[free]
    keep = torch.zeros(weight.shape[1], dtype=torch.bool, device=weight.device)
    keep[active] = True
    return keep


def geometric_scales(h0, k, alpha, output_dim, *, threshold=1e-12):
    from sparsifier.prune.rescale import row_scales
    diag = h0.diagonal()
    support = diag > threshold * diag.max()
    base = output_dim * float(diag[support].sum())
    if base <= 0:
        raise ValueError("zero supported ambient energy")
    if k.shape != (output_dim, len(diag)) or not torch.isfinite(k).all() or (k < 0).any():
        raise ValueError("finite nonnegative geometric weight diagonal required")
    raw, supported = float(k.sum()), float(k[:, support].sum())
    excluded = raw - supported
    if alpha > 0 and (supported <= 0 or excluded > max(1e-12, 1e-8 * raw)):
        raise ValueError("zero supported geometry or material geometry outside ambient support; align calibration")
    gamma = alpha * base / supported if supported else 0.
    denominator = torch.where(support, diag, 1.)
    # Preserve FP64 reference precision; shared helper is FP32 production.
    scale = row_scales(k / denominator, gamma, support=support).to(k.dtype)
    if k.dtype == torch.float64:
        scale = (1 + gamma * k / denominator).sqrt()
        scale[:, ~support] = 1
    return scale, support, {"gamma": gamma, "raw_geometry_trace": raw,
                            "supported_geometry_trace": supported, "excluded_geometry_trace": excluded,
                            "added_to_ambient_ratio": gamma * supported / base,
                            "averaging": "ambient_token_mean", "support_threshold": threshold}


def build_candidate(triplet, keep, compensated_down, diagnostics=None):
    g, u, d = triplet.weights
    keep = keep.to(device=g.device)
    if keep.dtype != torch.bool or keep.shape != (len(g),) or not keep.any():
        raise ValueError("invalid neuron mask")
    if compensated_down.shape != d.shape or not torch.isfinite(compensated_down).all():
        raise ValueError("invalid compensated down matrix")
    gh, uh = g.clone(), u.clone()
    gh[~keep] = 0
    uh[~keep] = 0
    dh = compensated_down.to(device=d.device, dtype=d.dtype).clone()
    dh[:, ~keep] = 0
    candidate = WholeNeuronCandidate(keep.clone(), gh, uh, dh, dict(diagnostics or {}))
    return candidate


@torch.no_grad()
def prune_triplet(triplet, calibration, config, *, geometric_diagonal=None, full_metric=None,
                  _requested_percdamp=None, _refit_retries=0, progress=None):
    """Pure out-of-place solve. calibration is [tokens, intermediate_width]."""
    start = time.monotonic()
    if config.alpha == 0 and config.input_alpha == 0:
        # Reused fitted metrics may retain a positive gamma.
        full_metric = None
    g, u, d = triplet.weights
    width = len(g)
    count = math.floor(config.sparsity * width)
    if count == 0:
        return build_candidate(triplet, torch.ones(width, dtype=torch.bool, device=d.device), d,
                               {"method_version": VERSION, "config": asdict(config), "removed_neurons": 0})
    dtype = torch.float64 if d.dtype == torch.float64 else torch.float32
    x = calibration.detach().to(device=d.device, dtype=dtype)
    if x.ndim != 2 or not len(x) or x.shape[1] != width or not torch.isfinite(x).all():
        raise ValueError("finite unpadded ambient token matrix required")
    h0 = x.T @ x / len(x)
    support = h0.diagonal() > config.support_threshold * h0.diagonal().max()
    scale = torch.ones_like(d, dtype=dtype)
    diag = {"gamma": 0., "averaging": "ambient_token_mean"}
    if config.metric != "ambient" and config.alpha+config.input_alpha > 0:
        if geometric_diagonal is None:
            raise ValueError("geometric metric requires paired curvature diagonal")
        base = geometric_diagonal["base"] if isinstance(geometric_diagonal, dict) else geometric_diagonal
        scale, support, diag = geometric_scales(h0, base.to(h0), config.alpha,
                                                len(d), threshold=config.support_threshold)
        if config.input_alpha:
            if not isinstance(geometric_diagonal, dict) or "input" not in geometric_diagonal:
                raise ValueError("two-sided pruning requires paired input moments")
            input_scale, _, input_diag = geometric_scales(h0, geometric_diagonal["input"].to(h0),
                config.input_alpha, len(d), threshold=config.support_threshold)
            if input_diag["supported_geometry_trace"] <= 0:
                raise ValueError("no supported input tangent geometry")
            scale = (scale.square()+input_scale.square()-1).sqrt()
            diag.update(eta=input_diag["gamma"], input_alpha=config.input_alpha,
                        input_geometry_trace=input_diag["supported_geometry_trace"],
                        input_to_ambient_ratio=input_diag["added_to_ambient_ratio"], input_structure=config.input_structure)
        if full_metric is not None:
            full_metric = (full_metric.with_weights(diag["gamma"], diag["eta"]) if config.input_alpha
                           else full_metric.with_gamma(diag["gamma"]))
            if hasattr(full_metric, "with_support"):
                full_metric = full_metric.with_support(support)
    eligible = support if config.unsupported_policy == "protect" else torch.ones_like(support)
    if count > int(eligible.sum()):
        raise ValueError("requested deletion count exceeds supported eligible columns")
    reference = d.to(dtype) * scale
    if progress is not None:
        progress({"event": "neuron_pruning_started", "neurons": width, "requested_deletions": count,
            "unsupported_columns": int((~support).sum()), "selector": config.selector,
            "compensation": config.compensation, "ambient_tokens": len(x),
            "scale_min": float(scale.min()), "scale_max": float(scale.max()), **diag})
    if config.selector == "block_obs":
        keep, block_candidate, h, damping = select_columns(reference, h0, count=count,
            blocksize=config.blocksize, damping=config.percdamp, eligible=eligible, progress=progress)
    else:
        h, _, damping = damped_curvature(h0, config.percdamp)
        if config.selector == "greedy_reference":
            keep = greedy_columns(reference, h, count, eligible)
        else:
            if config.selector == "direct_write":
                scores = d.to(dtype).square().sum(0) * h0.diagonal()
            elif config.selector == "direct_full_metric":
                if full_metric is None:
                    if config.alpha+config.input_alpha > 0:
                        raise ValueError("full-direction selector requires frames/probes")
                    scores = d.to(dtype).square().sum(0)*h0.diagonal()
                else:
                    scores = full_metric.direct_scores(d.to(dtype), h0)
            else:
                generator = torch.Generator(device="cpu").manual_seed(config.seed)
                scores = torch.rand(width, generator=generator).to(d.device)
            scores = scores.clone()
            scores[~eligible] = float("inf")
            keep = torch.ones(width, dtype=torch.bool, device=d.device)
            keep[torch.argsort(scores, stable=True)[:count]] = False
        block_candidate = reference * keep
    def objective(value):
        residual = reference - value
        return float(((residual @ h) * residual).sum())
    before = objective(block_candidate)
    refit_start = time.monotonic()
    try:
        if config.compensation == "none":
            compensated = reference * keep
        elif config.compensation == "full_metric_refit" and config.alpha+config.input_alpha > 0:
            if full_metric is None:
                raise ValueError("full refit requires retained frames or bounded frozen probes")
            induced_ridge = damping["damping"] * scale.square()
            method = config.full_metric_solver
            if method == "auto":
                method = "direct" if d.numel() <= 2048 else "cg"
            compensated_down, full_diagnostics = full_metric.refit(d.to(dtype), keep, h0, induced_ridge,
                method=method, rtol=config.full_metric_rtol, max_iterations=config.full_metric_max_iterations,
                **({"progress": progress} if progress is not None else {}))
            if not full_diagnostics["full_metric_converged"]:
                raise RuntimeError("full-metric CG did not converge; candidate was not accepted")
            compensated = compensated_down * scale
            diag.update(full_diagnostics)
        else:
            compensated = refit_columns(reference, h, keep)
    except torch.linalg.LinAlgError:
        if _refit_retries >= 19:
            raise RuntimeError("retained-column refit failed after 20 selection restarts")
        retry = replace(config, percdamp=max(damping["effective_percdamp"]*2, 1e-8))
        if progress is not None:
            progress({"event": "neuron_refit_retry", "selection_restart": _refit_retries+1,
                      "next_percdamp": retry.percdamp})
        return prune_triplet(triplet, calibration, retry, geometric_diagonal=geometric_diagonal,
            full_metric=full_metric, _requested_percdamp=config.percdamp if _requested_percdamp is None else _requested_percdamp,
            _refit_retries=_refit_retries+1, progress=progress)
    residual = reference - compensated
    gradient = (residual @ h)[:, keep]
    stationarity = float(gradient.norm() / ((reference @ h).norm().clamp_min(1e-30)))
    diag.update(method_version=VERSION, config=asdict(config), requested_deletions=count,
                removed_neurons=int((~keep).sum()), kept_indices=torch.where(keep)[0].tolist(),
                removed_indices=torch.where(~keep)[0].tolist(), unsupported_columns=int((~support).sum()),
                surrogate_pre_refit_objective=before, surrogate_post_refit_objective=objective(compensated),
                surrogate_relative_stationarity=stationarity, ridge="scaled_coordinate_delta_squared_weight_residual",
                ambient_tokens=len(x), refit_seconds=time.monotonic()-refit_start,
                selection_and_refit_seconds=time.monotonic()-start, **damping)
    candidate = build_candidate(triplet, keep, compensated / scale, diag)
    candidate.diagnostics["selection_restarts_for_refit"] = _refit_retries
    if _requested_percdamp is not None:
        candidate.diagnostics["requested_percdamp"] = _requested_percdamp
        candidate.diagnostics["config"]["percdamp"] = _requested_percdamp
    architectural = 3 * d.shape[0] * count
    removed = sum(int(((original != 0) & (value == 0)).sum()) for original, value in
                  zip((g, u, d), (candidate.gate, candidate.up, candidate.down)))
    incidental = int(((d[:, keep] != 0) & (candidate.down[:, keep] == 0)).sum())
    candidate.diagnostics.update(architectural_removed_slots=architectural,
        eligible_slots=3*d.numel(), original_nonzero_removed=removed, incidental_down_zeros=incidental)
    if progress is not None:
        progress({"event": "neuron_pruning_completed", "removed_neurons": int((~keep).sum()),
            "surrogate_pre_refit_objective": before, "surrogate_post_refit_objective": diag["surrogate_post_refit_objective"],
            "surrogate_relative_stationarity": stationarity, "refit_seconds": diag["refit_seconds"],
            "selection_and_refit_seconds": diag["selection_and_refit_seconds"],
            "surrogate_objective_definition": "scaled_weight_residual_damped_quadratic_without_half", **damping})
    return candidate


def apply_candidate(triplet, candidate):
    """Validate all tensors before mutation, and rollback the entire triplet."""
    g, u, d = triplet.weights
    keep = candidate.keep.to(g.device)
    values = (candidate.gate, candidate.up, candidate.down)
    if keep.dtype != torch.bool or keep.shape != (len(g),) or not keep.any():
        raise ValueError("invalid candidate mask")
    for old, value in zip((g, u, d), values):
        if value.shape != old.shape or value.dtype != old.dtype or not torch.isfinite(value).all():
            raise ValueError("candidate shape, deployment dtype or finite invariant failed")
    if (candidate.gate[~keep].any() or candidate.up[~keep].any() or candidate.down[:, ~keep].any()
            or not torch.equal(candidate.gate[keep], g[keep]) or not torch.equal(candidate.up[keep], u[keep])):
        raise ValueError("triplet deletion/survivor invariants failed")
    originals = tuple(t.clone() for t in (g, u, d))
    try:
        with torch.no_grad():
            for module, value in zip((triplet.gate, triplet.up, triplet.down), values):
                module.weight.copy_(value)
    except BaseException:
        with torch.no_grad():
            for module, value in zip((triplet.gate, triplet.up, triplet.down), originals):
                module.weight.copy_(value)
        raise
