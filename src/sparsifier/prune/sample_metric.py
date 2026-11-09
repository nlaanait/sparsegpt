"""Small exact quadratic reference for sample-dependent output metrics.

This is a verification tool, not a full-layer dense Hessian implementation.
Parameter vectorization is row-major: [output, input].
"""
from __future__ import annotations

import torch


def quadratic_hessian(ambient, geometric_inputs, metrics, gamma, damping=0.):
    x, z, m = ambient.double(), geometric_inputs.double(), metrics.double()
    if x.ndim != 2 or z.ndim != 2 or z.shape[1] != x.shape[1]:
        raise ValueError("ambient and geometric input axes must agree")
    if m.ndim != 3 or m.shape[0] != len(z) or m.shape[1] != m.shape[2]:
        raise ValueError("one square output metric per geometric sample required")
    if gamma < 0 or damping < 0 or not len(x):
        raise ValueError("invalid gamma/damping/sample count")
    dout, din = m.shape[1], x.shape[1]
    if dout * din > 2048:
        raise ValueError("exact reference limited to 2048 parameters")
    if not torch.allclose(m, m.transpose(1, 2)):
        raise ValueError("metrics must be symmetric")
    if float(torch.linalg.eigvalsh(m).min()) < -1e-8:
        raise ValueError("metrics must be positive semidefinite")
    h = torch.kron(torch.eye(dout, dtype=torch.float64), (x.T @ x / len(x)).contiguous())
    h += gamma * torch.einsum("nab,nc,nd->acbd", m, z, z).reshape(dout*din, dout*din) / len(x)
    h.diagonal().add_(damping)
    return h


def refit(weight, keep, hessian):
    w = weight.double().flatten()
    free = keep.bool().flatten()
    if hessian.shape != (w.numel(), w.numel()):
        raise ValueError("Hessian dimensions do not match weights")
    residual = torch.where(free, 0., w)
    if free.any():
        hff = hessian[free][:, free]
        residual[free] = torch.linalg.solve(hff, -(hessian @ residual)[free])
    candidate = (w - residual).reshape_as(weight)
    candidate.reshape(-1)[~free] = 0
    return candidate


def objective(weight, candidate, hessian):
    delta = (weight.double() - candidate.double()).flatten()
    return float(delta @ hessian @ delta)
