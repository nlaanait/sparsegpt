"""Frozen-J input-tangent pullbacks with declared sequence covariance."""
from __future__ import annotations

import torch

STRUCTURES = ("independent_tokens", "shared_directions")
VERSION = "two-sided-frozen-j-v1"


def input_probe_moments(event, basis, structure="independent_tokens"):
    """Return an exact input-covariance diagonal and unsquared factor events.

    Independent token coefficients give block-diagonal input covariance.
    Shared direction coefficients couple tokens and require aligned columns.
    Output probes and independent examples must never be summed together.
    """
    if structure not in STRUCTURES:
        raise ValueError("unknown input tangent sequence structure")
    g = event["gradient"].detach().to(device="cpu", dtype=torch.float64)
    u = basis.detach().to(device="cpu", dtype=torch.float64)
    if g.ndim != 2 or u.ndim != 3 or g.shape[0] != u.shape[0] or not torch.isfinite(u).all():
        raise ValueError("input frames must be finite [tokens, inputs, rank]")
    metadata = {k: v for k, v in event.items() if k not in ("contraction", "inputs")}
    metadata.update(term="input", input_structure=structure)
    if structure == "independent_tokens":
        diagonal = g.square().T @ u.square().sum(-1)
        return diagonal, [dict(metadata, gradient=event["gradient"], basis=basis.detach().cpu(), kind="independent")]
    diagonal = torch.zeros(g.shape[1], u.shape[1], dtype=torch.float64)
    events = []
    for direction in range(u.shape[-1]):
        z = g.T @ u[:, :, direction]
        diagonal.add_(z.square())
        events.append(dict(metadata, contraction=z, inputs=basis[:, :, direction].detach().cpu(),
                           kind="weight", input_direction=direction))
    return diagonal, events
