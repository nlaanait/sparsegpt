"""Capture and store per-linear input differences for Variant A.

``δφ`` is the layer input at the edited token on the left string minus the
same position on the right string. One row per knob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def capture_position_mean(
    modules: dict[str, nn.Module],
    run,
    token_ids: torch.Tensor,
    positions: tuple[int, ...] | list[int],
) -> dict[str, torch.Tensor]:
    """Mean layer input over ``positions`` for one forward."""
    if token_ids.ndim != 2:
        raise ValueError("token ids must have shape [1, sequence]")
    if not positions:
        raise ValueError("at least one position is required")
    if any(index < 0 or index >= token_ids.shape[1] for index in positions):
        raise ValueError("a presence position falls outside the token sequence")
    captured: dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def hook(_module, inputs):
            activation = inputs[0]
            if activation.ndim < 2:
                raise ValueError(f"{name} input must include a sequence axis")
            flat = activation.reshape(-1, activation.shape[-2], activation.shape[-1])
            if max(positions) >= flat.shape[1]:
                raise ValueError(f"{name} sequence is shorter than a presence position")
            selected = flat[0, list(positions)].float()
            captured[name] = selected.mean(dim=0).detach().cpu()

        return hook

    handles = [module.register_forward_pre_hook(_hook(name)) for name, module in modules.items()]
    try:
        run(token_ids)
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name in modules if name not in captured]
    if missing:
        raise RuntimeError(f"presence forward missed {missing[0]}")
    return captured


def capture_edit_deltas(
    modules: dict[str, nn.Module],
    run,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    edit_index: int,
) -> dict[str, torch.Tensor]:
    """Return ``name -> δφ`` at ``edit_index`` for each hooked linear.

    ``run`` is called twice, once per id tensor. Each id tensor is a long
    tensor of shape ``[1, sequence]``. The linear input's sequence axis is
    the second-to-last dimension.
    """
    if left_ids.ndim != 2 or right_ids.ndim != 2:
        raise ValueError("token ids must have shape [1, sequence]")
    if edit_index < 0 or edit_index >= left_ids.shape[1] or edit_index >= right_ids.shape[1]:
        raise ValueError("edit_index falls outside the token sequence")

    captured: dict[str, dict[str, torch.Tensor]] = {}

    def _hook(name: str, side: str):
        def hook(_module, inputs):
            activation = inputs[0]
            if activation.ndim < 2:
                raise ValueError(f"{name} input must include a sequence axis")
            flat = activation.reshape(-1, activation.shape[-2], activation.shape[-1])
            if edit_index >= flat.shape[1]:
                raise ValueError(f"{name} sequence is shorter than edit_index")
            captured.setdefault(name, {})[side] = flat[0, edit_index].detach().float().cpu()

        return hook

    handles = [
        module.register_forward_pre_hook(_hook(name, "left"))
        for name, module in modules.items()
    ]
    try:
        run(left_ids)
    finally:
        for handle in handles:
            handle.remove()

    handles = [
        module.register_forward_pre_hook(_hook(name, "right"))
        for name, module in modules.items()
    ]
    try:
        run(right_ids)
    finally:
        for handle in handles:
            handle.remove()

    deltas = {}
    for name in modules:
        sides = captured.get(name, {})
        if "left" not in sides or "right" not in sides:
            raise RuntimeError(f"{name} did not see both sides of the pair")
        deltas[name] = sides["left"] - sides["right"]
    return deltas


def save_pair_deltas(
    root: Path,
    deltas: dict[str, torch.Tensor],
    knob_ids: dict[str, torch.Tensor] | None,
) -> None:
    """Write ``{block}/{module}.npy`` and, when given, ``.knobs.npy``.

    Keys are ``"{block}/{module}"``.
    """
    root = Path(root)
    for key, rows in deltas.items():
        block, module = _split_key(key)
        directory = root / block
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{module}.npy", rows.detach().float().cpu().numpy())
        if knob_ids is not None:
            ids = knob_ids[key]
            np.save(directory / f"{module}.knobs.npy", ids.detach().cpu().numpy().astype(np.int64))


def load_knob_ids(path: Path) -> torch.Tensor | None:
    knob_path = path.with_name(path.stem + ".knobs.npy")
    if not knob_path.is_file():
        return None
    return torch.as_tensor(np.load(knob_path, allow_pickle=False)).reshape(-1).to(dtype=torch.int64)


def _split_key(key: str) -> tuple[str, str]:
    block, sep, module = key.partition("/")
    if sep != "/" or not block or not module or "/" in module:
        raise ValueError(f"pair key must be '{{block}}/{{module}}', got {key!r}")
    return block, module
