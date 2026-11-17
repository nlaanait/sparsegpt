"""Fit local input charts on geometry-fit readouts, then freeze assignments."""
from __future__ import annotations

import numpy as np
import torch
from sparsifier.prune.factor_atlas import FactorAtlas
from sparsifier.prune.gram_dictionary import GramBlockDictionary


def fit_input_tangents(readouts, rows, path, seed, structure="independent_tokens", *, progress=None):
    from sparsifier.prune.two_sided import STRUCTURES
    if structure not in STRUCTURES or any(r["split"] not in ("geometry_fit", "pruning_calibration") for r in rows):
        raise ValueError("input charts require calibration rows and declared sequence structure")
    fit = np.array([row["split"] == "geometry_fit" for row in rows])
    if len(readouts) != len(rows) or not fit.any():
        raise ValueError("aligned geometry-fit input readouts required")
    atlas = FactorAtlas(8, 4, 2, 4, 1, seed)
    atlas.dictionary = GramBlockDictionary(8, 4, 2, seed)
    atlas.fit(readouts[fit], **({"progress": progress} if progress is not None else {}))
    atlas.save(path)
    reference = (np.linalg.qr(atlas.dictionary.dictionary, mode="reduced")[0]
                 if structure == "shared_directions" else None)

    def basis(inputs):
        flat = inputs.detach().reshape(-1, inputs.shape[-1]).float().cpu().numpy()
        frames, _, _ = atlas.frames(flat, mode="local")
        rank = max((q.shape[1] for q in frames), default=0)
        result = torch.zeros(len(flat), inputs.shape[-1], rank)
        for n, frame in enumerate(frames):
            if reference is not None and frame.shape[1]:
                # Declare cross-token column identity using a frozen fit-only
                # dictionary frame. Rotation preserves each local projector.
                left, _, right = np.linalg.svd(frame.T@reference[:, :frame.shape[1]], full_matrices=False)
                frame = frame@(left@right)
            result[n, :, :frame.shape[1]] = torch.from_numpy(frame).float()
        return result.reshape(*inputs.shape[:-1], inputs.shape[-1], rank)
    return basis
