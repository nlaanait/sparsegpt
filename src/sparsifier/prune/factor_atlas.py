"""Calibration-only factor-specific local charts for the GSM8K experiment.

Sparse factors may coexist; each active factor selects its own local chart.
The block dictionary is an approximation, not the published BSF estimator.
No held-out target differences participate in chart assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from sparsifier.prune.geometry_features import BlockDictionary


def orth(matrix: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix.copy()
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(s) or s[0] <= 1e-12:
        return matrix[:, :0].copy()
    return u[:, s > tolerance * s[0]]


@dataclass
class FactorCharts:
    centers: np.ndarray
    bases: list[np.ndarray]
    anchors: np.ndarray
    counts: np.ndarray
    global_basis: np.ndarray
    rank_matched_basis: np.ndarray


class FactorAtlas:
    def __init__(self, groups=8, width=4, active=2, charts=4, rank=1, seed=0):
        if not 1 <= rank < width or charts < 1:
            raise ValueError("local rank must be positive and smaller than block width")
        self.dictionary = BlockDictionary(groups, width, active, seed)
        self.charts, self.rank, self.seed = charts, rank, seed
        self.factors: list[FactorCharts] = []

    def fit(self, outputs: np.ndarray, iterations=20):
        y = np.asarray(outputs, dtype=np.float64)
        self.dictionary.fit(y.T, iterations=iterations)
        return self.fit_charts(y)

    def fit_charts(self, outputs: np.ndarray):
        """Fit local structure using an already-fitted dictionary (also test hook)."""
        y = np.asarray(outputs, dtype=np.float64)
        if y.ndim != 2 or not np.isfinite(y).all():
            raise ValueError("finite sample-major outputs required")
        model = self.dictionary
        z = model.encode(y.T).reshape(model.groups, model.width, -1)
        self.factors = []
        rng = np.random.default_rng(self.seed)
        for factor in range(model.groups):
            block = model.dictionary[:, factor * model.width:(factor + 1) * model.width]
            codes = z[factor].T
            codes = codes[np.linalg.norm(codes, axis=1) > 1e-8]
            minimum = max(4, self.rank + 2)
            if len(codes) < minimum:
                # Keep the factor ID and abstain; never invent tangent columns.
                self.factors.append(FactorCharts(np.empty((0, model.width)), [],
                    np.empty((0, y.shape[1])), np.array([], dtype=np.int64),
                    orth(block), np.empty((y.shape[1], 0))))
                continue
            # Distances in factor contribution space, expressed in a small QR frame.
            _, r = np.linalg.qr(block, mode="reduced")
            coordinates = codes @ r.T
            count = min(self.charts, len(codes) // minimum)
            while count:
                centers = coordinates[rng.choice(len(codes), count, replace=False)].copy()
                for _ in range(30):
                    assignment = ((coordinates[:, None] - centers[None]) ** 2).sum(2).argmin(1)
                    if any(not np.any(assignment == k) for k in range(count)):
                        break
                    updated = np.stack([coordinates[assignment == k].mean(0) for k in range(count)])
                    if np.allclose(updated, centers):
                        centers = updated
                        break
                    centers = updated
                assignment = ((coordinates[:, None] - centers[None]) ** 2).sum(2).argmin(1)
                counts = np.bincount(assignment, minlength=count)
                if counts.min() >= minimum:
                    break
                count -= 1
            if not count:
                raise ValueError(f"factor {factor} cannot support a chart")
            bases, anchors = [], []
            for chart in range(count):
                local = codes[assignment == chart]
                centered = local - local.mean(0)
                _, singular, vt = np.linalg.svd(centered @ r.T, full_matrices=False)
                available = int((singular > max(singular[0] * 1e-6, 1e-12)).sum())
                if available < self.rank:
                    raise ValueError(f"factor {factor}, chart {chart} has degenerate local variance")
                q, _ = np.linalg.qr(block, mode="reduced")
                bases.append(orth(q @ vt[:self.rank].T))
                anchors.append(block @ local.mean(0))
            q, _ = np.linalg.qr(block, mode="reduced")
            _, _, vt = np.linalg.svd((codes - codes.mean(0)) @ r.T, full_matrices=False)
            matched = orth(q @ vt[:self.rank].T)
            self.factors.append(FactorCharts(centers, bases, np.stack(anchors), counts,
                                              orth(block), matched))
        return self

    def assign(self, outputs: np.ndarray):
        model = self.dictionary
        z = model.encode(np.asarray(outputs).T).reshape(model.groups, model.width, -1)
        active = np.linalg.norm(z, axis=1).T > 1e-8
        assignments = np.full(active.shape, -1, dtype=np.int64)
        for i, factor in enumerate(self.factors):
            if not len(factor.bases):
                continue
            block = model.dictionary[:, i * model.width:(i + 1) * model.width]
            _, r = np.linalg.qr(block, mode="reduced")
            coordinates = z[i].T @ r.T
            chosen = ((coordinates[:, None] - factor.centers[None]) ** 2).sum(2).argmin(1)
            assignments[active[:, i], i] = chosen[active[:, i]]
        return z, active, assignments

    def frames(self, outputs, mode="local", seed=0):
        if mode not in ("local", "shuffled", "factor", "rank_matched", "global"):
            raise ValueError(f"unknown atlas mode: {mode}")
        _, active, assignments = self.assign(outputs)
        if mode == "shuffled":
            rng = np.random.default_rng(seed)
            for factor in range(active.shape[1]):
                ix = np.flatnonzero(active[:, factor])
                assignments[ix, factor] = rng.permutation(assignments[ix, factor])
        if mode == "global":
            union = orth(np.concatenate([f.global_basis for f in self.factors], axis=1))
            return [union] * len(outputs), active, assignments
        frames = []
        for n in range(len(outputs)):
            bases = []
            for i in np.flatnonzero(active[n]):
                factor = self.factors[i]
                if mode == "factor":
                    bases.append(factor.global_basis)
                elif mode == "rank_matched":
                    bases.append(factor.rank_matched_basis)
                else:
                    if assignments[n, i] >= 0:
                        bases.append(factor.bases[assignments[n, i]])
            frames.append(orth(np.concatenate(bases, axis=1)) if bases else np.empty((outputs.shape[1], 0)))
        return frames, active, assignments

    def diagnostics(self, outputs):
        y = np.asarray(outputs)
        z, active, assignments = self.assign(y)
        reconstruction = (self.dictionary.dictionary @ z.reshape(-1, len(y)) + self.dictionary.mean).T
        return {"estimator": "approximate_block_dictionary_with_within_factor_charts",
                "relative_reconstruction_error": float(np.square(y - reconstruction).sum() / max(np.square(y).sum(), 1e-12)),
                "mean_active_factors": float(active.sum(1).mean()),
                "unsupported_factors": [i for i, f in enumerate(self.factors) if not f.bases],
                "active_factor_chart_coverage": float((assignments >= 0).sum() / max(active.sum(), 1)),
                "chart_fit_counts": [f.counts.tolist() for f in self.factors],
                "chart_eval_counts": [np.bincount(assignments[assignments[:, i] >= 0, i], minlength=len(f.bases)).tolist()
                                      for i, f in enumerate(self.factors)],
                "storage_scalars": int(self.dictionary.dictionary.size + self.dictionary.mean.size +
                    sum(f.centers.size + f.anchors.size + sum(b.size for b in f.bases) for f in self.factors))}

    def save(self, path: Path):
        config = {"groups": self.dictionary.groups, "width": self.dictionary.width,
                  "active": self.dictionary.active, "charts": self.charts,
                  "rank": self.rank, "seed": self.seed}
        arrays = {"config": np.array(json.dumps(config)), "dictionary": self.dictionary.dictionary,
                  "mean": self.dictionary.mean}
        for i, f in enumerate(self.factors):
            for name in ("centers", "anchors", "counts", "global_basis", "rank_matched_basis"):
                arrays[f"{i}_{name}"] = getattr(f, name)
            for j, basis in enumerate(f.bases):
                arrays[f"{i}_basis_{j}"] = basis
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as saved:
            atlas = cls(**json.loads(str(saved["config"])))
            atlas.dictionary.dictionary = saved["dictionary"].copy()
            atlas.dictionary.mean = saved["mean"].copy()
            for i in range(atlas.dictionary.groups):
                centers = saved[f"{i}_centers"].copy()
                atlas.factors.append(FactorCharts(
                    centers, [saved[f"{i}_basis_{j}"].copy() for j in range(len(centers))],
                    saved[f"{i}_anchors"].copy(), saved[f"{i}_counts"].copy(),
                    saved[f"{i}_global_basis"].copy(), saved[f"{i}_rank_matched_basis"].copy()))
        return atlas


def projector_diagonals(frames):
    return np.stack([np.square(q).sum(1) for q in frames])


def geometry_coupling(inputs, frames):
    """Sample-matched row-diagonal approximation, not the full output metric."""
    x = np.asarray(inputs, dtype=np.float64)
    diag = projector_diagonals(frames)
    if x.shape[0] != diag.shape[0]:
        raise ValueError("input/projector sample axes differ")
    energy = np.square(x)
    return (diag.T @ energy) / np.maximum(energy.sum(0), 1e-12)


def geometric_error(residual, frames):
    if len(residual) != len(frames):
        raise ValueError("residual/frame samples differ")
    return float(np.mean([np.square(r @ q).sum() for r, q in zip(residual, frames)]))
