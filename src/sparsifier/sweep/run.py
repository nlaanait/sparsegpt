"""One checkpoint: which pruner, which metric, which calibration corpus.

A sweep validates these configs. The Click command in ``sparsifier.cli``
runs one checkpoint.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

# Corpora a runner accepts as calibration or eval. Sweep validation uses this set.
CORPORA = frozenset({"wikitext2", "ptb", "c4"})


@dataclass(frozen=True)
class RunConfig:
    """One checkpoint prune.

    ``zeros_per_group`` / ``group_size`` are N:M sparsity: zero that many
    weights in every block of ``group_size`` consecutive columns. Both zero
    means unstructured ``sparsity``.
    Geometry copies ``geometry_alpha``/``geometry_probes`` into
    ``pruner_options``. Positive ``tangent_alpha`` selects the output-tangent
    mask heuristic and copies its estimator settings into ``pruner_options``.
    """

    model: str
    calibration: str
    sparsity: float = 0.0
    eval_datasets: tuple[str, ...] = ("wikitext2",)
    pruner: str = "sparsegpt"
    metric: str = "perplexity"
    seed: int = 0
    nsamples: int = 128
    seqlen: int = 2048
    percdamp: float = 0.01
    zeros_per_group: int = 0
    group_size: int = 0
    blocksize: int = 128
    geometry_alpha: float = 0.0
    geometry_probes: Path | None = None
    tangent_alpha: float = 0.0
    tangent_charts: int = 8
    tangent_rank: int = 2
    tangent_samples: int = 512
    # Layer filter and true-sequential order. The Click namespace carries the
    # same fields; a bare RunConfig prunes every decoder linear in one group.
    minlayer: int = -1
    maxlayer: int = 1000
    prune_only: str = ""
    invert: bool = False
    true_sequential: bool = False
    pruner_options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.calibration not in CORPORA:
            raise ValueError(f"unknown calibration corpus {self.calibration!r}")
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        if self.nsamples <= 0 or self.seqlen <= 0:
            raise ValueError("nsamples and seqlen must be positive")
        unknown = set(self.eval_datasets) - CORPORA
        if unknown:
            raise ValueError(f"unknown eval corpora: {sorted(unknown)}")
        if not 0.0 <= self.tangent_alpha <= 1.0:
            raise ValueError("tangent alpha must be in [0, 1]")
        if self.tangent_alpha > 0.0 and self.geometry_alpha > 0.0:
            raise ValueError("input geometry and output tangent modes are mutually exclusive")
        if self.tangent_alpha > 0.0:
            if self.tangent_charts < 1 or self.tangent_rank < 1 or self.tangent_samples < 3:
                raise ValueError("tangent charts/rank must be positive and samples >= 3")
            if self.pruner not in {"sparsegpt", "tangent"}:
                raise ValueError("tangent alpha requires the tangent pruner")
            object.__setattr__(self, "pruner", "tangent")
            options = {
                "alpha": self.tangent_alpha,
                "charts": self.tangent_charts,
                "rank": self.tangent_rank,
                "calibration_samples": self.tangent_samples,
                **self.pruner_options,
            }
            object.__setattr__(self, "pruner_options", options)
        elif self.pruner == "tangent":
            raise ValueError("tangent pruner requires positive tangent_alpha")
        elif self.pruner == "geometry":
            options = {
                "alpha": self.geometry_alpha,
                "probes": self.geometry_probes,
                **self.pruner_options,
            }
            object.__setattr__(self, "pruner_options", options)
        elif self.pruner_options:
            raise ValueError(f"{self.pruner} does not take pruner options")


def default_prune_args(model: str, sparsity: float, dataset: str = "c4") -> Namespace:
    """Namespace the cache matches. Eval-only flags stay at their defaults."""
    return Namespace(
        model=model,
        dataset=dataset,
        seed=0,
        nsamples=128,
        percdamp=0.01,
        sparsity=sparsity,
        zeros_per_group=0,
        group_size=0,
        blocksize=128,
        gmp=False,
        wbits=16,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        true_sequential=False,
        geometry_alpha=0.0,
        geometry_probes=None,
    )


def default_prune_config(model: str, sparsity: float, dataset: str = "c4") -> dict:
    """Cache key for one unstructured SparseGPT checkpoint."""
    from sparsifier.data.cache import prune_config_from_args

    return prune_config_from_args(default_prune_args(model, sparsity, dataset))
