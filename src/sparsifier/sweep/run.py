"""One checkpoint: which pruner, which metric, which calibration corpus.

A sweep validates these configs. The Click command in ``sparsifier.cli``
runs one checkpoint.
"""

from __future__ import annotations

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
    A geometry pruner copies ``geometry_alpha`` and ``geometry_probes`` into
    ``pruner_options`` so ``get_pruner`` can take them as keyword arguments.
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
        if self.pruner == "geometry":
            options = {
                "alpha": self.geometry_alpha,
                "probes": self.geometry_probes,
                **self.pruner_options,
            }
            object.__setattr__(self, "pruner_options", options)
        elif self.pruner_options:
            raise ValueError(f"{self.pruner} does not take pruner options")
