"""One checkpoint: which pruner, which metric, which calibration corpus.

``run`` is not wired to a family yet. Sweep commands stay
``python -m sparsegpt.<family>`` until those loops move here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
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
    prunen: int = 0
    prunem: int = 0
    blocksize: int = 128
    geometry_alpha: float = 0.0
    geometry_probes: Path | None = None
    pruner_options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.calibration not in {"wikitext2", "ptb", "c4"}:
            raise ValueError(f"unknown calibration corpus {self.calibration!r}")
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        if self.nsamples <= 0 or self.seqlen <= 0:
            raise ValueError("nsamples and seqlen must be positive")
        unknown = set(self.eval_datasets) - {"wikitext2", "ptb", "c4"}
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
