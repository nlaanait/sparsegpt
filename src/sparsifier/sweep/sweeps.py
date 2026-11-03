"""Sweep records application code builds, and the check a runner requires.

A ``SweepSpec`` is data. ``validate`` resolves its models in the catalog,
requires one shared runner, one family, and a stored parameter count, and returns a
``CheckedSweep``. Command builders take that checked value.

A ``SweepRecord`` is the family record experiments store. The runner and the
results directories are already filled in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sparsifier.models import ModelRecord, measured_stored_counts, resolve
from sparsifier.sweep.run import CORPORA

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class SweepSpec:
    """One named measurement. Application code fills this in."""

    name: str
    models: tuple[str, ...]
    calibration: str
    eval_datasets: tuple[str, ...]
    log_infix: str
    attention: str
    holdout_suffix: str
    plot: bool = True
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"sweep name must be a results directory, got {self.name!r}")
        if not isinstance(self.models, tuple) or not self.models:
            raise ValueError("models must be a non-empty tuple of Hugging Face ids")
        if len(self.models) != len(set(self.models)):
            raise ValueError(f"sweep {self.name} lists a model more than once")
        for model in self.models:
            if not isinstance(model, str) or not model or model.strip() != model:
                raise ValueError(f"model id must be a non-empty string, got {model!r}")
        if self.calibration not in CORPORA:
            raise ValueError(f"unknown calibration corpus {self.calibration!r}")
        if not isinstance(self.eval_datasets, tuple) or not self.eval_datasets:
            raise ValueError("eval_datasets must be a non-empty tuple")
        unknown = set(self.eval_datasets) - CORPORA
        if unknown:
            raise ValueError(f"unknown eval corpora: {sorted(unknown)}")
        for label, value in (
            ("log_infix", self.log_infix),
            ("attention", self.attention),
            ("holdout_suffix", self.holdout_suffix),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{label} must be a non-empty string")
        if not isinstance(self.plot, bool):
            raise ValueError("plot must be a bool")
        if not isinstance(self.extra_args, tuple) or not all(isinstance(arg, str) for arg in self.extra_args):
            raise ValueError("extra_args must be a tuple of strings")


@dataclass(frozen=True)
class SweepRecord:
    """One family sweep whose runner and result paths are already chosen."""

    name: str
    models: tuple[str, ...]
    runner: str
    extra_args: tuple[str, ...]
    result_dir: Path
    log_dir: Path
    log_infix: str
    attention: str
    holdout_suffix: str
    plot: bool = True
    # SparseGPT calibration corpus (positional dataset arg to the runner).
    calibration: str = "c4"
    # Perplexity corpora after prune (or dense). Order is the sweep order.
    eval_datasets: tuple[str, ...] = ("ptb", "wikitext2")

    @classmethod
    def from_spec(cls, spec: SweepSpec, root: Path) -> SweepRecord:
        """Fill the runner and the results directories from ``validate``."""
        checked = validate(spec, root)
        return cls(
            name=spec.name,
            models=spec.models,
            runner=checked.runner,
            extra_args=spec.extra_args,
            result_dir=checked.result_dir,
            log_dir=checked.log_dir,
            log_infix=spec.log_infix,
            attention=spec.attention,
            holdout_suffix=spec.holdout_suffix,
            plot=spec.plot,
            calibration=spec.calibration,
            eval_datasets=spec.eval_datasets,
        )

    def stored_params(self) -> dict[str, float]:
        """Catalog stored N, replaced once that checkpoint's index is cached."""
        return measured_stored_counts(self.models)


@dataclass(frozen=True)
class CheckedSweep:
    """A spec that has passed ``validate``. The runner reads this, not the raw spec."""

    spec: SweepSpec
    runner: str
    result_dir: Path
    log_dir: Path

    def __getattr__(self, name: str):
        """Attribute reads fall through to the underlying ``SweepSpec``."""
        return getattr(self.spec, name)

    def record(self, model: str) -> ModelRecord:
        """Catalog row for one model in this sweep."""
        if model not in self.spec.models:
            raise ValueError(f"{model} is not in sweep {self.spec.name}")
        return resolve(model)

    def prepare(self) -> None:
        """Create the results directory and its log directory."""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def repo_root(start: Path | None = None) -> Path:
    """Directory that contains ``pixi.toml``, walking up from ``start`` or this file."""
    here = Path(__file__ if start is None else start).resolve()
    for path in (here, *here.parents):
        if (path / "pixi.toml").is_file():
            return path
    raise FileNotFoundError(f"pixi.toml not found above {here}")


def validate(spec: SweepSpec, root: Path) -> CheckedSweep:
    """Resolve models and paths. Raise before a runner is launched."""
    if not isinstance(spec, SweepSpec):
        raise TypeError(f"expected SweepSpec, got {type(spec).__name__}")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"sweep root does not exist: {root}")
    records: list[ModelRecord] = []
    for model in spec.models:
        try:
            record = resolve(model)
        except LookupError as exc:
            raise ValueError(str(exc)) from exc
        if record.stored_b is None:
            raise ValueError(f"{model} has no stored parameter count")
        records.append(record)
    runners = {record.runner for record in records}
    if len(runners) != 1:
        found = ", ".join(f"{record.model_id} -> {record.runner}" for record in records)
        raise ValueError(f"sweep {spec.name} must use one runner; found {found}")
    families = {record.family for record in records}
    if len(families) != 1:
        found = ", ".join(f"{record.model_id} -> {record.family}" for record in records)
        raise ValueError(f"sweep {spec.name} must use one family; found {found}")
    result_dir = root / "results" / spec.name
    return CheckedSweep(
        spec=spec,
        runner=next(iter(runners)),
        result_dir=result_dir,
        log_dir=result_dir / "logs",
    )
