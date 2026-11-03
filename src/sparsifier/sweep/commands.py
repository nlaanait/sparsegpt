"""Argv for one sweep point, and the jobs that are still missing.

``plan_jobs`` does not read the disk. The caller supplies ``done`` and
``cached``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from sparsifier.sweep.sweeps import SweepRecord

# Unstructured tenths the family sweeps measure. Dense 0.0 is not in this list.
SPARSITIES = [round(index / 10, 1) for index in range(1, 9)]

Done = Callable[[str, str, str, float], bool]
Cached = Callable[[str, float], bool]


def short_name(model: str) -> str:
    """Hugging Face repo tail, lowercased. This is the checkpoint directory name."""
    return model.rstrip("/").split("/")[-1].lower()


def sparsity_token(sparsity: float) -> str:
    """CLI token that keeps half-steps (0.15) distinct from tenths (0.1)."""
    return f"{sparsity:.2f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Job:
    kind: str
    model: str
    sparsity: float
    datasets: tuple[str, ...]
    command: tuple[str, ...]


def eval_command(
    spec: SweepRecord,
    model: str,
    datasets: list[str] | tuple[str, ...],
    sparsity: float,
    *,
    load_cached: bool,
) -> list[str]:
    """Argv of one eval. A positive sparsity either loads the cache or saves it."""
    command = [
        sys.executable,
        "-m",
        spec.runner,
        model,
        spec.calibration,
        "--sparsity",
        sparsity_token(sparsity),
        "--eval-datasets",
        *datasets,
        *spec.extra_args,
    ]
    if sparsity > 0:
        if load_cached:
            command.append("--load-cached")
        else:
            command.extend(["--save", f"sparse_opt/{short_name(model)}"])
    return command


def prune_command(spec: SweepRecord, model: str, sparsity: float) -> list[str]:
    """Argv of one ``--skip-eval`` prune."""
    return [
        sys.executable,
        "-m",
        spec.runner,
        model,
        spec.calibration,
        "--sparsity",
        sparsity_token(sparsity),
        "--skip-eval",
        *spec.extra_args,
        "--save",
        f"sparse_opt/{short_name(model)}",
    ]


def plan_jobs(spec: SweepRecord, *, done: Done, cached: Cached) -> list[Job]:
    """Missing dense evals, then each missing sparsity: prune once, then eval.

    ``done(model, dataset, method, sparsity)`` is true when that JSON point
    exists. ``cached(model, sparsity)`` is true when the prune checkpoint
    exists. After a prune job the later evals in that sparsity load the cache.
    """
    jobs: list[Job] = []
    for model in spec.models:
        for dataset in spec.eval_datasets:
            if done(model, dataset, "dense", 0.0):
                continue
            jobs.append(
                Job(
                    "dense",
                    model,
                    0.0,
                    (dataset,),
                    tuple(eval_command(spec, model, [dataset], 0.0, load_cached=False)),
                )
            )
        for sparsity in SPARSITIES:
            have_cache = cached(model, sparsity)
            for dataset in spec.eval_datasets:
                if done(model, dataset, "sparsegpt", sparsity):
                    continue
                if not have_cache:
                    jobs.append(
                        Job(
                            "prune",
                            model,
                            sparsity,
                            (),
                            tuple(prune_command(spec, model, sparsity)),
                        )
                    )
                    have_cache = True
                jobs.append(
                    Job(
                        "eval",
                        model,
                        sparsity,
                        (dataset,),
                        tuple(
                            eval_command(
                                spec,
                                model,
                                [dataset],
                                sparsity,
                                load_cached=True,
                            )
                        ),
                    )
                )
    return jobs
