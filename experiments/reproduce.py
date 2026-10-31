"""Commands and figure rewrites for a registered sweep.

Nothing here loads a model. ``plan`` prints the prune and eval commands that
``analysis.sweep.run_sweep`` would launch for points that are still missing.
A point counts as measured when ``analysis.series.load_rows`` can see it:
the per-model sparsity file, a Qwen 3.5 family bundle, or a likelihood dump.
``figures`` rewrites the three sparsity pages and the family principle page
from those same rows.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

from analysis.sweep import SPARSITIES, short_name, sparsity_token
from experiments.protocol import SweepSpec
from experiments.sweeps import sweeps

Done = Callable[[str, str, str, float], bool]
Cached = Callable[[str, float], bool]


@dataclass(frozen=True)
class Job:
    kind: str
    model: str
    sparsity: float
    datasets: tuple[str, ...]
    command: tuple[str, ...]


def eval_command(
    spec: SweepSpec,
    model: str,
    datasets: list[str] | tuple[str, ...],
    sparsity: float,
    *,
    load_cached: bool,
) -> list[str]:
    """Argv of one eval. Matches ``analysis.sweep.family_cmd`` for the same cache state."""
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


def prune_command(spec: SweepSpec, model: str, sparsity: float) -> list[str]:
    """Argv of one ``--skip-eval`` prune. Matches the prune branch of ``run_sweep``."""
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


def plan_jobs(spec: SweepSpec, *, done: Done, cached: Cached) -> list[Job]:
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


def write_reports(spec: SweepSpec, sizes: dict[str, float] | None = None) -> dict:
    """Refit and rewrite the three pages and the family principle page."""
    from analysis.plots import refresh_family
    from analysis.principle import write_family_page

    return {
        "law": refresh_family(spec, sizes),
        "principle": write_family_page(spec, sizes),
    }


def _spec_by_name(name: str) -> SweepSpec:
    for spec in sweeps():
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in sweeps())
    raise SystemExit(f"unknown sweep {name!r}; known: {known}")


def _done_from_disk(spec: SweepSpec) -> Done:
    """True when the figure loader already has this perplexity.

    Likelihood dumps and the Qwen 3.5 family bundles count. A per-model
    sparsity file is not required.
    """
    from analysis.series import load_rows, lookup

    rows = load_rows(spec)

    def done(model: str, dataset: str, method: str, sparsity: float) -> bool:
        if lookup(rows, model, dataset, sparsity) is None:
            return False
        if method == "dense":
            return sparsity == 0.0
        return method == "sparsegpt" and sparsity > 0.0

    return done


def _cached_from_disk(spec: SweepSpec) -> Cached:
    from analysis.sweep import checkpoint

    def cached(model: str, sparsity: float) -> bool:
        return checkpoint(model, sparsity, spec.calibration) is not None

    return cached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "plan", "figures"))
    parser.add_argument("sweep", nargs="?")
    args = parser.parse_args(argv)
    if args.action == "list":
        for spec in sweeps():
            print(f"{spec.name}\t{spec.runner}\t{spec.calibration}\t{len(spec.models)} models")
        return 0
    if not args.sweep:
        raise SystemExit(f"{args.action} requires a sweep name")
    spec = _spec_by_name(args.sweep)
    if args.action == "plan":
        jobs = plan_jobs(spec, done=_done_from_disk(spec), cached=_cached_from_disk(spec))
        if not jobs:
            print(f"{spec.name}: nothing to run")
            return 0
        for job in jobs:
            print(f"{job.kind}\t{job.model}\t{job.sparsity}\t{' '.join(job.command)}")
        return 0
    written = write_reports(spec)
    print(f"{spec.name}: law={written['law'] is not None} principle={written['principle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
