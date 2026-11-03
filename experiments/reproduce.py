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

from experiments.sweeps import sweeps
from sparsifier.sweep.commands import Cached, Done, plan_jobs
from sparsifier.sweep import SweepRecord


def write_reports(spec: SweepRecord, sizes: dict[str, float] | None = None) -> dict:
    """Refit and rewrite the three pages and the family principle page."""
    from analysis.plots import refresh_family
    from analysis.principle import write_family_page

    return {
        "law": refresh_family(spec, sizes),
        "principle": write_family_page(spec, sizes),
    }


def _spec_by_name(name: str) -> SweepRecord:
    for spec in sweeps():
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in sweeps())
    raise SystemExit(f"unknown sweep {name!r}; known: {known}")


def _done_from_disk(spec: SweepRecord) -> Done:
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


def _cached_from_disk(spec: SweepRecord) -> Cached:
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
