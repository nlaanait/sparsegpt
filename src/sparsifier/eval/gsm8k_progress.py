"""Read numerical progress from saved pruning stages without a model or GPU.

Use --follow for a worker launched before numerical progress logs were added.
Historical stages cannot recover iteration losses or spectra that were never
recorded; retained ranks can be read from the saved tangent bases.
"""
import argparse
import json
from pathlib import Path
import time

from sparsifier.prune.progress import compact_diagnostics


def site_record(run, index, stage):
    metrics = compact_diagnostics(stage.get("diagnostics", {}))
    triplet = stage["site"][1] == "mlp_triplet"
    atlas_path = run/f"atlas-{stage['site'][0] if triplet else index:03d}.npz"
    if atlas_path.exists():
        import numpy as np
        with np.load(atlas_path, allow_pickle=False) as saved:
            ranks = [saved[key].shape[1] for key in saved.files if "_basis_" in key]
            atlas = metrics.setdefault("atlas", {})
            atlas["retained_singular_vectors"] = sum(ranks)
            atlas["charts"] = len(ranks)
            if "fit_diagnostics" in saved:
                fit = json.loads(str(saved["fit_diagnostics"]))
                atlas["dictionary_iterations"] = fit.get("dictionary", {}).get("iterations")
    return {"event": "saved_site_diagnostics", "stage_index": index, "site": stage["site"],
            "removed_weights": stage.get("removed", metrics.get("original_nonzero_removed")),
            "metrics": metrics, "provenance": "saved_stage_and_atlas_no_refit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=5.)
    args = parser.parse_args()
    if args.last < 1 or not 0 < args.interval <= 60:
        parser.error("positive --last and an --interval in (0, 60] required")
    seen = None
    while True:
        path = args.run/"report.json"
        if not path.exists():
            if not args.follow:
                parser.error(f"no report at {path}")
        else:
            report = json.loads(path.read_text())
            stages = report.get("stages", [])
            if seen is None:
                seen = max(0, len(stages)-args.last)
            for index in range(seen, len(stages)):
                print(json.dumps(site_record(args.run, index, stages[index]), allow_nan=False), flush=True)
            seen = len(stages)
            if not args.follow or report.get("complete"):
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
