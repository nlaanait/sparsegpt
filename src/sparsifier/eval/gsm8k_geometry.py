"""CPU factor-atlas fit and held-out diagnostics from recorded Qwen readouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.prune.factor_atlas import FactorAtlas


def load_capture(directory: Path):
    config = json.loads((directory / "config.json").read_text())
    index = json.loads((directory / "index.json").read_text())
    config_hash = digest(config)
    if not index.get("complete") or index["config_sha256"] != config_hash:
        raise ValueError("capture is incomplete or config hash differs")
    by_id, inputs, outputs = {}, [], []
    for i, row in enumerate(index["rows"]):
        with np.load(directory / row["file"], allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            if metadata["id"] != row["id"] or metadata["config_sha256"] != config_hash:
                raise ValueError("per-example capture provenance mismatch")
            inputs.append(arrays["x"].copy())
            outputs.append(arrays["y"].copy())
        by_id[row["id"]] = i
    return config, index, by_id, np.stack(inputs), np.stack(outputs)


def paired_diagnostics(atlas, outputs, index, by_id, pairs, split):
    selected = [p for p in pairs if p["split"] == split and p["left"] in by_id and p["right"] in by_id]
    result = {"pair_count": len(selected), "factor_changes": {}}
    if not selected:
        return result
    left = np.array([by_id[p["left"]] for p in selected])
    right = np.array([by_id[p["right"]] for p in selected])
    changes = outputs[right] - outputs[left]
    for mode in ("local", "factor", "rank_matched", "global", "shuffled"):
        frames, _, _ = atlas.frames(outputs[left], mode=mode, seed=19)
        residual = np.array([np.square(delta - q @ (q.T @ delta)).sum() for delta, q in zip(changes, frames)])
        energy = np.square(changes).sum(1)
        result[mode] = {"relative_difference_residual": float(residual.sum() / max(energy.sum(), 1e-12)),
                        "mean_rank": float(np.mean([q.shape[1] for q in frames]))}
    codes, active, _ = atlas.assign(outputs)
    for factor in sorted({p["factor"] for p in selected}):
        ix = np.array([i for i, p in enumerate(selected) if p["factor"] == factor])
        delta = codes[:, :, right[ix]] - codes[:, :, left[ix]]
        energies = np.square(delta).sum((1, 2))
        probability = energies / max(energies.sum(), 1e-12)
        result["factor_changes"][factor] = {
            "code_energy_by_block": energies.tolist(),
            "largest_block_share": float(probability.max()),
            "effective_participating_blocks": float(1 / max(np.square(probability).sum(), 1e-12)),
            "activity_change_rate": float(np.mean(active[left[ix]] != active[right[ix]]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--charts", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    capture, index, by_id, _, y = load_capture(args.capture)
    manifest = load_manifest(args.manifest)
    if capture["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("capture/manifest mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"capture_sha256": digest(capture), "manifest_sha256": manifest["manifest_sha256"],
              "groups": args.groups, "width": args.width, "active": 2,
              "charts": args.charts, "rank": args.rank, "seed": args.seed,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    from sparsifier.prune import factor_atlas, geometry_features
    config["estimator_sources"] = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (factor_atlas, geometry_features)}
    write_immutable(args.output / "config.json", config)
    train = np.array([r["split"] == "geometry_fit" for r in index["rows"]])
    atlas = FactorAtlas(args.groups, args.width, 2, args.charts, args.rank, args.seed).fit(y[train])
    atlas.save(args.output / "atlas.npz")
    report = {"config": config, "fit": atlas.diagnostics(y[train]), "splits": {},
              "claim": "development_diagnostics_only_no_semantic_factor_recovery_asserted"}
    for split in ("geometry_fit", "pruning_calibration", "validation", "stage2_holdout"):
        selected = np.array([r["split"] == split for r in index["rows"]])
        if not selected.any():
            continue
        report["splits"][split] = {
            "reconstruction": atlas.diagnostics(y[selected]),
            "paired": paired_diagnostics(atlas, y, index, by_id, manifest["pairs"], split)}
        atomic_json(args.output / "report.json", report)
    report["complete"] = True
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
