"""CPU-only verification of the local-65 checkpoint saved at the intentional stop."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    folder = Path(__file__).resolve().parent
    queue = json.loads((folder / "queue-state.json").read_text())
    repo = Path(queue["working_directory"])
    source = Path(queue["partial_source"]["path"])
    config = json.loads((folder / "local-s65-config.json").read_text())
    report = json.loads((folder / "local-s65-report.json").read_text())

    # Configure before importing any driver, just as the recorded GPU command does.
    from sparsifier.eval.gsm8k_batch_runtime import configure_batch_size
    configure_batch_size(config["batching"]["batch_size"])
    from sparsifier.eval.gsm8k_batching import policy
    from sparsifier.data.gsm8k import digest
    from sparsifier.eval.gsm8k_sequential import sites_for, verify_stage
    from sparsifier.prune.whole_neuron_io import file_hash

    torch.set_num_threads(4)
    assert policy() == config["batching"], "resume requires the saved execution policy"
    config_hash = digest(config)
    assert digest(json.loads((source / "config.json").read_text())) == config_hash
    assert report["config_sha256"] == config_hash and not report["complete"]
    expected = sites_for(list(range(32)), ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"], 32)
    assert len(report["stages"]) == 25
    assert [tuple(s["site"]) for s in report["stages"]] == expected[:25]
    current = json.loads((source / "report.json").read_text())
    assert current["stages"][:25] == report["stages"]
    source_files = {}
    for name, expected_hash in config["sources"].items():
        matches = list((repo / "src/sparsifier").rglob(name))
        assert len(matches) == 1
        actual_hash = file_hash(matches[0])
        assert actual_hash == expected_hash, f"resume source changed: {name}"
        source_files[str(matches[0].relative_to(repo))] = actual_hash

    upstream, verified = config_hash, []
    for index, stage in enumerate(report["stages"]):
        path = source / f"site-{index:03d}.pt"
        checksum = file_hash(path)
        assert checksum == stage["candidate_sha256"]
        stored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        verify_stage(stored, config_hash, upstream, tuple(stage["site"]))
        assert stored["removed"] == stage["removed"]
        weight = stored["weight"]
        fraction = float((weight == 0).double().mean())
        assert abs(fraction - config["sparsity"]) < 1e-4
        verified.append({"site": stage["site"], "file": path.name, "sha256": checksum,
                         "upstream": upstream, "finite": True,
                         "shape": list(weight.shape), "zero_fraction": fraction})
        upstream = checksum
        del stored, weight

    result = {"scope": "saved_25_stage_prefix_not_completed_model_or_accuracy_measurement",
              "observed_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "source_config_sha256": config_hash,
              "report_snapshot_sha256": file_hash(folder / "local-s65-report.json"),
              "verified_stages": len(verified), "target_stages": 96,
              "next_stage_index": 25, "next_site": list(expected[25]),
              "execution_policy_matches": True, "source_files_match": source_files,
              "uses_GPU": False, "verifier_sha256": file_hash(Path(__file__)),
              "stages": verified}
    (folder / "local-s65-prefix-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("stages", "source_files_match")}))


if __name__ == "__main__":
    main()
