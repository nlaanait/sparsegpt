"""Compare actual common-128 inputs and gold answers across corrected evaluations."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


FIELDS = ("id", "prompt_sha256", "input_tokens", "batch_ids", "batch_size",
          "batching", "expected_answer")


def load(path):
    return json.loads(path.read_text())


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compare_rows(reference, candidate):
    assert len(reference) == len(candidate) == 128
    for left, right in zip(reference, candidate):
        assert all(left[key] == right[key] for key in FIELDS)


def compare_raw(reference, candidate):
    assert reference["batch_ids"] == candidate["batch_ids"]
    assert reference["input_ids_sha256"] == candidate["input_ids_sha256"]


def read_verified(root, label):
    audit_path = root / "audits" / f"{label}.json"
    input_path = root / "audits" / f"{label}-generation-inputs.json"
    audit, inputs = load(audit_path), load(input_path)
    folder = Path(audit["evaluation"])
    config = load(folder / "config.json")
    assert audit["common128_complete"] and inputs["common128_complete"]
    assert audit["raw_token_rescore_mismatches"] == inputs["packed_input_and_prompt_mismatches"] == 0
    assert audit["config_sha256"] == inputs["config_sha256"] == digest(config)
    assert audit["report_sha256"] == checksum(folder / "report.json")
    body = (folder / "answers.jsonl").read_bytes()
    assert body.endswith(b"\n")
    assert sha256(body).hexdigest() == audit["answers_sha256"] == inputs["answers_snapshot_sha256"]
    rows = [json.loads(line) for line in body.decode().splitlines()]
    assert digest(rows) == audit["audited_records_sha256"] == inputs["audited_records_sha256"]
    assert len(rows) == audit["count"] == inputs["count"]
    assert digest([r["id"] for r in rows[:128]]) == audit["common128"]["ids_sha256"]
    raw = {}
    for ids in [r["batch_ids"] for r in rows[:128]]:
        filename = digest(ids) + ".json"
        if filename not in raw:
            path = folder / "raw-generations" / filename
            assert checksum(path) == audit["raw_files"][filename] == inputs["raw_files_sha256"][filename]
            raw[filename] = load(path)
    assert len(raw) == 4
    evidence = {"label": label, "evaluation": str(folder), "count_compared": 128,
                "config_sha256": audit["config_sha256"], "answers_sha256": audit["answers_sha256"],
                "raw_audit_sha256": checksum(audit_path),
                "generation_inputs_audit_sha256": checksum(input_path)}
    return rows[:128], raw, config, evidence


def main():
    repo = Path(__file__).resolve().parents[4]
    root = repo / "results/tangent/gsm8k/eos-corrected-v1"
    collector = load(root / "collector/progress.json")
    reference, reference_raw, reference_config, _ = read_verified(root, "dense")
    evaluations = []
    for item in collector["verified"]:
        assert item["state"] == "verified"
        rows, raw, config, evidence = read_verified(root, item["label"])
        compare_rows(reference, rows)
        assert raw.keys() == reference_raw.keys()
        for filename in raw:
            compare_raw(reference_raw[filename], raw[filename])
        for key in ("model", "revision", "max_new_tokens", "batching"):
            assert config[key] == reference_config[key]
        evaluations.append(evidence)

    detected = []
    for key in ("prompt_sha256", "expected_answer"):
        corrupted = deepcopy(reference)
        corrupted[0][key] = "corrupted:" + str(reference[0][key])
        try:
            compare_rows(reference, corrupted)
        except AssertionError:
            detected.append(key)
        else:
            raise AssertionError(f"{key} corruption was not detected")
    first = next(iter(reference_raw.values()))
    try:
        compare_raw(first, dict(first, input_ids_sha256="corrupted"))
    except AssertionError:
        detected.append("input_ids_sha256")
    else:
        raise AssertionError("packed-input corruption was not detected")

    result = {"scope": "actual_common128_prompt_packed_ids_and_gold_parity_not_attention_mask_byte_audit",
              "observed_at": datetime.now(timezone.utc).isoformat(),
              "verified_evaluations": len(evaluations),
              "expected_evaluations": collector["expected_evaluations"],
              "all_expected_evaluations_present": len(evaluations) == collector["expected_evaluations"],
              "common128_ids_sha256": digest([r["id"] for r in reference]),
              "identical_recorded_fields": list(FIELDS),
              "common_batch_packed_input_hashes": {k: v["input_ids_sha256"] for k, v in reference_raw.items()},
              "evaluations": evaluations, "in_memory_corruptions_detected": detected,
              "uses_GPU": False, "auditor_sha256": checksum(Path(__file__))}
    output = root / "audits/common128-input-parity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}))


if __name__ == "__main__":
    main()
