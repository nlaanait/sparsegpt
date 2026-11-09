"""Immutable GSM8K experiment inputs and strict, exact answer parsing.

Official test data is never opened during preparation of the development pilot.
"""
from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?"


def exact_number(text: str) -> Fraction | None:
    text = text.strip()
    if not re.fullmatch(NUMBER, text):
        return None
    try:
        return Fraction(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def parse_answer(text: str, *, truncated: bool = False) -> Fraction | None:
    """Require an explicit final-answer marker; never guess the last numeral.

    Ignore text inside Qwen's reasoning block. Multiple conflicting markers,
    malformed numbers, and budget-truncated output fail rather than get credit.
    """
    if truncated:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    candidates = re.findall(r"(?:Final answer\s*:|####)\s*([^\n]+)", text, re.I)
    if not candidates:
        return None
    values = []
    for candidate in candidates:
        # Permit a currency prefix and a sentence-ending period, but no units
        # or explanatory arithmetic after the final value.
        candidate = candidate.strip().removeprefix("$").strip()
        if candidate.endswith("."):
            candidate = candidate[:-1]
        value = exact_number(candidate)
        if value is None:
            return None
        values.append(value)
    return values[0] if len(set(values)) == 1 else None


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_immutable(path: Path, value) -> None:
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    with path.open("x") as handle:
        handle.write(content)


def prepare_pilot(source: Path, output: Path, count: int = 200) -> dict:
    """Reserve a training-only prefix for development, hashing every source row."""
    from datasets import Dataset

    if source.name != "gsm8k-train.arrow":
        raise ValueError("pilot source must be the GSM8K training Arrow file")
    dataset = Dataset.from_file(str(source))
    if not 1 <= count <= len(dataset):
        raise ValueError("invalid pilot count")
    rows = []
    for index in range(count):
        item = dataset[index]
        answer = parse_answer(item["answer"])
        if answer is None:
            raise ValueError(f"invalid source answer at {index}")
        rows.append({"id": f"gsm8k/train/{index}", "source_index": index,
                     "family": f"gsm8k/train/{index}", "split": "development",
                     "question": item["question"], "answer": str(answer),
                     "source_sha256": digest(item)})
    manifest = {"schema": 1, "dataset": "openai/gsm8k", "source_split": "train",
                "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "purpose": "development_only_not_confirmatory", "rows": rows}
    manifest["manifest_sha256"] = digest(manifest)
    write_immutable(output, manifest)
    return manifest


def load_manifest(path: Path) -> dict:
    result = json.loads(path.read_text())
    expected = result.pop("manifest_sha256")
    if digest(result) != expected:
        raise ValueError("dataset manifest hash mismatch")
    result["manifest_sha256"] = expected
    ids = [row["id"] for row in result["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example IDs")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    manifest = prepare_pilot(args.train_arrow, args.output, args.count)
    print(f"Reserved {len(manifest['rows'])} training problems: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
