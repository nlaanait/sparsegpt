"""Token-aligned math, logic, and code knobs for Variant A.

The default bank is the nine pairs chosen for Qwen 3.5-0.8B. Each pair
shares a token length under ``add_special_tokens=False`` and differs at
one index. That index is the only position subtracted into ``H_Δ``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConceptPair:
    knob: str
    domain: str
    left: str
    right: str


DEFAULT_CONCEPT_PAIRS: tuple[ConceptPair, ...] = (
    ConceptPair("ones_digit", "math", "the value is 17.", "the value is 18."),
    ConceptPair("tens_digit", "math", "the value is 17.", "the value is 27."),
    ConceptPair("add_sub", "math", "x = 2 + 3", "x = 2 - 3"),
    ConceptPair("mul_div", "math", "x = 2 * 3", "x = 2 / 3"),
    ConceptPair("and_or", "logic", "p and q", "p or q"),
    ConceptPair("quantifier", "logic", "all x are even", "some x are even"),
    ConceptPair("polarity", "logic", "return True", "return False"),
    ConceptPair("equality", "code", "x == y", "x != y"),
    ConceptPair("increment", "code", "return a + b", "return a - b"),
)


def knob_ids(pairs: tuple[ConceptPair, ...] | list[ConceptPair]) -> torch.Tensor:
    """Stable integer id per pair, in bank order."""
    return torch.arange(len(pairs), dtype=torch.int64)


def encode_pair(tokenizer, pair: ConceptPair) -> tuple[list[int], list[int]]:
    """Encode both sides with special tokens off so the edit stays aligned."""
    left = tokenizer.encode(pair.left, add_special_tokens=False)
    right = tokenizer.encode(pair.right, add_special_tokens=False)
    return left, right


@dataclass(frozen=True)
class PresenceReference:
    """Concept-present strings measured against one explicit absent reference."""

    knob: str
    present: tuple[str, ...]
    absent: str
    positions: tuple[int, ...]


# h_0 is the absent string. Ones and tens share the numeral reference, so
# their presence vectors match. Knobs with two present strings use the mean.
PRESENCE_REFERENCES: tuple[PresenceReference, ...] = (
    PresenceReference("ones_digit", ("the value is 17.",), "the value is 00.", (4, 5)),
    PresenceReference("tens_digit", ("the value is 17.",), "the value is 00.", (4, 5)),
    PresenceReference("add_sub", ("x = 2 + 3", "x = 2 - 3"), "x = 2 @ 3", (4,)),
    PresenceReference("mul_div", ("x = 2 * 3", "x = 2 / 3"), "x = 2 @ 3", (4,)),
    PresenceReference("and_or", ("p and q", "p or q"), "p @ q", (1,)),
    PresenceReference("quantifier", ("all x are even", "some x are even"), "the x are even", (0,)),
    PresenceReference("polarity", ("return True", "return False"), "return None", (1,)),
    PresenceReference("equality", ("x == y", "x != y"), "x @ y", (1,)),
    PresenceReference("increment", ("return a + b", "return a - b"), "return a @ b", (2,)),
)


def edit_index(tokenizer, pair: ConceptPair) -> int:
    """Index of the single token that differs between the two sides."""
    left, right = encode_pair(tokenizer, pair)
    if len(left) != len(right):
        raise ValueError(f"{pair.knob} changes token length ({len(left)} vs {len(right)})")
    diffs = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(diffs) != 1:
        raise ValueError(f"{pair.knob} differs at {len(diffs)} tokens, expected 1")
    return diffs[0]
