"""Default math / logic / code knob bank for Variant A.

The nine strings are the agreed set. Each pair is one token apart under the
Qwen 3.5 tokenizer with special tokens left off.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sparsifier.data.concept_pairs import (
    DEFAULT_CONCEPT_PAIRS,
    edit_index,
    knob_ids,
)
from sparsifier.data.pair_deltas import capture_edit_deltas, save_pair_deltas
from sparsifier.prune.pruners import get_pruner


def test_default_bank_is_the_nine_agreed_strings():
    assert [(p.knob, p.domain, p.left, p.right) for p in DEFAULT_CONCEPT_PAIRS] == [
        ("ones_digit", "math", "the value is 17.", "the value is 18."),
        ("tens_digit", "math", "the value is 17.", "the value is 27."),
        ("add_sub", "math", "x = 2 + 3", "x = 2 - 3"),
        ("mul_div", "math", "x = 2 * 3", "x = 2 / 3"),
        ("and_or", "logic", "p and q", "p or q"),
        ("quantifier", "logic", "all x are even", "some x are even"),
        ("polarity", "logic", "return True", "return False"),
        ("equality", "code", "x == y", "x != y"),
        ("increment", "code", "return a + b", "return a - b"),
    ]


def test_knob_ids_are_stable_and_one_per_pair():
    ids = knob_ids(DEFAULT_CONCEPT_PAIRS)
    assert ids.tolist() == list(range(9))
    assert len(set(p.knob for p in DEFAULT_CONCEPT_PAIRS)) == 9


def test_each_pair_differs_at_one_qwen_token():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    expected = {
        "ones_digit": 5,
        "tens_digit": 4,
        "add_sub": 4,
        "mul_div": 4,
        "and_or": 1,
        "quantifier": 0,
        "polarity": 1,
        "equality": 1,
        "increment": 2,
    }
    for pair in DEFAULT_CONCEPT_PAIRS:
        assert edit_index(tokenizer, pair) == expected[pair.knob]


def test_presence_references_follow_the_nine_knobs():
    from sparsifier.data.concept_pairs import PRESENCE_REFERENCES

    assert [item.knob for item in PRESENCE_REFERENCES] == [pair.knob for pair in DEFAULT_CONCEPT_PAIRS]
    number = PRESENCE_REFERENCES[0]
    assert number.absent == "the value is 00."
    assert number.positions == (4, 5)
    assert PRESENCE_REFERENCES[1].absent == number.absent
    assert PRESENCE_REFERENCES[2].absent == "x = 2 @ 3"
    assert PRESENCE_REFERENCES[2].present == ("x = 2 + 3", "x = 2 - 3")


def test_position_mean_averages_the_requested_tokens():
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    embed = nn.Embedding(6, 4)

    def run(token_ids: torch.Tensor) -> None:
        linear(embed(token_ids))

    from sparsifier.data.pair_deltas import capture_position_mean

    ids = torch.tensor([[1, 2, 3]])
    got = capture_position_mean({"block.0.lin": linear}, run, ids, (0, 2))
    expected = 0.5 * (embed.weight[1] + embed.weight[3])
    torch.testing.assert_close(got["block.0.lin"], expected)


def test_capture_subtracts_only_the_edited_position():
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    embed = nn.Embedding(6, 4)

    def run(token_ids: torch.Tensor) -> None:
        linear(embed(token_ids))

    left = torch.tensor([[1, 2, 3]])
    right = torch.tensor([[1, 5, 3]])
    deltas = capture_edit_deltas({"block.0.lin": linear}, run, left, right, edit_index=1)
    expected = embed.weight[2] - embed.weight[5]
    torch.testing.assert_close(deltas["block.0.lin"], expected)


def test_saved_knob_ids_reach_the_input_side_hessian(tmp_path):
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    loud = torch.tensor([[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    quiet = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    deltas = {"0/up": torch.cat((loud, quiet), dim=0)}
    knobs = {"0/up": torch.tensor([0, 0, 1])}
    save_pair_deltas(tmp_path, deltas, knobs)

    x = torch.eye(4).unsqueeze(0)
    plain = nn.Linear(4, 2, bias=False)
    plain.load_state_dict(layer.state_dict())
    bound_plain = get_pruner("tangent_input_side", alpha_in=1.0, pairs=tmp_path).bind(
        plain, block_index=0, module_name="up"
    )
    # A sibling directory without knob ids, same deltas.
    bare = tmp_path / "bare"
    save_pair_deltas(bare, deltas, knob_ids=None)
    other = nn.Linear(4, 2, bias=False)
    other.load_state_dict(layer.state_dict())
    bound_bare = get_pruner("tangent_input_side", alpha_in=1.0, pairs=bare).bind(
        other, block_index=0, module_name="up"
    )
    bound_plain.add_batch(x, plain(x))
    bound_bare.add_batch(x, other(x))
    bound_plain.finalize()
    bound_bare.finalize()
    assert not torch.allclose(bound_plain.solver.H, bound_bare.solver.H)
    stored = np.load(tmp_path / "0" / "up.knobs.npy")
    assert stored.tolist() == [0, 0, 1]
