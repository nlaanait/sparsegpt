"""Click command flags and the quantization stub chain."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch
from click.testing import CliRunner

from sparsifier.data.cache import find_pruned_model, prune_config_from_args, write_prune_config
from sparsifier.cli import main
from sparsifier.prune.quantize import apply_quantization, quantization_chain


def test_help_uses_group_names_and_not_the_old_flags():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--zeros-per-group" in result.output
    assert "--group-size" in result.output
    assert "--wbits" in result.output
    assert "prunen" not in result.output
    assert "prunem" not in result.output


def test_sixteen_bits_is_an_empty_chain():
    weights = torch.ones(2, 2)
    assert quantization_chain(16) == ()
    assert torch.equal(apply_quantization(weights, 16), weights)


def test_fewer_than_sixteen_bits_is_not_implemented():
    with pytest.raises(NotImplementedError):
        apply_quantization(torch.ones(2, 2), 4)


def test_cli_refuses_quantization_before_loading_a_model():
    result = CliRunner().invoke(
        main,
        ["Qwen/Qwen3.5-2B", "c4", "--wbits", "4"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)


def test_opt_stays_off_this_command():
    result = CliRunner().invoke(main, ["facebook/opt-125m", "c4", "--skip-eval"])
    assert result.exit_code != 0
    assert "not a measurement family" in result.output


def test_group_fields_are_the_checkpoint_keys(tmp_path):
    args = Namespace(
        model="Qwen/Qwen3.5-2B",
        dataset="c4",
        seed=0,
        nsamples=128,
        percdamp=0.01,
        sparsity=0.0,
        zeros_per_group=2,
        group_size=4,
        blocksize=128,
        gmp=False,
        wbits=16,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        true_sequential=False,
        geometry_alpha=0.0,
        geometry_probes=None,
    )
    config = prune_config_from_args(args)
    assert config["zeros_per_group"] == 2
    assert config["group_size"] == 4
    assert "prunen" not in config
    assert "prunem" not in config
    dest = tmp_path / "qwen"
    write_prune_config(dest, config)
    (dest / "model.safetensors").write_bytes(b"")
    assert find_pruned_model(config, cache_dir=tmp_path) == dest

    legacy_root = tmp_path / "old"
    legacy = legacy_root / "qwen-legacy"
    legacy.mkdir(parents=True)
    written = json.loads((dest / "prune_config.json").read_text())
    written["prunen"] = written.pop("zeros_per_group")
    written["prunem"] = written.pop("group_size")
    (legacy / "prune_config.json").write_text(json.dumps(written))
    (legacy / "model.safetensors").write_bytes(b"")
    assert find_pruned_model(config, cache_dir=legacy_root) == legacy
