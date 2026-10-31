"""CPU checks for metrics and the two pruners. No model load."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments.protocol import PPL_RE
from sparsifier.catalog import resolve
from sparsifier.metrics import BitsPerByte, Likelihood, Perplexity, format_result, get_metric
from sparsifier.pruners import get_pruner
from sparsifier.run import RunConfig


def test_perplexity_matches_the_eval_reduction():
    likelihood = Likelihood(nats=math.log(20.0) * 8, tokens=8)
    score = Perplexity().score(likelihood)
    assert score.name == "perplexity"
    assert score.value == pytest.approx(20.0)


def test_perplexity_log_line_matches_the_sweep_parser():
    score = Perplexity().score(Likelihood(nats=math.log(12.5) * 4, tokens=4))
    line = format_result(score)
    match = PPL_RE.search(line)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(12.5)


def test_bits_per_byte_uses_bytes_not_tokens():
    likelihood = Likelihood(nats=math.log(2) * 16, tokens=100, bytes=8)
    score = BitsPerByte().score(likelihood)
    assert score.name == "bits_per_byte"
    assert score.value == pytest.approx(2.0)
    assert format_result(score) == "bits_per_byte: 2.000000"


def test_bits_per_byte_refuses_a_token_only_likelihood():
    with pytest.raises(ValueError, match="bytes"):
        BitsPerByte().score(Likelihood(nats=1.0, tokens=4))


def test_unknown_metric():
    with pytest.raises(KeyError, match="bits_per_byte"):
        get_metric("cross_entropy")


def test_run_config_rejects_an_unknown_corpus():
    with pytest.raises(ValueError, match="calibration"):
        RunConfig(model="Qwen/Qwen3.5-2B", calibration="wiki")


def test_geometry_options_are_part_of_the_config():
    probes = Path("/tmp/probes")
    config = RunConfig(
        model="Qwen/Qwen3.5-2B",
        calibration="c4",
        pruner="geometry",
        geometry_alpha=0.5,
        geometry_probes=probes,
    )
    assert config.pruner_options == {"alpha": 0.5, "probes": probes}


def test_unknown_model_id_is_rejected():
    with pytest.raises(LookupError, match="none"):
        resolve("openai/gpt-2")


def _observe(bound, layer: nn.Linear) -> None:
    inp = torch.randn(8, layer.in_features)
    bound.add_batch(inp, layer(inp))


def test_sparsegpt_pruner_runs_on_cpu():
    layer = nn.Linear(4, 3, bias=False)
    bound = get_pruner("sparsegpt").bind(layer, block_index=0, module_name="mlp.down_proj")
    _observe(bound, layer)
    bound.finalize()
    bound.prune(sparsity=0.5, blocksize=4)
    zeros = int((layer.weight.data == 0).sum())
    assert zeros > 0
    bound.free()


def test_geometry_pruner_adds_the_tangent_term(tmp_path: Path):
    layer = nn.Linear(4, 3, bias=False)
    probes = torch.randn(4, 6).numpy()
    path = tmp_path / "0"
    path.mkdir()
    np.save(path / "mlp.down_proj.npy", probes)

    plain = get_pruner("sparsegpt").bind(layer, block_index=0, module_name="mlp.down_proj")
    _observe(plain, layer)
    plain.finalize()
    hessian = plain.solver.H.detach().clone()

    geometry = get_pruner("geometry", alpha=0.5, probes=tmp_path).bind(
        layer, block_index=0, module_name="mlp.down_proj"
    )
    _observe(geometry, layer)
    before = geometry.solver.H.detach().clone()
    geometry.finalize()
    assert not torch.allclose(geometry.solver.H, before)
    assert not torch.allclose(geometry.solver.H, hessian)
    geometry.prune(sparsity=0.25, blocksize=4)
    geometry.free()
    plain.free()


def test_geometry_without_probes_is_refused():
    with pytest.raises(ValueError, match="probes"):
        get_pruner("geometry", alpha=0.5)
