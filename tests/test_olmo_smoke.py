"""Smoke-range and SweepSpec checks for the OLMo 2 driver (no GPU)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from analysis.sweep import load_results, lookup, result_path, run_smoke_dense
from experiments.olmo import MODELS, OLMO2, SMOKE_DATASET, SMOKE_PPL_RANGE, stored_params


def test_olmo2_spec_protocol():
    assert OLMO2.calibration == "c4"
    assert OLMO2.eval_datasets == ("wikitext2", "ptb")
    assert OLMO2.runner == "sparsegpt.olmo"
    assert SMOKE_DATASET == "wikitext2"
    assert len(MODELS) == 4
    assert all(name.endswith("-Instruct") for name in MODELS)
    low, high = SMOKE_PPL_RANGE
    assert 0 < low < high


def test_stored_params_cover_models():
    sizes = stored_params()
    assert set(sizes) == set(MODELS)
    assert sizes[MODELS[0]] < sizes[MODELS[-1]]


def _spec_at(tmp_path):
    return replace(OLMO2, result_dir=tmp_path, log_dir=tmp_path / "logs", plot=False)


def test_smoke_rejects_out_of_range(tmp_path):
    spec = _spec_at(tmp_path)
    model = MODELS[0]
    path = result_path(spec, model)
    results = load_results(path, spec)
    results["points"].append(
        {
            "model": model,
            "dataset": "wikitext2",
            "method": "dense",
            "sparsity": 0.0,
            "perplexity": 12000.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))
    with pytest.raises(SystemExit, match="smoke failed"):
        run_smoke_dense(spec, [model], dataset="wikitext2", ppl_range=SMOKE_PPL_RANGE)


def test_smoke_accepts_in_range(tmp_path):
    spec = _spec_at(tmp_path)
    model = MODELS[0]
    path = result_path(spec, model)
    results = load_results(path, spec)
    results["points"].append(
        {
            "model": model,
            "dataset": "wikitext2",
            "method": "dense",
            "sparsity": 0.0,
            "perplexity": 12.5,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))
    cleared = run_smoke_dense(spec, [model], dataset="wikitext2", ppl_range=SMOKE_PPL_RANGE)
    assert cleared[model] == 12.5
    assert lookup(load_results(path, spec), "wikitext2", "dense", 0.0) == 12.5
