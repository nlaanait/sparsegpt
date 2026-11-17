"""Argv and resume rules for the Qwen 3.5-0.8B method sweep."""

from __future__ import annotations

import copy
import json
import sys

import pytest

from drivers.qwen08_method_sweep import (
    METHODS,
    SPARSITIES,
    pending_points,
    point_command,
)


def test_sparsity_ladder_starts_at_half_and_ends_at_99():
    assert SPARSITIES[0] == 0.5
    assert SPARSITIES[-1] == 0.99
    assert 0.9 in SPARSITIES


def test_sparsegpt_is_the_baseline_and_fisher_is_absent():
    assert METHODS[0] == "sparsegpt"
    assert "tangent_fisher" not in METHODS
    assert "tangent_input_side" in METHODS


def test_point_command_carries_the_method_flags():
    baseline = point_command("sparsegpt", 0.5, python="python")
    assert baseline[baseline.index("--sparsity") + 1] == "0.5"
    assert "--pruner" not in baseline
    assert baseline[baseline.index("--forward-batch") + 1] == "1"
    assert "drivers.capped_sparsifier" in baseline
    variant = point_command("tangent_input_side", 0.99, python="python")
    assert variant[variant.index("--sparsity") + 1] == "0.99"
    assert variant[variant.index("--pruner") + 1] == "tangent_input_side"
    assert "--alpha-in" in variant
    assert "--pair-deltas" in variant


def test_pending_points_skip_saved_perplexities():
    saved = {
        "points": [
            {"dataset": "wikitext2", "method": "sparsegpt", "sparsity": 0.5, "perplexity": 12.0}
        ]
    }
    missing = pending_points(saved)
    assert ("sparsegpt", 0.5) not in missing
    assert ("sparsegpt", 0.6) in missing
    assert missing[0] == ("sparsegpt", 0.6)


def test_legacy_results_cannot_resume_as_corrected_measurements(tmp_path,monkeypatch):
    from drivers import qwen08_method_sweep as sweep
    current = {'protocol':sweep.PROTOCOL,'sources':{'rescale.py':'curvature-trace-v2'}}
    monkeypatch.setattr(sweep,'current_provenance',lambda:current)
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps({'points':[{'method':'tangent_rescale','sparsity':.7,'perplexity':565.259254}]}))
    original = path.read_bytes()
    with pytest.raises(ValueError,match='historical points cannot be relabeled'):
        sweep.load_results(path)
    assert path.read_bytes()==original
    assert sweep.RESULT.name!='qwen35-0.8b-method-sweep.json'
    assert 'curvature-trace-v2' in sweep.RESULT.name


@pytest.mark.parametrize('changed',['sources','runtime','pair_files'])
def test_resume_requires_matching_sources_child_runtime_and_pair_bank(tmp_path,monkeypatch,changed):
    from drivers import qwen08_method_sweep as sweep
    current = {'protocol':sweep.PROTOCOL,'sources':{'rescale.py':'v2'},
               'runtime':{'torch':'original'},'pair_files':{'pairs.npy':'original'}}
    monkeypatch.setattr(sweep,'current_provenance',lambda:current)
    path = tmp_path/'current.json'
    result = sweep.load_results(path)
    sweep.save_results(path,result)
    assert sweep.load_results(path)==result
    current[changed] = {'changed':'different'}
    with pytest.raises(ValueError,match='provenance'):
        sweep.load_results(path)


@pytest.mark.parametrize('value',[float('inf'),float('nan'),0.])
def test_invalid_cached_metric_does_not_mark_a_point_done(value):
    saved = {'points':[{'dataset':'wikitext2','method':'sparsegpt','sparsity':.7,'perplexity':value}]}
    assert ('sparsegpt',.7) in pending_points(saved)


def test_source_change_during_point_does_not_write_a_mixed_result(tmp_path,monkeypatch):
    from drivers import qwen08_method_sweep as sweep
    current = {'protocol':sweep.PROTOCOL,'sources':{'solver':'before'}}
    monkeypatch.setattr(sweep,'current_provenance',lambda:copy.deepcopy(current))
    monkeypatch.setattr(sweep,'METHODS',('sparsegpt',))
    monkeypatch.setattr(sweep,'SPARSITIES',[.7])
    monkeypatch.setattr(sweep,'assert_room',lambda where:None)
    observed = []
    def run(method,sparsity,log_path):
        observed.append(log_path)
        current['sources']['solver'] = 'after'
        return 10.
    monkeypatch.setattr(sweep,'run_point',run)
    output = tmp_path/'current.json'
    monkeypatch.setattr(sys,'argv',['sweep','--output',str(output)])
    with pytest.raises(RuntimeError,match='result not saved'):
        sweep.main()
    assert not output.exists()
    assert observed[0].parent==tmp_path/'current.json.artifacts'/'logs'
