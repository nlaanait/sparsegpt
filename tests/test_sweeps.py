"""SweepSpec construction and validation. No model load."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparsifier.sweep import RunConfig, SweepSpec, repo_root, validate


def _qwen() -> SweepSpec:
    return SweepSpec(
        name="qwen",
        models=("Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
        calibration="c4",
        eval_datasets=("ptb", "wikitext2"),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="27B",
        plot=False,
    )


def test_validate_qwen_uses_the_catalog_runner(tmp_path: Path):
    checked = validate(_qwen(), tmp_path)
    assert checked.runner == "sparsegpt.qwen"
    assert checked.calibration == "c4"
    assert checked.eval_datasets == ("ptb", "wikitext2")
    assert checked.plot is False
    assert checked.result_dir == tmp_path.resolve() / "results" / "qwen"
    assert checked.log_dir == checked.result_dir / "logs"
    assert checked.record("Qwen/Qwen3.6-27B").stored_b == checked.record("Qwen/Qwen3.8-27B").stored_b


def test_prepare_creates_the_results_directories(tmp_path: Path):
    checked = validate(_qwen(), tmp_path)
    checked.prepare()
    assert checked.result_dir.is_dir()
    assert checked.log_dir.is_dir()


def test_repo_root_is_the_pixi_project():
    root = repo_root()
    assert (root / "pixi.toml").is_file()
    assert (root / "src" / "sparsifier" / "sweep" / "sweeps.py").is_file()


def test_a_model_outside_the_spec_is_rejected(tmp_path: Path):
    checked = validate(_qwen(), tmp_path)
    with pytest.raises(ValueError, match="not in sweep"):
        checked.record("Qwen/Qwen3.5-2B")


def test_a_prefix_without_a_stored_count_is_rejected(tmp_path: Path):
    spec = SweepSpec(
        name="qwen-preview",
        models=("Qwen/Qwen3.5-1B",),
        calibration="c4",
        eval_datasets=("wikitext2",),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="1B",
    )
    with pytest.raises(ValueError, match="no stored parameter count"):
        validate(spec, tmp_path)


def test_mixed_runners_are_rejected(tmp_path: Path):
    spec = SweepSpec(
        name="mixed",
        models=("Qwen/Qwen3.5-2B", "google/gemma-4-E2B"),
        calibration="c4",
        eval_datasets=("wikitext2",),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="2B",
    )
    with pytest.raises(ValueError, match="one runner"):
        validate(spec, tmp_path)


def test_an_unknown_model_is_rejected(tmp_path: Path):
    spec = SweepSpec(
        name="toy",
        models=("org/Toy-1B",),
        calibration="c4",
        eval_datasets=("wikitext2",),
        log_infix="sdpa",
        attention="sdpa",
        holdout_suffix="1B",
    )
    with pytest.raises(ValueError, match="Toy-1B"):
        validate(spec, tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "qwen/bad"}, "results directory"),
        ({"models": ()}, "non-empty tuple"),
        ({"models": ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B")}, "more than once"),
        ({"calibration": "pile"}, "unknown calibration"),
        ({"eval_datasets": ("pile",)}, "unknown eval"),
        ({"log_infix": ""}, "log_infix"),
        ({"plot": 1}, "plot must be a bool"),
    ],
)
def test_construction_rejects_a_bad_field(kwargs: dict, match: str):
    fields = dict(
        name="qwen",
        models=("Qwen/Qwen3.5-2B",),
        calibration="c4",
        eval_datasets=("wikitext2",),
        log_infix="pytorch",
        attention="atlas-gdn",
        holdout_suffix="2B",
    )
    fields.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SweepSpec(**fields)


def test_corpora_are_the_ones_run_config_accepts():
    spec = _qwen()
    RunConfig(model=spec.models[0], calibration=spec.calibration, eval_datasets=spec.eval_datasets)


def test_missing_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        validate(_qwen(), tmp_path / "absent")
