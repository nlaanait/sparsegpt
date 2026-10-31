"""Catalog, sweep plans, and figure rewrites. No model load and no GPU run."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from analysis.sweep import SPARSITIES, family_cmd
from experiments.gemma import GEMMA
from experiments.protocol import SweepSpec
from experiments.reproduce import (
    _done_from_disk,
    eval_command,
    main,
    plan_jobs,
    prune_command,
    write_reports,
)
from experiments.sweeps import sweeps
from sparsifier.catalog import models, resolve


def test_every_sweep_model_resolves_to_its_runner():
    seen = {}
    names = []
    for spec in sweeps():
        assert spec.name not in names
        names.append(spec.name)
        for model in spec.models:
            record = resolve(model)
            assert record.model_id == model
            assert record.runner == spec.runner
            assert record.stored_b is not None and record.stored_b > 0
            seen[model] = record
    assert set(seen) == {record.model_id for record in models()}
    gemma = seen["google/gemma-4-26B-A4B"]
    assert gemma.family == "gemma"
    assert gemma.loop == "own"
    assert seen["Qwen/Qwen3.5-27B"].loop == "decoder"
    assert seen["Qwen/Qwen3.6-27B"].stored_b == seen["Qwen/Qwen3.5-27B"].stored_b
    assert seen["Qwen/Qwen3.8-27B"].stored_b == seen["Qwen/Qwen3.5-27B"].stored_b
    assert seen["allenai/OLMo-2-0325-32B-Instruct"].family == "olmo"
    assert seen["meta-llama/Llama-3.2-90B-Vision"].family == "llama"
    assert seen["meta-llama/Llama-3.2-11B-Vision"].stored_b > seen["meta-llama/Llama-3.1-8B"].stored_b


def test_a_known_prefix_resolves_without_a_stored_count():
    qwen = resolve("Qwen/Qwen3.5-1B")
    assert qwen.family == "qwen"
    assert qwen.runner == "sparsegpt.qwen"
    assert qwen.stored_b is None
    opt = resolve("facebook/opt-125m")
    assert opt.runner == "sparsegpt.opt"
    assert opt.loop == "decoder"
    bloom = resolve("bigscience/bloom-560m")
    assert bloom.runner == "sparsegpt.bloom"
    tuned = resolve("google/gemma-4-E2B-it")
    assert tuned.family == "gemma"
    assert tuned.loop == "own"


def test_sparsifier_does_not_import_the_sweep_stack():
    root = Path(__file__).resolve().parents[1] / "sparsifier"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                assert not module.startswith(("experiments", "analysis", "drivers")), (
                    f"{path.name} imports {module}"
                )


def test_list_prints_every_sweep(capsys):
    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    for spec in sweeps():
        assert spec.name in printed
        assert spec.runner in printed


def test_unknown_sweep_is_rejected():
    with pytest.raises(SystemExit, match="unknown sweep"):
        main(["plan", "no-such-sweep"])


def test_eval_command_matches_the_sweep(monkeypatch):
    spec = GEMMA
    model = spec.models[0]
    monkeypatch.setattr("analysis.sweep.checkpoint", lambda *_args, **_kwargs: None)
    assert eval_command(spec, model, ["wikitext2"], 0.0, load_cached=False) == family_cmd(
        spec, model, ["wikitext2"], 0.0
    )
    assert eval_command(spec, model, ["ptb"], 0.5, load_cached=False) == family_cmd(
        spec, model, ["ptb"], 0.5
    )
    monkeypatch.setattr("analysis.sweep.checkpoint", lambda *_args, **_kwargs: Path("/tmp/cached"))
    assert eval_command(spec, model, ["ptb"], 0.5, load_cached=True) == family_cmd(
        spec, model, ["ptb"], 0.5
    )


def test_plan_prunes_once_then_loads_the_cache():
    spec = _toy_spec(Path("/tmp/unused"))
    jobs = plan_jobs(spec, done=lambda *_args: False, cached=lambda *_args: False)
    dense = [job for job in jobs if job.kind == "dense"]
    assert [job.datasets for job in dense] == [("wikitext2",), ("ptb",)]
    assert all("--skip-eval" not in job.command for job in dense)
    first = next(job for job in jobs if job.sparsity == 0.1)
    assert first.kind == "prune"
    assert first.command == tuple(prune_command(spec, spec.models[0], 0.1))
    assert "--skip-eval" in first.command
    assert "--eval-datasets" not in first.command
    evals = [job for job in jobs if job.kind == "eval" and job.sparsity == 0.1]
    assert [job.datasets for job in evals] == [("wikitext2",), ("ptb",)]
    assert all("--load-cached" in job.command and "--true-sequential" in job.command for job in evals)
    assert len([job for job in jobs if job.kind == "prune"]) == len(SPARSITIES)


def test_plan_skips_a_cached_sparsity():
    spec = _toy_spec(Path("/tmp/unused"))
    jobs = plan_jobs(
        spec,
        done=lambda *_args: False,
        cached=lambda _model, sparsity: sparsity == 0.2,
    )
    prunes = [job.sparsity for job in jobs if job.kind == "prune"]
    assert 0.2 not in prunes
    cached_eval = next(job for job in jobs if job.kind == "eval" and job.sparsity == 0.2)
    assert "--load-cached" in cached_eval.command


def test_likelihood_dump_counts_as_measured(tmp_path: Path):
    spec = _toy_spec(tmp_path)
    directory = tmp_path / "likelihood"
    directory.mkdir()
    (directory / "toy-1b_wikitext2.json").write_text(
        json.dumps({"model": spec.models[0], "0.4": {"perplexity": 21.0}})
    )
    done = _done_from_disk(spec)
    assert done(spec.models[0], "wikitext2", "sparsegpt", 0.4)
    assert not done(spec.models[0], "wikitext2", "dense", 0.0)
    assert not done(spec.models[0], "wikitext2", "dense", 0.4)
    jobs = plan_jobs(spec, done=done, cached=lambda *_args: True)
    assert not any(job.sparsity == 0.4 and job.datasets == ("wikitext2",) for job in jobs)
    assert any(job.sparsity == 0.4 and job.datasets == ("ptb",) for job in jobs)


def test_saved_gemma_scores_are_not_planned_again():
    done = _done_from_disk(GEMMA)
    assert done("google/gemma-4-12B", "ptb", "dense", 0.0)
    assert done("google/gemma-4-12B", "wikitext2", "sparsegpt", 0.1)
    jobs = plan_jobs(GEMMA, done=done, cached=lambda *_args: True)
    assert not any(
        job.model == "google/gemma-4-12B" and job.sparsity == 0.0 and "ptb" in job.datasets
        for job in jobs
    )
    assert not any(
        job.model == "google/gemma-4-12B" and job.sparsity == 0.1 and "wikitext2" in job.datasets
        for job in jobs
    )


def test_figures_rewrite_from_saved_json(tmp_path: Path):
    spec = _toy_spec(tmp_path, models=("org/Toy-1B", "org/Toy-3B"))
    sizes = {"org/Toy-1B": 1.0, "org/Toy-3B": 3.0}
    for model in spec.models:
        dense = 10.0 + sizes[model]
        points = []
        for dataset in spec.eval_datasets:
            points.append(_point(model, dataset, 0.0, dense))
            # Larger stored N loses less. The shared law needs that size split.
            excess = 0.9 if sizes[model] == 1.0 else 0.25
            for sparsity in SPARSITIES:
                points.append(_point(model, dataset, sparsity, dense * (1.0 + excess * sparsity)))
        path = spec.result_dir / f"{model.split('/')[-1].lower()}_sparsity.json"
        path.write_text(json.dumps({"points": points}))
    written = write_reports(spec, sizes)
    assert set(written["law"]) == {"wikitext2", "ptb"}
    assert written["law"]["wikitext2"]["law"].startswith("ln(perplexity / dense)")
    assert written["principle"] == tmp_path / "sparsity_principle.html"
    for suffix in ("", "_ptb"):
        for page in ("perplexity", "loglog", "normalized"):
            html = tmp_path / f"toy_sparsity_{page}{suffix}.html"
            assert html.is_file()
            assert "Toy-1B" in html.read_text()
    assert written["law"]["wikitext2"]["a"] > 0


def _toy_spec(result_dir: Path, models: tuple[str, ...] = ("org/Toy-1B",)) -> SweepSpec:
    return SweepSpec(
        name="toy",
        models=models,
        runner="sparsegpt.llama",
        extra_args=("--true-sequential",),
        result_dir=result_dir,
        log_dir=result_dir / "logs",
        log_infix="sdpa",
        attention="sdpa",
        holdout_suffix="1B",
        calibration="c4",
        eval_datasets=("wikitext2", "ptb"),
    )


def _point(model: str, dataset: str, sparsity: float, perplexity: float) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "method": "dense" if sparsity == 0.0 else "sparsegpt",
        "sparsity": sparsity,
        "perplexity": perplexity,
    }
