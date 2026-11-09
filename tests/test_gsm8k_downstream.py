import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import MODEL, REVISION
from sparsifier.eval.gsm8k_downstream import AnchorPrefix, PULLBACK, anchor_block, probe_rows, prune_site, readout_basis
from sparsifier.eval.gsm8k_downstream_sweep import PLAN, SMOKE, build_plan, smoke_gate, worker
from sparsifier.prune.downstream import DownstreamTangentCalibrator, TangentAnchor
from sparsifier.prune.sparsegpt import SparseGPT


def manifests():
    return ({"manifest_sha256": "cal", "rows": [{"source_index": 1}]},
            {"manifest_sha256": "val", "rows": [{"source_index": 1001}]})


def test_plan_adds_matching_ladder_with_smoke_and_original_validation_gates():
    calibration, validation = manifests()
    plan = build_plan(calibration, validation)
    assert len(plan["tasks"]) == 9
    assert plan["allowed_accuracy_drop"] is None
    assert plan["max_new_tokens"] == 2048 and plan["alpha"] == 1
    assert plan["pullback"] == PULLBACK
    assert plan["tasks"][0]["name"] == SMOKE
    assert plan["tasks"][0]["blocks"] == "15,16"
    prune = [task for task in plan["tasks"] if task["phase"] == "prune"]
    assert [task["eligible_sparsity"] for task in prune] == [.1, .3, .5, .7]
    assert all(task["blocks"] == "all" and task["depends_on"] == [SMOKE] for task in prune)
    evaluation = [task for task in plan["tasks"] if task["phase"] == "original_validation"]
    assert [task["depends_on"][0] for task in evaluation] == [task["name"] for task in prune]
    with pytest.raises(ValueError, match="leakage"):
        build_plan(calibration, calibration)


def test_probes_use_only_calibration_and_next_block_anchor_is_bounded():
    rows = [{"id": str(i), "family": str(i % 3), "split": split}
            for split in ("geometry_fit", "pruning_calibration", "validation", "final_test")
            for i in range(12)]
    chosen = probe_rows(rows, 6, 0)
    assert chosen == probe_rows(rows, 6, 0)
    assert all(row["split"] == "pruning_calibration" for row in chosen)
    assert {row["family"] for row in chosen} == {"0", "1", "2"}
    assert anchor_block(0, 3) == 1
    assert anchor_block(2, 3) == 2
    with pytest.raises(ValueError):
        anchor_block(3, 3)


def test_prefix_keeps_anchor_autograd_and_stops_before_later_layers():
    target, anchor, tail = nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([target, anchor, tail])
            self.tail_calls = 0

        def forward(self, input_ids, **kwargs):
            y = anchor(target(input_ids))
            self.tail_calls += 1
            return tail(y)

    model = Model().requires_grad_(False)
    prefix = AnchorPrefix(model, anchor)
    calibration = DownstreamTangentCalibrator(
        {"target": target}, {"anchor": TangentAnchor(anchor, torch.eye(3))}, num_probes=None,
    )
    x = torch.ones(1, 4, 3)
    calibration.run(lambda: prefix(input_ids=x))
    assert model.tail_calls == 0
    assert calibration.solvers["target"].geometry_energy.sum() > 0
    assert not anchor._forward_hooks and not target._forward_hooks


class KnownAtlas:
    """Known tangent for isolating integration from atlas estimation quality."""
    def __init__(self, *args):
        pass

    def fit(self, samples):
        assert samples.shape == (65, 4)

    def save(self, path):
        path.write_bytes(b"known-tangent-atlas")

    def frames(self, samples, mode="local"):
        return [np.eye(4)[:, :1]] * len(samples), None, None


def test_readout_basis_never_seeds_other_prompt_positions():
    frames = readout_basis(KnownAtlas(), torch.ones(1, 9, 4))
    assert frames.shape == (1, 9, 4, 1)
    assert not frames[:, :-1].any()
    assert frames[0, -1, 0, 0] == 1


def test_separate_ambient_budget_and_geometry_do_not_double_count_tokens():
    layer = nn.Linear(2, 2).requires_grad_(False)
    calibration = DownstreamTangentCalibrator(
        {"target": layer}, {"anchor": TangentAnchor(layer, torch.eye(2))}, num_probes=None,
    )
    solver = calibration.solvers["target"]
    solver.add_ambient_batch(torch.ones(10, 2))
    before = solver.H.clone()
    calibration.run(lambda: layer(torch.ones(1, 3, 2)), collect_ambient=False)
    assert solver.vector_count == 10 and solver.geometry_vector_count == 3
    torch.testing.assert_close(solver.H, before)
    torch.testing.assert_close(solver.input_energy, torch.full((2,), 10., dtype=torch.double))
    solver.finalize_tangents()
    assert solver.tangent_diagnostics["sample_count"] == 10
    assert solver.tangent_diagnostics["geometry_sample_count"] == 3


def test_real_site_pipeline_preserves_ambient_budget_and_alpha_zero_parity(tmp_path, monkeypatch):
    import sparsifier.eval.gsm8k_downstream as module
    from sparsifier.eval.gsm8k_sequential import collect

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.target = nn.Linear(4, 4, bias=False)
            self.anchor = nn.Tanh()

        def forward(self, input_ids, **kwargs):
            t = input_ids.float() / 128
            x = torch.stack((t, t.sin(), t.cos(), torch.ones_like(t)), -1)
            return self.anchor(self.target(x))

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def encode(self, *args, **kwargs):
            return list(range(1, 129))

    torch.manual_seed(7)
    model = Model().requires_grad_(False)
    rows = [{"id": str(i), "question": "q", "family": str(i % 3),
             "split": "geometry_fit" if i < 65 else "pruning_calibration"} for i in range(130)]
    c4_tokens = np.tile(np.arange(1, 129), (32, 1))
    monkeypatch.setattr(module, "FactorAtlas", KnownAtlas)
    original = model.target.weight.clone()
    candidate, diagnostics = prune_site(
        model, Tokenizer(), model.target, model.anchor, rows, c4_tokens, tmp_path, 0,
        alpha=0, num_probes=2, probe_count=3,
    )
    assert diagnostics["sample_count"] == 8192
    assert diagnostics["geometry_sample_count"] == 3 * 128
    assert len(diagnostics["probe_ids"]) == 3 and len(diagnostics["geometry_fit_ids"]) == 65
    assert diagnostics["geometry_trace"] > 0
    assert not model.target._forward_hooks and not model.anchor._forward_hooks
    torch.testing.assert_close(model.target.weight, original)
    _, _, inputs, _ = collect(model, Tokenizer(), model.target, rows, c4_tokens, 0)
    reference = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        reference.weight.copy_(original)
    ambient = SparseGPT(reference)
    ambient.add_batch(inputs)
    ambient.fasterprune(.5, mask_weight=torch.ones_like(original))
    torch.testing.assert_close(candidate, reference.weight)
    with pytest.raises(ValueError, match="held-out"):
        prune_site(model, Tokenizer(), model.target, model.anchor,
                   rows + [{"split": "validation"}], c4_tokens, tmp_path, 1)


def test_smoke_gate_requires_hashed_six_site_candidates_and_supported_geometry(tmp_path):
    plan = build_plan(*manifests())
    folder = tmp_path / SMOKE
    folder.mkdir()
    config = {"model": MODEL, "revision": REVISION, "arm": "downstream", "pullback": PULLBACK,
              "manifest_sha256": "cal", "alpha": 1., "seed": 0, "max_new_tokens": 2048}
    (folder / "config.json").write_text(json.dumps(config))
    upstream = digest(config)
    stages = []
    for index, site in enumerate((b, m) for b in (15, 16) for m in plan["modules"]):
        path = folder / f"site-{index:03d}.pt"
        torch.save({"config_sha256": digest(config), "upstream": upstream,
                    "site": list(site), "weight": torch.ones(2, 2)}, path)
        upstream = hashlib.sha256(path.read_bytes()).hexdigest()
        stages.append({"site": list(site), "candidate_sha256": upstream,
                       "diagnostics": {"geometry_trace": 1., "sample_count": 8192,
                                       "num_probes": 8, "probe_ids": list(range(32))}})
    report = {"config_sha256": digest(config), "complete": True, "stages": stages,
              "behavior": {"count": 4}, "whole_model_removed_fraction": .01}
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report))
    smoke_gate(tmp_path, plan)
    report["stages"][0]["diagnostics"]["geometry_trace"] = 0
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="calibration"):
        smoke_gate(tmp_path, plan)
    report["stages"][0]["diagnostics"]["geometry_trace"] = 1
    report_path.write_text(json.dumps(report))
    (folder / "site-000.pt").write_bytes(b"changed candidate")
    with pytest.raises(ValueError, match="hash"):
        smoke_gate(tmp_path, plan)


def test_worker_refuses_plan_drift_before_dispatch(tmp_path):
    plan = build_plan(*manifests())
    (tmp_path / PLAN).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan drift"):
        worker(tmp_path, SMOKE, "incorrect")


def test_submission_is_exclusive_gated_and_idempotent(tmp_path, monkeypatch):
    import scripts.submit_gsm8k_downstream_sweep as submit
    import sys
    from types import SimpleNamespace

    root = tmp_path / "results/tangent/gsm8k"
    root.mkdir(parents=True)
    plan = build_plan(*manifests())
    (root / PLAN).write_text(json.dumps(plan))
    monkeypatch.setattr(submit, "__file__", str(tmp_path / "scripts/submit_gsm8k_downstream_sweep.py"))
    monkeypatch.setattr(sys, "argv", ["submit", "--dense-job", "64", "--submit"])
    commands = []

    def scheduled(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {100 + len(commands)}\n")

    monkeypatch.setattr(submit.subprocess, "run", scheduled)
    submit.main()
    submit.main()
    assert len(commands) == 9
    for index, argv in enumerate(commands):
        assert "--shared" not in argv and argv[argv.index("--gpu-memory") + 1] == "64G"
        assert argv[argv.index("--memory") + 1] == "80G"
        assert argv[argv.index("--plan-sha256") + 1] == digest(plan)
        if index == 0:
            assert "--depends-on-all" not in argv
        else:
            dependencies = argv[argv.index("--depends-on-all") + 1]
            assert dependencies == "101" if index % 2 else dependencies.endswith(",64")
