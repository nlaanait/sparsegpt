import json
import random
from pathlib import Path
import pytest
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import read_completed
from sparsifier.eval.gsm8k_decision import prepare_manifest, validate_protocol, screen_decision, run_tiers
from scripts.submit_gsm8k_decision_pilot import build_plan, command


def manifests(seed=0):
    reserved = {"source_split":"train","manifest_sha256":"reserved", "rows":[
        {"id":f"gsm8k/train/{i}","source_index":i,"split":"validation"} for i in range(1000,1512)]}
    calibration = {"manifest_sha256":"calibration","rows":[{"source_index":2}]}
    return prepare_manifest(reserved,calibration,seed)


def record(row, correct=True):
    return {"id":row["id"],"correct":correct,"config_sha256":"candidate","truncated":False}


def test_one_local_seeded_shuffle_nested_identity_for_every_arm():
    state = random.getstate()
    manifest = manifests()
    assert random.getstate() == state
    assert manifest == manifests()
    assert [r["id"] for r in manifest["rows"]] != [r["id"] for r in manifests(1)["rows"]]
    validate_protocol(manifest)
    plan = build_plan(manifest)
    assert len(plan["tasks"]) == 13
    assert plan["screen"] == 64 and plan["light"] == 128 and plan["max_new_tokens"] == 2048
    assert all(t["block"]==15 and t["sparsity"]==.3 for t in plan["tasks"] if t["phase"]=="prune")
    assert manifest["screen_ids_sha256"] == digest([r["id"] for r in manifest["rows"][:64]])
    assert manifest["light_ids_sha256"] == digest([r["id"] for r in manifest["rows"]])


def test_shared_64_threshold_is_exact_and_does_not_depend_on_suffix():
    manifest = manifests()
    baseline = {r["id"]:record(r) for r in manifest["rows"]}
    done = {r["id"]:record(r,i>=9) for i,r in enumerate(manifest["rows"])}
    assert screen_decision(manifest,baseline,done)["passed"]
    done[manifest["rows"][9]["id"]]["correct"] = False
    assert not screen_decision(manifest,baseline,done)["passed"]


@pytest.mark.parametrize("passed",[True,False])
def test_screen_resume_preserves_partial_batch_and_never_advances_rejected_answers(tmp_path,passed):
    manifest = manifests()
    baseline = {r["id"]:record(r) for r in manifest["rows"]}
    with (tmp_path/"answers.jsonl").open("w") as stream:
        for row in manifest["rows"][:10]:
            stream.write(json.dumps(record(row,passed))+"\n")
    calls = []
    def evaluate(model,tokenizer,rows,output,config_hash,*,max_new_tokens):
        assert max_new_tokens == 2048
        assert len(rows) == 32
        calls.extend(r["id"] for r in rows)
        with (output/"answers.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(record(row,passed))+"\n")
    done, decision = run_tiers(evaluate,None,None,manifest,tmp_path,"candidate",baseline=baseline)
    assert decision["passed"] == passed
    assert len(done) == (128 if passed else 64)
    assert len(calls) == (128 if passed else 64)
    assert calls[:32] == [r["id"] for r in manifest["rows"][:32]]
    assert len(set(calls)) == len(calls)
    before = list(calls)
    run_tiers(evaluate,None,None,manifest,tmp_path,"candidate",baseline=baseline)
    assert calls == before


def test_manifest_tampering_and_resources_are_explicit(tmp_path):
    manifest = manifests()
    manifest["rows"].reverse()
    with pytest.raises(ValueError,match="drift"):
        validate_protocol(manifest)
    argv = command(tmp_path,{"name":"pilot","phase":"prune"},[1,2],"hash")
    assert "--shared" not in argv and "1,2" in argv
    assert ".pixi/envs/gb10/bin/python" in " ".join(argv)


def test_migration_refuses_to_cancel_running_or_wrong_jobs(monkeypatch):
    from types import SimpleNamespace
    import scripts.migrate_gsm8k_decision_evaluations as migration
    monkeypatch.setattr(migration.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="State=Running (R) source-original512"))
    with pytest.raises(RuntimeError,match="do not cancel"):
        migration.queued_check(1,"source")
    monkeypatch.setattr(migration.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="State=Queued (PD) wrong-original512"))
    with pytest.raises(RuntimeError,match="do not cancel"):
        migration.queued_check(1,"source")
    monkeypatch.setattr(migration.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="State=Queued (PD) source-original512"))
    migration.queued_check(1,"source")
