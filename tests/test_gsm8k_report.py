import pytest
import json
from sparsifier.eval.gsm8k_report import paired_accuracy, summarize, check_protocol


def test_summary_ignores_dense_smoke(tmp_path):
    folder = tmp_path / "qwen-9b-gsm8k-dense-smoke-v1"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": "test"}))
    assert summarize(tmp_path)["arms"] == {}


def test_summary_excludes_invalid_legacy_local_results(tmp_path):
    folder = tmp_path / 'qwen-9b-gsm8k-local-a1-s0-v1'
    folder.mkdir()
    (folder/'config.json').write_text(json.dumps({'arm':'local','alpha':1,'seed':0}))
    # Exclusion must happen before interpreting the old behavioral report.
    (folder/'report.json').write_text(json.dumps({'complete':True}))
    entry = summarize(tmp_path)['arms'][folder.name]
    assert entry['excluded']
    assert not entry['complete']
    assert 'paired_vs_task_sparsegpt' not in entry


def test_protocol_rejects_changed_budget_or_checkpoint():
    config = dict(model="test", revision="a", capture_sha256="b", manifest_sha256="c",
                  evaluation_split="validation", max_new_tokens=512)
    check_protocol(config, config)
    for key in config:
        with pytest.raises(ValueError, match=key):
            check_protocol({**config, key: "changed"}, config)


def test_paired_summary_joins_by_id_and_checks_targets():
    base = {"a": {"correct": False, "family": "x", "expected_answer": "2"},
            "b": {"correct": True, "family": "y", "expected_answer": "3"}}
    better = {"b": dict(base["b"]), "a": {**base["a"], "correct": True}}
    assert paired_accuracy(better, base, draws=50)["accuracy_difference"] == .5
    with pytest.raises(ValueError):
        paired_accuracy({"a": better["a"]}, base)
    better["a"]["expected_answer"] = "9"
    with pytest.raises(ValueError):
        paired_accuracy(better, base)
