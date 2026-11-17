import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("size", [8, 16, 32])
def test_worker_override_preserves_order_and_explicit_batch_sizes(size):
    script = f"""
import json
from sparsifier.eval.gsm8k_batch_runtime import configure_batch_size
configure_batch_size({size})
from sparsifier.eval.gsm8k_batching import batches, policy
values = list(range(70))
groups = list(batches(values))
print(json.dumps({{'sizes': list(map(len, groups)), 'values': sum(groups, []),
                  'explicit': list(map(len, batches(values, 7))), 'policy': policy()}}))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
    result = subprocess.run([sys.executable, "-c", script], env=env, check=True,
                            capture_output=True, text=True)
    actual = json.loads(result.stdout)
    assert actual["values"] == list(range(70))
    assert actual["sizes"] == [size]*(70//size)+([70 % size] if 70 % size else [])
    assert actual["explicit"] == [7]*10
    assert actual["policy"]["batch_size"] == size
    assert actual["policy"]["version"] == f"gsm8k-batch{size}-v2"
    assert len(actual["policy"]["source_sha256"]) == 64
    assert len(actual["policy"]["legacy_source_sha256"]) == 64


def test_direct_default_matches_explicit_32_and_preserves_legacy_worker():
    script = """
import importlib.util, json
from pathlib import Path
path = Path('src/sparsifier/eval/gsm8k_batching.py')
spec = importlib.util.spec_from_file_location('already_running_v1', path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
before = old.policy()
from sparsifier.eval.gsm8k_batching import BATCH_SIZE, batches, policy
direct = policy()
assert BATCH_SIZE == 32 and [len(x) for x in batches(list(range(70)))] == [32,32,6]
from sparsifier.eval.gsm8k_batch_runtime import configure_batch_size
configure_batch_size(32)
assert policy() == direct
assert old.policy() == before and before['batch_size'] == 8
print(json.dumps(direct))
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root, check=True,
        env=dict(os.environ, PYTHONPATH=str(root/"src")), capture_output=True, text=True)
    assert json.loads(result.stdout)["batch_size"] == 32


def test_launcher_defaults_to_32_and_announces_actual_policy():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "sparsifier.eval.gsm8k_batch_runtime",
        "sparsifier.eval.gsm8k_decision_full", "--help"], cwd=root, check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root/"src")), capture_output=True, text=True)
    event = json.loads(result.stdout.splitlines()[0])
    assert event["event"] == "batching_policy" and event["batching"]["batch_size"] == 32


def test_dense_preparation_uses_same_batch_and_only_reference_inputs():
    from sparsifier.eval.gsm8k_batch_runtime import reference_command
    command = reference_command(32, "sparsifier.eval.gsm8k_decision_full",
        ["--manifest", "full.json", "--calibration-manifest", "cal.json",
         "--source", "candidate", "--reference", "dense32", "--output", "evaluation"], "light.json")
    assert command == [sys.executable, "-u", "-m", "sparsifier.eval.gsm8k_batch_runtime",
                       "--batch-size", "32", "sparsifier.eval.gsm8k_decision",
                       "--manifest", "light.json", "--calibration-manifest", "cal.json",
                       "--output", "dense32"]
