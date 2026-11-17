"""Process-local GSM8K defaults, installed before experiment drivers import.

Keep gsm8k_batching.py intact while historical workers are running: its v1
policy hashes that file on every answer. New workers use this frozen policy
and default to 32; explicit legacy batch sizes remain available at startup.
"""
from hashlib import sha256
from pathlib import Path

from sparsifier.eval import gsm8k_batching as _batching

DEFAULT_BATCH_SIZE = 32
_size = DEFAULT_BATCH_SIZE
_source_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
_legacy_policy = _batching.policy()
_legacy_batches = _batching.batches


def policy(size=None):
    selected = _size if size is None else size
    _validate(selected)
    return dict(_legacy_policy, version=f"gsm8k-batch{selected}-v2", batch_size=selected,
                source_sha256=_source_sha256, legacy_source_sha256=_legacy_policy["source_sha256"])


def batches(values, size=None):
    return _legacy_batches(values, _size if size is None else size)


def _validate(size):
    if type(size) is not int or size not in (8, 16, 32):
        raise ValueError("batch size must be 8, 16, or 32")


def configure(size):
    global _size
    _validate(size)
    _size = size
    _batching.BATCH_SIZE = size
    _batching.VERSION = f"gsm8k-batch{size}-v2"
    _batching.batches, _batching.policy = batches, policy


configure(DEFAULT_BATCH_SIZE)
