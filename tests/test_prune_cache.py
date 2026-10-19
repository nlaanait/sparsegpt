"""Checkpoint lookup for a matching SparseGPT pruning config."""

import json
import unittest
from argparse import Namespace
from pathlib import Path

from sparsegpt.prune_cache import (
    canonical_config,
    find_pruned_model,
    prune_config_from_args,
    resolve_save_dir,
    write_prune_config,
)


def _args(**overrides):
    values = dict(
        model="Qwen/Qwen3.5-2B",
        dataset="c4",
        seed=0,
        nsamples=128,
        percdamp=0.01,
        sparsity=0.5,
        prunen=0,
        prunem=0,
        blocksize=128,
        gmp=False,
        wbits=16,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        true_sequential=False,
        geometry_alpha=0.0,
        geometry_probes=None,
    )
    values.update(overrides)
    return Namespace(**values)


class PruneCacheTest(unittest.TestCase):
    def test_finds_matching_checkpoint_and_ignores_other_configs(self):
        with self._tmpdir() as root:
            match = root / "qwen3.5-2b"
            other = root / "qwen3.5-2b-sp0.9"
            self._checkpoint(match, _args())
            self._checkpoint(other, _args(sparsity=0.9))

            found = find_pruned_model(prune_config_from_args(_args()), cache_dir=root)
            self.assertEqual(found, match)

            found_other = find_pruned_model(
                prune_config_from_args(_args(sparsity=0.9)), cache_dir=root
            )
            self.assertEqual(found_other, other)

    def test_ignores_directory_without_weights(self):
        with self._tmpdir() as root:
            bare = root / "config-only"
            write_prune_config(bare, prune_config_from_args(_args()))
            self.assertIsNone(
                find_pruned_model(prune_config_from_args(_args()), cache_dir=root)
            )

    def test_resolve_save_dir_does_not_overwrite_a_different_model(self):
        with self._tmpdir() as root:
            occupied = root / "qwen3.5-2b"
            self._checkpoint(occupied, _args(sparsity=0.5))
            dest = resolve_save_dir(
                str(occupied),
                prune_config_from_args(_args(sparsity=0.75)),
                cache_dir=root,
            )
            self.assertNotEqual(dest, occupied)
            self.assertEqual(dest.parent, root)

    def test_canonical_config_matches_json_roundtrip(self):
        config = prune_config_from_args(_args(percdamp=0.01, sparsity=0.5))
        self.assertEqual(canonical_config(config), json.loads(json.dumps(config)))

    def _checkpoint(self, path: Path, args):
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"weights")
        write_prune_config(path, prune_config_from_args(args))

    def _tmpdir(self):
        import tempfile

        class _PathDir(tempfile.TemporaryDirectory):
            def __enter__(self):
                return Path(super().__enter__())

        return _PathDir()


if __name__ == "__main__":
    unittest.main()
