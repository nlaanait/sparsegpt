"""Unit tests for SparseGPT unstructured vs N:M (block) sparsity."""

from __future__ import annotations

import contextlib
import io
import unittest

import torch
from torch import nn

from sparsegpt import SparseGPT

CUDA = torch.cuda.is_available()


def _make_layer(in_features=64, out_features=32, device="cpu", seed=0):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    layer = nn.Linear(in_features, out_features, bias=False, device=device)
    # Non-degenerate Hessian: full-rank activation covariance.
    x = torch.randn(8, 16, in_features, device=device)
    return layer, x


def _calibrate_and_prune(layer, x, *, sparsity=0.0, prunen=0, prunem=0, blocksize=16):
    gpt = SparseGPT(layer)
    gpt.add_batch(x, None)
    with contextlib.redirect_stdout(io.StringIO()):
        gpt.fasterprune(
            sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
            percdamp=0.01,
        )
    gpt.free()
    return layer.weight.data


def _nm_zero_counts(weight, n, m):
    """Per-row zero counts inside consecutive groups of size m."""
    rows, cols = weight.shape
    assert cols % m == 0
    grouped = weight.reshape(rows, cols // m, m)
    return (grouped == 0).sum(dim=-1)


class SparseGPTUnstructuredTests(unittest.TestCase):
    def test_unstructured_hits_target_sparsity(self):
        layer, x = _make_layer()
        w = _calibrate_and_prune(layer, x, sparsity=0.5, blocksize=16)
        density = (w == 0).float().mean().item()
        self.assertGreaterEqual(density, 0.45)
        self.assertLessEqual(density, 0.55)
        # Unstructured: some m-blocks will not be exactly n zeros.
        counts = _nm_zero_counts(w, n=2, m=4)
        self.assertFalse(torch.equal(counts, torch.full_like(counts, 2)))

    def test_dense_sparsity_zero_keeps_weights_nonzero(self):
        layer, x = _make_layer()
        dense = layer.weight.data.clone()
        w = _calibrate_and_prune(layer, x, sparsity=0.0, blocksize=16)
        # Zero sparsity may still damp / compensate slightly, but should not
        # force a 50% mask. Compare nonzero fraction to the dense teacher.
        self.assertGreater((w != 0).float().mean().item(), 0.9)
        self.assertEqual(w.shape, dense.shape)


class SparseGPTBlockSparsityTests(unittest.TestCase):
    def test_nm_2_4_enforces_exact_pattern(self):
        layer, x = _make_layer(in_features=64, out_features=32)
        w = _calibrate_and_prune(
            layer, x, sparsity=0.0, prunen=2, prunem=4, blocksize=16
        )
        counts = _nm_zero_counts(w, n=2, m=4)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))
        self.assertAlmostEqual((w == 0).float().mean().item(), 0.5, places=5)

    def test_nm_1_4_enforces_exact_pattern(self):
        layer, x = _make_layer(in_features=64, out_features=16, seed=1)
        w = _calibrate_and_prune(
            layer, x, sparsity=0.0, prunen=1, prunem=4, blocksize=16
        )
        counts = _nm_zero_counts(w, n=1, m=4)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 1)))
        self.assertAlmostEqual((w == 0).float().mean().item(), 0.25, places=5)

    def test_unstructured_and_block_differ_on_same_layer(self):
        layer_u, x = _make_layer(seed=7)
        layer_b = nn.Linear(64, 32, bias=False)
        layer_b.load_state_dict(layer_u.state_dict())

        w_u = _calibrate_and_prune(layer_u, x, sparsity=0.5, blocksize=16)
        w_b = _calibrate_and_prune(
            layer_b, x, sparsity=0.0, prunen=2, prunem=4, blocksize=16
        )

        self.assertFalse(torch.equal(w_u, w_b))
        counts_b = _nm_zero_counts(w_b, n=2, m=4)
        self.assertTrue(torch.equal(counts_b, torch.full_like(counts_b, 2)))
        counts_u = _nm_zero_counts(w_u, n=2, m=4)
        self.assertFalse(torch.equal(counts_u, torch.full_like(counts_u, 2)))


class SparseGPTDampeningTests(unittest.TestCase):
    def test_ill_conditioned_hessian_retries_instead_of_raising(self):
        """Near-singular H used to abort on the second Cholesky; damp must rise."""
        torch.manual_seed(0)
        layer = nn.Linear(64, 32, bias=False)
        gpt = SparseGPT(layer)
        # Rank-1 covariance: PD only after enough diagonal damp.
        direction = torch.randn(64, 1)
        gpt.H = direction @ direction.T
        gpt.nsamples = 8
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            gpt.fasterprune(0.5, blocksize=16, percdamp=1e-8)
        gpt.free()
        self.assertIn("Cholesky failed", buf.getvalue())
        self.assertGreater((layer.weight.data == 0).float().mean().item(), 0.4)


@unittest.skipUnless(CUDA, "CUDA required")
class SparseGPTCUDATests(unittest.TestCase):
    def tearDown(self):
        if CUDA:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _gpu_copy(self, cpu_layer, cpu_x):
        gpu_layer = nn.Linear(
            cpu_layer.in_features,
            cpu_layer.out_features,
            bias=False,
            device="cuda",
        )
        gpu_layer.weight.data.copy_(cpu_layer.weight.data)
        return gpu_layer, cpu_x.cuda()

    def test_unstructured_cuda_matches_cpu(self):
        cpu_layer, cpu_x = _make_layer(device="cpu", seed=3)
        gpu_layer, gpu_x = self._gpu_copy(cpu_layer, cpu_x)

        w_cpu = _calibrate_and_prune(cpu_layer, cpu_x, sparsity=0.5, blocksize=16)
        w_gpu = _calibrate_and_prune(gpu_layer, gpu_x, sparsity=0.5, blocksize=16)
        torch.testing.assert_close(w_gpu.cpu(), w_cpu, atol=1e-4, rtol=1e-4)
        del gpu_layer, gpu_x, w_gpu

    def test_nm_2_4_cuda_matches_cpu(self):
        cpu_layer, cpu_x = _make_layer(device="cpu", seed=5)
        gpu_layer, gpu_x = self._gpu_copy(cpu_layer, cpu_x)

        w_cpu = _calibrate_and_prune(
            cpu_layer, cpu_x, sparsity=0.0, prunen=2, prunem=4, blocksize=16
        )
        w_gpu = _calibrate_and_prune(
            gpu_layer, gpu_x, sparsity=0.0, prunen=2, prunem=4, blocksize=16
        )
        torch.testing.assert_close(w_gpu.cpu(), w_cpu, atol=1e-4, rtol=1e-4)
        counts = _nm_zero_counts(w_gpu, n=2, m=4)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))
        del gpu_layer, gpu_x, w_gpu


if __name__ == "__main__":
    unittest.main()
