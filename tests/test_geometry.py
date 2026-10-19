import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
import transformers

from sparsegpt import SparseGPT
from sparsegpt.geometry import GeometrySparseGPT, add_geometry_arguments, apply_geometry
from sparsegpt.geometry_demo import evaluate, prune, resolve_device, save_csr
from sparsegpt.geometry_export import export_layer_probes
from sparsegpt.geometry_features import BlockDictionary


CUDA = torch.cuda.is_available()


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.layer = nn.Linear(8, 4, bias=False)
        self.x = torch.randn(3, 5, 8)
        self.t = torch.randn(8, 7)

    def test_covariance_matches_augmented_objective(self):
        g = GeometrySparseGPT(self.layer)
        g.add_batch(self.x[:1], None)
        g.add_batch(self.x[1:], None)
        g.add_geometry(self.t, alpha=.7, chunk_size=3)
        x = self.x.reshape(-1, 8).T
        expected = 2 * 15 / 3 * (x @ x.T / 15 + .7 * self.t @ self.t.T / 7)
        torch.testing.assert_close(g.H, expected)

    def test_zero_alpha_exact_upstream_hessian_and_weights(self):
        original = SparseGPT(self.layer)
        layer = nn.Linear(8, 4, bias=False)
        layer.load_state_dict(self.layer.state_dict())
        geometry = GeometrySparseGPT(layer)
        for g in (original, geometry):
            g.add_batch(self.x, None)
        geometry.add_geometry(alpha=0)
        self.assertTrue(torch.equal(original.H, geometry.H))
        with contextlib.redirect_stdout(io.StringIO()):
            original.fasterprune(.5, blocksize=4)
            geometry.fasterprune(.5, blocksize=4)
        self.assertTrue(torch.equal(original.layer.weight, geometry.layer.weight))

    def test_matches_explicit_augmented_calibration_and_pruning(self):
        g = GeometrySparseGPT(self.layer)
        other = nn.Linear(8, 4, bias=False)
        other.load_state_dict(self.layer.state_dict())
        direct = SparseGPT(other)
        g.add_batch(self.x, None)
        g.add_geometry(self.t, alpha=.7)
        # Exactly the same Hessian, including upstream 2/sample scale.
        x = self.x.reshape(-1, 8).T
        a = torch.cat((x / np.sqrt(3), self.t * np.sqrt(.7 * 15 / (3 * 7))), dim=1)
        direct.add_batch(a.T, None)
        torch.testing.assert_close(g.H, direct.H)
        with contextlib.redirect_stdout(io.StringIO()):
            g.fasterprune(.5, blocksize=4)
            direct.fasterprune(.5, blocksize=4)
        torch.testing.assert_close(g.layer.weight, direct.layer.weight)

    def test_probe_count_invariance(self):
        a, b = GeometrySparseGPT(self.layer), GeometrySparseGPT(self.layer)
        for g in (a, b):
            g.add_batch(self.x, None)
        a.add_geometry(self.t, alpha=.4)
        b.add_geometry(self.t.repeat(1, 3), alpha=.4, chunk_size=5)
        torch.testing.assert_close(a.H, b.H)

    def test_geometry_rescues_dead_activation_direction(self):
        # Unobserved inputs are force-zeroed as dead columns without geometry,
        # even when the dense teacher has mass on those columns.
        x = torch.tensor([[1., 0., 0., 0.], [2., 0., 0., 0.]])
        act_layer = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            act_layer.weight.zero_()
            act_layer.weight[:, 1:] = 1
        act = SparseGPT(act_layer)
        act.add_batch(x, None)
        self.assertTrue((torch.diag(act.H)[1:] == 0).all())
        with contextlib.redirect_stdout(io.StringIO()):
            act.fasterprune(0.0, blocksize=4)
        self.assertTrue(torch.equal(act_layer.weight[:, 1:], torch.zeros(2, 3)))

        # Geometry lifts those diagonals so SparseGPT can retain the mass.
        # Use sparsity below the equal-score tie cliff of the upstream threshold.
        layer = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, 1:] = 1
        g = GeometrySparseGPT(layer)
        g.add_batch(x, None)
        g.add_geometry(torch.eye(4), alpha=1)
        self.assertTrue((torch.diag(g.H) > 0).all())
        with contextlib.redirect_stdout(io.StringIO()):
            g.fasterprune(0.1, blocksize=4)
        self.assertGreater(torch.count_nonzero(layer.weight[:, 1:]).item(), 0)

    def test_state_and_invalid_probe_errors_do_not_change_hessian(self):
        g = GeometrySparseGPT(self.layer)
        with self.assertRaises(RuntimeError):
            g.add_geometry(self.t)
        g.add_batch(self.x, None)
        old = g.H.clone()
        for t in (torch.zeros(7, 3), torch.zeros(8, 0), torch.full((8, 3), float('nan'))):
            with self.assertRaises(ValueError):
                g.add_geometry(t)
        torch.testing.assert_close(g.H, old)
        g.add_geometry(self.t)
        with self.assertRaises(RuntimeError):
            g.add_geometry(self.t)
        with self.assertRaises(RuntimeError):
            g.add_batch(self.x, None)
        with self.assertRaises(TypeError):
            GeometrySparseGPT(nn.Conv2d(2, 2, 3))

    def test_conv1d_and_nm_pruning(self):
        layer = transformers.Conv1D(4, 8)
        g = GeometrySparseGPT(layer)
        g.add_batch(self.x, None)
        g.add_geometry(self.t)
        with contextlib.redirect_stdout(io.StringIO()):
            g.fasterprune(.5, prunen=2, prunem=4, blocksize=4)
        counts = (layer.weight.T.reshape(4, 2, 4) == 0).sum(dim=2)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))

    def test_cli_probe_loading_and_csr_roundtrip(self):
        parser = argparse.ArgumentParser()
        add_geometry_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / '0').mkdir()
            np.save(path / '0' / 'fc1.npy', self.t.numpy())
            args = parser.parse_args(['--geometry-alpha', '0.5', '--geometry-probes', directory])
            g = GeometrySparseGPT(self.layer)
            g.add_batch(self.x, None)
            apply_geometry(g, args, 0, 'fc1')
            self.assertTrue(g.geometry_finalized)
            with self.assertRaises(FileNotFoundError):
                apply_geometry(g, args, 0, 'missing')
            s = prune(
                self.layer.weight.detach().numpy(),
                self.x.reshape(-1, 8).T.numpy(),
                self.t.numpy(),
                .5,
                .5,
            )
            save_csr(path / 's.npz', s)
            with np.load(path / 's.npz') as f:
                restored = np.zeros(tuple(f['shape']), dtype=np.float32)
                for row in range(restored.shape[0]):
                    lo, hi = f['indptr'][row:row + 2]
                    restored[row, f['indices'][lo:hi]] = f['data'][lo:hi]
                np.testing.assert_array_equal(restored, s)

    def test_small_gram_metric_matches_explicit(self):
        rng = np.random.default_rng(0)
        w, s, t = rng.normal(size=(3, 8)), rng.normal(size=(3, 8)), rng.normal(size=(8, 20))
        u, v = w @ t, s @ t
        expected = np.linalg.norm(v.T @ v - u.T @ u) ** 2 / np.linalg.norm(u.T @ u) ** 2
        self.assertAlmostEqual(evaluate(s, w, t, t)['probe_gram_relative_mse'], expected)

    def test_learned_probe_pipeline(self):
        rng = np.random.default_rng(3)
        h = rng.normal(size=(8, 80))
        model = BlockDictionary(4, 2, 2).fit(h, iterations=3)
        t = model.tangents(h)
        np.testing.assert_allclose(np.linalg.norm(t, axis=0), 1)
        s = prune(self.layer.weight.detach().numpy(), h, t, 1, .5)
        self.assertTrue(np.isfinite(s).all())
        self.assertGreaterEqual((s == 0).mean(), .5)

    def test_export_layer_probes_layout(self):
        rng = np.random.default_rng(4)
        h = rng.normal(size=(8, 60)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = export_layer_probes(
                Path(directory),
                block_index=2,
                module_name='fc1',
                activations=h,
                groups=4,
                width=2,
                active=2,
                seed=0,
            )
            self.assertEqual(path, Path(directory) / '2' / 'fc1.npy')
            probes = np.load(path)
            self.assertEqual(probes.shape[0], 8)
            self.assertGreater(probes.shape[1], 0)

    def test_resolve_device_cpu(self):
        self.assertEqual(resolve_device('cpu').type, 'cpu')
        if not CUDA:
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')


@unittest.skipUnless(CUDA, 'CUDA required')
class GeometryCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        self.device = torch.device('cuda')
        self.layer = nn.Linear(8, 4, bias=False, device=self.device)
        self.x = torch.randn(3, 5, 8, device=self.device)
        self.t = torch.randn(8, 7, device=self.device)

    def test_hessian_and_prune_match_cpu(self):
        cpu_layer = nn.Linear(8, 4, bias=False)
        cpu_layer.load_state_dict({k: v.cpu() for k, v in self.layer.state_dict().items()})
        cpu = GeometrySparseGPT(cpu_layer)
        gpu = GeometrySparseGPT(self.layer)
        cpu.add_batch(self.x.cpu(), None)
        gpu.add_batch(self.x, None)
        cpu.add_geometry(self.t.cpu(), alpha=.7, chunk_size=3)
        gpu.add_geometry(self.t, alpha=.7, chunk_size=3)
        torch.testing.assert_close(gpu.H.cpu(), cpu.H, atol=1e-5, rtol=1e-5)
        with contextlib.redirect_stdout(io.StringIO()):
            cpu.fasterprune(.5, blocksize=4)
            gpu.fasterprune(.5, blocksize=4)
        torch.testing.assert_close(
            gpu.layer.weight.cpu(),
            cpu.layer.weight,
            atol=1e-5,
            rtol=1e-5,
        )
        gpu.free()
        cpu.free()

    def test_numpy_probe_chunks_on_cuda_layer(self):
        g = GeometrySparseGPT(self.layer)
        g.add_batch(self.x, None)
        probes = self.t.detach().cpu().numpy().astype(np.float64)
        g.add_geometry(probes, alpha=1.0, chunk_size=2)
        self.assertEqual(g.H.device.type, 'cuda')
        self.assertTrue(torch.isfinite(g.H).all())
        with contextlib.redirect_stdout(io.StringIO()):
            g.fasterprune(.5, blocksize=4)
        self.assertEqual(self.layer.weight.device.type, 'cuda')
        g.free()

    def test_demo_prune_cuda_matches_cpu(self):
        w = self.layer.weight.detach().cpu().numpy()
        h = self.x.reshape(-1, 8).T.cpu().numpy()
        t = self.t.cpu().numpy()
        cpu = prune(w, h, t, alpha=.7, sparsity=.5, device='cpu', blocksize=4)
        gpu = prune(w, h, t, alpha=.7, sparsity=.5, device='cuda', blocksize=4)
        np.testing.assert_allclose(gpu, cpu, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
