"""Content-addressed replay of frozen low-rank curvature factors.

No network graph, dense parameter Hessian, or collection of dense Z matrices
is retained. A bounded LRU caches tensors; one factor and solver workspaces
are additional memory. Large tensors are memory mapped and applied in chunks.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import torch

from sparsifier.prune.whole_neuron_metric import FrozenNeuronMetric

VERSION = "frozen-factor-replay-v1"


class DiskFactorReplay:
    def __init__(self, path, *, provenance=None, cache_budget_bytes=256*1024**2,
                 disk_budget_bytes=64*1024**3):
        self.path = Path(path)
        if cache_budget_bytes < 0 or disk_budget_bytes < 1:
            raise ValueError("nonnegative cache and positive disk budgets required")
        self.path.mkdir(parents=True, exist_ok=True)
        self.cache_budget_bytes, self.disk_budget_bytes = cache_budget_bytes, disk_budget_bytes
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.entries, self.blobs = [], {}
        self.provenance = provenance or {}
        self.sealed = False
        self.verified = set()
        self.disk_bytes = sum(p.stat().st_size for p in self.path.glob("*.pt"))

    def _store(self, tensor):
        tensor = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        key = digest.hexdigest()
        path = self.path/(key+".pt")
        if key not in self.blobs:
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if old.dtype != tensor.dtype or old.shape != tensor.shape or not torch.equal(old, tensor):
                    raise ValueError("curvature replay content key mismatch")
            if not path.exists():
                if self.disk_bytes+tensor.numel()*tensor.element_size()+8192 > self.disk_budget_bytes:
                    raise ValueError("curvature replay disk budget exceeded")
                temporary = path.with_suffix(".tmp")
                torch.save(tensor, temporary)
                temporary.replace(path)
                self.disk_bytes += path.stat().st_size
            self.blobs[key] = {"sha256": self._hash(path), "bytes": tensor.numel()*tensor.element_size()}
        return key

    @staticmethod
    def _hash(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                result.update(block)
        return result.hexdigest()

    def append(self, kind, tensors, *, weight=1., term="base"):
        if self.sealed:
            raise RuntimeError("cannot append to frozen replay")
        if kind not in ("weight", "independent", "local") or term not in ("base", "input"):
            raise ValueError("unknown curvature factor")
        expected = {"gradient", "inputs"} if kind == "weight" else ({"gradient", "basis"} if kind == "independent" else {"inputs", "basis"})
        if tensors.keys() != expected:
            raise ValueError("curvature factor fields do not match its kind")
        first = tensors["gradient"] if kind != "local" else tensors["inputs"]
        second = tensors["inputs"] if kind == "weight" else tensors["basis"]
        if (first.ndim != 2 or second.ndim != (2 if kind == "weight" else 3)
            or len(first) < 1 or len(first) != len(second) or any(not t.is_floating_point() for t in tensors.values())):
            raise ValueError("invalid low-rank curvature factor axes or dtype")
        if not torch.isfinite(torch.tensor(weight)) or weight < 0:
            raise ValueError("finite nonnegative curvature weight required")
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("nonfinite curvature factor")
        self.entries.append({"kind": kind, "term": term, "weight": float(weight),
                             "tensors": {k: self._store(v) for k, v in tensors.items()}})

    def seal(self):
        from sparsifier.data.gsm8k import write_immutable
        manifest = {"version": VERSION, "provenance": self.provenance,
                    "entries": self.entries, "blobs": self.blobs}
        write_immutable(self.path/"manifest.json", manifest)
        self.sealed = True
        self.signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.manifest_stamp = (self.path/"manifest.json").stat().st_mtime_ns
        return self._hash(self.path/"manifest.json")

    @classmethod
    def open(cls, path, *, provenance=None, expected_manifest_sha256=None, **kwargs):
        instance = cls(path, provenance=provenance, **kwargs)
        if expected_manifest_sha256 is not None and instance._hash(instance.path/"manifest.json") != expected_manifest_sha256:
            raise ValueError("curvature replay manifest hash mismatch")
        manifest = json.loads((instance.path/"manifest.json").read_text())
        if manifest["version"] != VERSION or (provenance is not None and manifest["provenance"] != provenance):
            raise ValueError("curvature replay provenance mismatch")
        instance.entries, instance.blobs = manifest["entries"], manifest["blobs"]
        instance.provenance, instance.sealed = manifest["provenance"], True
        instance.signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        instance.manifest_stamp = (instance.path/"manifest.json").stat().st_mtime_ns
        return instance

    def _load(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or key not in self.blobs:
            raise ValueError("invalid curvature replay content key")
        path = self.path/(key+".pt")
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if (key, stamp) not in self.verified:
            if self._hash(path) != self.blobs[key]["sha256"]:
                raise ValueError("curvature replay content hash mismatch")
            self.verified.add((key, stamp))
        if key in self.cache:
            tensor = self.cache.pop(key)
            self.cache[key] = tensor
            return tensor
        tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        size = tensor.numel()*tensor.element_size()
        if size <= self.cache_budget_bytes:
            while self.cache_bytes+size > self.cache_budget_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.numel()*old.element_size()
            self.cache[key], self.cache_bytes = tensor, self.cache_bytes+size
        return tensor

    def __iter__(self):
        if not self.sealed:
            raise RuntimeError("seal curvature factors before solving")
        manifest = {"version": VERSION, "provenance": self.provenance, "entries": self.entries, "blobs": self.blobs}
        if (hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != self.signature
            or (self.path/"manifest.json").stat().st_mtime_ns != self.manifest_stamp):
            raise ValueError("frozen curvature replay manifest changed")
        for entry in self.entries:
            yield dict(entry, tensors={k: self._load(v) for k, v in entry["tensors"].items()})


def weight_contraction(g, x, like, chunk):
    z = torch.zeros_like(like)
    for start in range(0, len(g), chunk):
        z.addmm_(g[start:start+chunk].to(like).T, x[start:start+chunk].to(like))
    return z


class ReplayNeuronMetric(FrozenNeuronMetric):
    def __init__(self, ambient_count, replay, *, gamma=1., eta=0., chunk_size=32):
        super().__init__(ambient_count, gamma=gamma)
        if chunk_size < 1 or not math.isfinite(eta) or eta < 0:
            raise ValueError("positive chunk size and nonnegative eta required")
        self.replay, self.eta, self.chunk_size = replay, eta, chunk_size
        self.support = None

    def with_support(self, support):
        result = copy.copy(self)
        result.support = None if bool(support.all()) else support.detach().clone()
        return result

    def with_weights(self, gamma, eta):
        if any(not math.isfinite(v) or v < 0 for v in (gamma, eta)):
            raise ValueError("finite nonnegative geometric coefficients required")
        result = copy.copy(self)
        result.gamma, result.eta = gamma, eta
        return result

    def _factors(self):
        for factor in self.replay:
            coefficient = (self.gamma if factor["term"] == "base" else self.eta)*factor["weight"]/self.ambient_count
            if coefficient:
                yield factor["kind"], factor["tensors"], coefficient

    def action(self, a, h0, ridge=0.):
        result = a@h0+torch.as_tensor(ridge, device=a.device, dtype=a.dtype)*a
        support = self.support.to(a) if self.support is not None else None
        working = a if support is None else a*support
        for kind, values, coefficient in self._factors():
            if kind == "weight":
                g, x = values["gradient"], values["inputs"]
                scalar = a.new_zeros(())
                for start in range(0, len(g), self.chunk_size):
                    gs, xs = g[start:start+self.chunk_size].to(a), x[start:start+self.chunk_size].to(a)
                    scalar += ((gs@working)*xs).sum()
                for start in range(0, len(g), self.chunk_size):
                    gs, xs = g[start:start+self.chunk_size].to(a), x[start:start+self.chunk_size].to(a)
                    result.addmm_(gs.T, xs*(coefficient*scalar) if support is None else xs*(coefficient*scalar)*support)
            elif kind == "independent":
                g, u = values["gradient"], values["basis"]
                for start in range(0, len(g), self.chunk_size):
                    gs, us = g[start:start+self.chunk_size].to(a), u[start:start+self.chunk_size].to(a)
                    if support is not None:
                        us = us*support[None, :, None]
                    scores = ((gs@working).unsqueeze(-1)*us).sum(1)
                    result.addmm_(gs.T, (us*scores[:, None, :]).sum(-1), alpha=coefficient)
            else:
                for start in range(0, len(values["inputs"]), self.chunk_size):
                    x = values["inputs"][start:start+self.chunk_size].to(a)
                    q = values["basis"][start:start+self.chunk_size].to(a)
                    if support is not None:
                        x = x*support
                    response = x@working.T
                    projected = (q*response[..., None]).sum(1)
                    result.addmm_((q*projected[:, None, :]).sum(-1).T, x, alpha=coefficient)
        return result

    def diagonal(self, shape, *, device=None, dtype=torch.float64):
        # Diagonals are unnormalized by gamma/eta, matching FrozenNeuronMetric.
        result = torch.zeros(shape, device=device, dtype=dtype)
        for factor in self.replay:
            if factor["term"] != "base":
                continue
            result += self.factor_diagonal(factor, result)*factor["weight"]/self.ambient_count
        return result

    def input_diagonal(self, shape, *, device=None, dtype=torch.float64):
        result = torch.zeros(shape, device=device, dtype=dtype)
        for factor in self.replay:
            if factor["term"] == "input":
                result += self.factor_diagonal(factor, result)*factor["weight"]/self.ambient_count
        return result

    def factor_diagonal(self, factor, like):
        v = factor["tensors"]
        if factor["kind"] == "weight":
            return weight_contraction(v["gradient"], v["inputs"], like, self.chunk_size).square()
        if factor["kind"] == "independent":
            result = torch.zeros_like(like)
            for start in range(0, len(v["gradient"]), self.chunk_size):
                g, u = v["gradient"][start:start+self.chunk_size].to(like), v["basis"][start:start+self.chunk_size].to(like)
                result.addmm_(g.square().T, u.square().sum(-1))
            return result
        result = torch.zeros_like(like)
        for start in range(0, len(v["inputs"]), self.chunk_size):
            q, x = v["basis"][start:start+self.chunk_size].to(like), v["inputs"][start:start+self.chunk_size].to(like)
            result.addmm_(q.square().sum(-1).T, x.square())
        return result

    def curvature_diagonal(self, shape, *, device=None, dtype=torch.float64):
        value = torch.zeros(shape, device=device, dtype=dtype)
        if self.gamma:
            value += self.gamma*self.diagonal(shape, device=device, dtype=dtype)
        if self.eta:
            value += self.eta*self.input_diagonal(shape, device=device, dtype=dtype)
        return value if self.support is None else value*self.support.to(value)

    def direct_scores(self, weight, h0):
        scores = weight.square().sum(0)*h0.diagonal()
        geometric_weight = weight if self.support is None else weight*self.support.to(weight)
        for kind, v, coefficient in self._factors():
            if kind == "weight":
                z = weight_contraction(v["gradient"], v["inputs"], weight, self.chunk_size)
                scores += coefficient*(z*geometric_weight).sum(0).square()
            elif kind == "independent":
                scores += coefficient*((v["gradient"].to(weight)@geometric_weight).square()*v["basis"].to(weight).square().sum(-1)).sum(0)
            else:
                projection = torch.einsum("bpr,pq->brq", v["basis"].to(weight), geometric_weight)
                scores += coefficient*(projection.square().sum(1)*v["inputs"].to(weight).square()).sum(0)
        return scores

    def refit(self, weight, keep, h0, ridge, *, method="cg", **kwargs):
        if method == "auto":
            method = "cg"
        return super().refit(weight, keep, h0, ridge, method=method, **kwargs)


class ReplayProbeConsumer:
    def __init__(self, reference, replay=None, *, term="base", collect_group_scores=True):
        self.reference = reference.detach().to(device="cpu", dtype=torch.float64) if collect_group_scores else None
        self.group_scores = torch.zeros(reference.shape[1], dtype=torch.float64)
        self.replay, self.term, self.count = replay, term, 0

    def __call__(self, event):
        coefficient = event["coefficient"]
        kind = event.get("kind", "weight")
        if kind == "weight":
            if self.reference is not None:
                self.group_scores += coefficient*(event["contraction"]*self.reference).sum(0).square()
            values = {k: event[k] for k in ("gradient", "inputs")}
        else:
            if self.reference is not None:
                g, u = event["gradient"].double(), event["basis"].double()
                self.group_scores += coefficient*((g@self.reference).square()*u.square().sum(-1)).sum(0)
            values = {k: event[k] for k in ("gradient", "basis")}
        if self.replay is not None:
            self.replay.append(kind, values, weight=coefficient, term=self.term)
        self.count += 1
