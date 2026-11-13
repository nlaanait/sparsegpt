"""Atomic triplet stages and self-contained mixed-width compact artifacts."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from sparsifier.prune.whole_neuron import VERSION, SwiGLUTriplet, WholeNeuronCandidate, apply_candidate


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_stage(path, triplet, candidate, *, config_hash, upstream, block, metadata=None):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite a completed triplet stage")
    tensors = {"gate": candidate.gate.cpu(), "up": candidate.up.cpu(), "down": candidate.down.cpu()}
    payload = {"schema": 1, "method_version": VERSION, "config_sha256": config_hash,
               "upstream": upstream, "site": [block, "mlp_triplet"],
               "keep": candidate.keep.cpu(), **tensors, "diagnostics": candidate.diagnostics,
               "original_hashes": dict(zip(tensors, map(tensor_hash, triplet.weights))),
               "candidate_hashes": {name: tensor_hash(value) for name, value in tensors.items()},
               "original_shapes": {name: list(value.shape) for name, value in zip(tensors, triplet.weights)},
               "projection_names": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
               "kept_indices": torch.where(candidate.keep)[0].cpu().tolist(),
               "removed_indices": torch.where(~candidate.keep)[0].cpu().tolist(),
               "metadata": metadata or {}}
    # Validate serialized data on the actual reference without changing it.
    validate_stage(payload, triplet, config_hash=config_hash, upstream=upstream, block=block)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    checksum = file_hash(path)
    hash_temporary = path.with_suffix(".sha256.tmp")
    hash_temporary.write_text(checksum+"\n")
    hash_temporary.replace(path.with_suffix(".sha256"))
    return checksum


def validate_stage(payload, triplet, *, config_hash, upstream, block):
    if (payload.get("schema") != 1 or payload.get("method_version") != VERSION
            or payload["config_sha256"] != config_hash or payload["upstream"] != upstream
            or payload["site"] != [block, "mlp_triplet"]):
        raise ValueError("triplet stage provenance mismatch")
    values = []
    for name, reference in zip(("gate", "up", "down"), triplet.weights):
        value = payload[name]
        if (tensor_hash(reference) != payload["original_hashes"][name]
                or tensor_hash(value) != payload["candidate_hashes"][name]
                or value.shape != reference.shape or value.dtype != reference.dtype
                or not torch.isfinite(value).all()):
            raise ValueError("triplet tensor hash/shape/dtype mismatch")
        values.append(value.to(reference.device))
    keep = payload["keep"].to(triplet.gate.weight.device)
    if keep.dtype != torch.bool or keep.shape != (len(triplet.gate.weight),) or not keep.any():
        raise ValueError("invalid serialized neuron mask")
    if (payload["kept_indices"] != torch.where(keep)[0].cpu().tolist()
            or payload["removed_indices"] != torch.where(~keep)[0].cpu().tolist()):
        raise ValueError("serialized neuron ordering mismatch")
    g, u, d = values
    original_g, original_u, _ = triplet.weights
    if (g[~keep].any() or u[~keep].any() or d[:, ~keep].any()
            or not torch.equal(g[keep], original_g[keep]) or not torch.equal(u[keep], original_u[keep])):
        raise ValueError("serialized triplet invariant failure")
    return WholeNeuronCandidate(keep, g, u, d, payload["diagnostics"])


def load_stage(path, triplet, *, expected_hash, config_hash, upstream, block):
    if file_hash(path) != expected_hash:
        raise ValueError("stage file hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_stage(payload, triplet, config_hash=config_hash, upstream=upstream, block=block)


class CompactSwiGLU(nn.Module):
    """Dense execution with the retained intermediate width only."""
    def __init__(self, gate, up, down):
        super().__init__()
        if gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError("compact SwiGLU shapes do not agree")
        h, d = gate.shape
        self.gate_proj = nn.Linear(d, h, bias=False, device=gate.device, dtype=gate.dtype)
        self.up_proj = nn.Linear(d, h, bias=False, device=gate.device, dtype=gate.dtype)
        self.down_proj = nn.Linear(h, d, bias=False, device=gate.device, dtype=gate.dtype)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate)
            self.up_proj.weight.copy_(up)
            self.down_proj.weight.copy_(down)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


def compact_mlp(candidate):
    keep = candidate.keep.to(candidate.gate.device)
    return CompactSwiGLU(candidate.gate[keep], candidate.up[keep], candidate.down[:, keep])


def export_compact(model, output, kept_by_block, *, provenance=None):
    """Consume an already validated/applied stage chain, without calibration.

    Mutates the supplied diagnostic model into compact modules. Saves all
    remaining model weights, including attention, so loading needs no base
    checkpoint. HF config constructs the skeleton only; widths come from
    the explicit per-layer export manifest, not intermediate_size.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        manifest = json.loads((output/"manifest.json").read_text())
        if manifest["provenance"] != (provenance or {}) or file_hash(output/"model.pt") != manifest["weights_sha256"]:
            raise ValueError("existing export differs or is corrupt")
        return manifest
    widths = {}
    for block, keep in kept_by_block.items():
        mlp = model.model.layers[int(block)].mlp
        triplet = SwiGLUTriplet(mlp)
        g, u, d = triplet.weights
        keep = keep.to(device=g.device, dtype=torch.bool)
        if keep.shape != (len(g),) or not keep.any() or g[~keep].any() or u[~keep].any() or d[:, ~keep].any():
            raise ValueError("compact export requires an applied validated triplet mask")
        model.model.layers[int(block)].mlp = CompactSwiGLU(g[keep], u[keep], d[:, keep])
        widths[str(block)] = {"original_width": len(g), "width": int(keep.sum()),
                              "kept_indices": torch.where(keep)[0].tolist()}
    temporary = output/"model.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(output/"model.pt")
    config = model.config.to_dict() if hasattr(model, "config") else None
    manifest = {"schema": 1, "method_version": VERSION, "widths": widths,
                "model_config": config, "provenance": provenance or {},
                "weights_sha256": file_hash(output/"model.pt"),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "tensor_file_bytes": (output/"model.pt").stat().st_size}
    temporary_manifest = output/"manifest.tmp"
    temporary_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    temporary_manifest.replace(output/"manifest.json")
    return manifest


def load_compact(output, *, model_factory=None, device="cpu"):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest["method_version"] != VERSION:
        raise ValueError("unsupported compact artifact schema")
    if file_hash(output/"model.pt") != manifest["weights_sha256"]:
        raise ValueError("compact artifact hash mismatch")
    if model_factory is None:
        from transformers import AutoConfig, AutoModelForCausalLM
        config = dict(manifest["model_config"])
        model_type = config.pop("model_type")
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **config))
    else:
        model = model_factory()
    state = torch.load(output/"model.pt", map_location="cpu", weights_only=True)
    floating = next(v for v in state.values() if v.is_floating_point())
    model = model.to(dtype=floating.dtype)
    for block, info in manifest["widths"].items():
        h, indices = info["width"], info["kept_indices"]
        if len(indices) != h or sorted(set(indices)) != indices or not h:
            raise ValueError("invalid compact neuron index manifest")
        prefix = f"model.layers.{block}.mlp."
        g, u, d = (state[prefix+name+".weight"] for name in ("gate_proj", "up_proj", "down_proj"))
        if len(g) != h or indices[-1] >= info["original_width"]:
            raise ValueError("compact width/index mismatch")
        model.model.layers[int(block)].mlp = CompactSwiGLU(g, u, d)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), manifest


def benchmark_compact(model, input_ids, *, repeats=5, decode_tokens=32):
    """Benchmark the actual loaded representation; no pruning or weight edits.

    HF causal-LM cache interface, independent fresh prefill for each repeat.
    GPU callers must run this measurement through the guarded scheduler.
    """
    import time
    if repeats < 1 or decode_tokens < 1:
        raise ValueError("positive repeat/decode counts required")
    def sync():
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
    prefill, decode = [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            output = model(input_ids=input_ids, use_cache=True)
            sync()
            prefill.append(time.perf_counter()-start)
            cache = output.past_key_values
            token = output.logits[:, -1:].argmax(-1)
            sync()
            start = time.perf_counter()
            for _ in range(decode_tokens):
                output = model(input_ids=token, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                token = output.logits[:, -1:].argmax(-1)
            sync()
            decode.append(time.perf_counter()-start)
    return {"prefill_seconds": prefill, "decode_seconds": decode,
            "decode_tokens_per_repeat": decode_tokens, "prompt_tokens": input_ids.shape[1],
            "batch_size": input_ids.shape[0], "device": str(input_ids.device),
            "dtype": str(next(model.parameters()).dtype), "warmup": "none; report first and subsequent repeats separately"}
