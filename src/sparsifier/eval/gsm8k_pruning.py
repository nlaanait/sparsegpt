"""Development one-layer pruning arms with matched data and real behavior.

Each arm is one resumable guarded gflow job. This is a validation experiment,
not a final GSM8K-test or whole-model compression result. Output geometry uses
the explicitly approximate row-diagonal congruence solver.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable
from sparsifier.eval.gsm8k import MODEL, REVISION, atomic_json, read_completed, render_prompt
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.memory_guard import configure_cuda, host_memory
from sparsifier.prune.factor_atlas import FactorAtlas, projector_diagonals
from sparsifier.prune.input_side import InputSideSparseGPT
from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION, local_row_scales
from sparsifier.prune.sparsegpt import SparseGPT

ARMS = ("dense", "sparsegpt_c4", "sparsegpt_task", "input_side", "input_repaired",
        "input_rotated", "local", "factor", "rank_matched", "global", "shuffled")


def calibration(capture_dir, index, x, *, seed=0, math_tokens=4096):
    """Include all fit/calibration readouts, then sample other task tokens."""
    use = [i for i, r in enumerate(index["rows"]) if r["split"] in ("geometry_fit", "pruning_calibration")]
    if len(use) > math_tokens:
        raise ValueError("math budget cannot include all pair-endpoint readouts")
    pool = []
    for i in use:
        with np.load(capture_dir / index["rows"][i]["file"], allow_pickle=False) as saved:
            pool.append(saved["token_x"][saved["positions"] != int(saved["readout_position"])])
    pool = np.concatenate(pool)
    needed = math_tokens - len(use)
    if len(pool) < needed:
        raise ValueError("insufficient distinct math token samples")
    selected = np.random.default_rng(seed).choice(len(pool), needed, replace=False)
    math = np.concatenate((x[use], pool[selected])).astype(np.float32)
    with np.load(capture_dir / "c4.npz", allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != index["config_sha256"]:
            raise ValueError("C4 provenance mismatch")
        c4 = saved["inputs"].copy()
    return torch.from_numpy(np.concatenate((c4, math))), torch.from_numpy(c4), use


def pair_inputs(manifest, by_id, x, *, mode="input_side", seed=0):
    pairs = [p for p in manifest["pairs"] if p["split"] == "geometry_fit"
             and p["left"] in by_id and p["right"] in by_id]
    names = sorted({p["factor"] for p in pairs})
    if not pairs:
        raise ValueError("no fitting pairs")
    labels = np.array([names.index(p["factor"]) for p in pairs])
    left = np.array([by_id[p["left"]] for p in pairs])
    right = np.array([by_id[p["right"]] for p in pairs])
    rng = np.random.default_rng(seed)
    if mode == "input_repaired":
        # Cyclic derangement within each knob; signs/row order alone do nothing.
        for label in np.unique(labels):
            ix = np.flatnonzero(labels == label)
            if len(ix) > 1:
                order = rng.permutation(ix)
                right[order] = np.roll(right[order].copy(), 1)
    delta = x[right] - x[left]
    if mode == "input_rotated":
        delta = delta[:, rng.permutation(delta.shape[1])] * rng.choice([-1., 1.], delta.shape[1])
    return torch.tensor(delta, dtype=torch.float32), torch.tensor(labels)


def select_validation(rows, count, seed):
    """Round-robin family-balanced selection without seeing any outcomes."""
    groups = {}
    for row in rows:
        if row["split"] == "validation":
            groups.setdefault(row["family"], []).append(row)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while groups and len(selected) < count:
        for group in sorted(list(groups)):
            selected.append(groups[group].pop())
            if not groups[group]:
                del groups[group]
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError("insufficient validation problems")
    return selected


def prune_candidate(weight, inputs, *, arm, sparsity=.5, alpha=1., deltas=None,
                    labels=None, readout_inputs=None, pdiag=None, device="cuda"):
    """Same SparseGPT compensation settings for all applicable arms."""
    if arm == "dense":
        return weight.clone(), {"solver": "dense"}
    w = weight.to(device=device, dtype=torch.float32)
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False, device=device, dtype=torch.float32)
    scale = None
    diagnostics = {"solver": "ambient_obs", "damping": .01, "blocksize": 128}
    if pdiag is not None:
        scale, local_diagnostics = local_row_scales(
            inputs.to(device), readout_inputs, pdiag, alpha)
        if scale.shape != w.shape:
            raise ValueError("projector output dimension must match weights")
        diagnostics.update(local_diagnostics, solver="row_diagonal_congruence_approximation")
    with torch.no_grad():
        layer.weight.copy_(w if scale is None else w * scale)
    solver = InputSideSparseGPT(layer) if arm.startswith("input_") else SparseGPT(layer)
    solver.add_batch(inputs.to(device))
    if isinstance(solver, InputSideSparseGPT):
        solver.add_pairs(deltas, labels)
        solver.finalize_pairs(alpha)
        diagnostics["solver"] = "input_augmented_obs"
    solver.fasterprune(sparsity, blocksize=128, percdamp=.01, mask_weight=torch.ones_like(w))
    result = layer.weight.detach()
    if scale is not None:
        result = result / scale
    result = result.cpu().clone()
    solver.free()
    return result, diagnostics


def answer_likelihood(model, tokenizer, question, answer):
    """Direct-answer teacher forcing, separate from generated reasoning accuracy."""
    device = next(model.parameters()).device
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Give only the numeric answer.\n\n" + question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(" " + answer, add_special_tokens=False)
    ids = torch.tensor([prefix + target], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target) + 1).logits
        predicted = logits[0, -len(target)-1:-1].float()
        nll = F.cross_entropy(predicted, torch.tensor(target, device=device), reduction="mean")
    return float(nll)


def behavior(model, tokenizer, rows, output, config_hash, *, max_new_tokens=512):
    path = output / "answers.jsonl"
    done = read_completed(path, config_hash)
    if set(done) - {r["id"] for r in rows}:
        raise ValueError("unexpected result IDs")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    device = next(model.parameters()).device
    with path.open("a", buffering=1) as handle:
        for row in rows:
            if row["id"] in done:
                continue
            start = time.monotonic()
            prompt = render_prompt(tokenizer, row["question"])
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                result = model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens,
                                        use_cache=True, pad_token_id=tokenizer.eos_token_id)
            generated = result[0, ids.input_ids.shape[1]:].tolist()
            truncated = len(generated) >= max_new_tokens and generated[-1] not in eos
            text = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_answer(text, truncated=truncated)
            record = {"id": row["id"], "family": row["family"], "config_sha256": config_hash,
                      "text": text, "generated_tokens": len(generated), "truncated": truncated,
                      "parsed_answer": str(parsed) if parsed is not None else None,
                      "expected_answer": row["answer"],
                      "correct": parsed is not None and parsed == Fraction(row["answer"]),
                      "answer_nll": answer_likelihood(model, tokenizer, row["question"], row["answer"]),
                      "seconds": time.monotonic() - start}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done[row["id"]] = record
            print(json.dumps({"evaluated": len(done), "total": len(rows),
                              "correct": sum(r["correct"] for r in done.values())}), flush=True)
    records = list(done.values())
    return {"count": len(records), "accuracy": float(np.mean([r["correct"] for r in records])),
            "answer_nll": float(np.mean([r["answer_nll"] for r in records])),
            "truncated": sum(r["truncated"] for r in records)}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsifier.models.qwen import force_pytorch_attention
    from sparsifier.eval.natural_layer import replaced_weight, vocabulary_reference, language_metrics
    from sparsifier.data.loaders import get_loaders
    from sparsifier.prune import rescale

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--sparsity", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.sparsity < 1 or not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("invalid sparsity/alpha")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    capture, index, by_id, x, y = load_capture(args.capture)
    manifest = load_manifest(args.manifest)
    if manifest["manifest_sha256"] != capture["manifest_sha256"]:
        raise ValueError("capture/manifest mismatch")
    rows = select_validation(manifest["rows"], args.eval_count, seed=0)
    geometry_arm = args.arm in ("local", "factor", "rank_matched", "global", "shuffled")
    if geometry_arm and args.atlas is None:
        parser.error("geometry arm requires an atlas")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "capture_sha256": digest(capture),
              "manifest_sha256": manifest["manifest_sha256"], "arm": args.arm,
              "alpha": args.alpha, "sparsity": args.sparsity, "seed": args.seed,
              "local_pruning_version": LOCAL_PRUNING_VERSION,
              "rescale_source_sha256": hashlib.sha256(Path(rescale.__file__).read_bytes()).hexdigest(),
              "evaluation_ids": [r["id"] for r in rows], "evaluation_split": "validation",
              "max_new_tokens": 512, "purpose": "development_single_layer_not_confirmatory",
              "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest() if args.atlas else None,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_immutable(args.output / "config.json", config)
    config_hash = digest(config)
    report_path = args.output / "report.json"
    if report_path.exists() and json.loads(report_path.read_text()).get("complete"):
        print("Already complete", flush=True)
        return
    memory = configure_cuda(56.)
    inputs, c4, use = calibration(args.capture, index, x, seed=args.seed)
    if args.arm == "sparsegpt_c4":
        inputs = c4
    deltas = labels = None
    if args.arm.startswith("input_"):
        deltas, labels = pair_inputs(manifest, by_id, x, mode=args.arm, seed=args.seed)
    pdiag = None
    if geometry_arm:
        atlas = FactorAtlas.load(args.atlas)
        frames, _, _ = atlas.frames(y[use], mode=args.arm, seed=args.seed)
        atlas_diagnostics = atlas.diagnostics(y[use])
        pdiag = projector_diagonals(frames)
        del frames
    weight = torch.from_numpy(np.load(args.capture / "weight.npy"))
    candidate_path = args.output / "candidate.pt"
    if candidate_path.exists():
        stored = torch.load(candidate_path, weights_only=True, map_location="cpu")
        if stored["config_sha256"] != config_hash:
            raise ValueError("candidate provenance mismatch")
        candidate, diagnostics = stored["weight"], stored["diagnostics"]
    else:
        candidate, diagnostics = prune_candidate(weight, inputs, arm=args.arm, sparsity=args.sparsity,
            alpha=args.alpha, deltas=deltas, labels=labels, readout_inputs=torch.from_numpy(x[use]), pdiag=pdiag)
        if geometry_arm:
            diagnostics["atlas"] = atlas_diagnostics
        candidate = candidate.to(torch.bfloat16).float()
        temporary = candidate_path.with_suffix(".tmp")
        torch.save({"weight": candidate, "diagnostics": diagnostics, "config_sha256": config_hash}, temporary)
        temporary.replace(candidate_path)
    torch.cuda.empty_cache()
    test_ids = [by_id[r["id"]] for r in rows]
    residual = torch.from_numpy(x[test_ids]) @ (candidate - weight).T
    local = {"ambient_readout_mse": float(residual.square().sum(1).mean()),
             "actual_sparsity": float((candidate == 0).float().mean()),
             "calibration_tokens": len(inputs), "solver": diagnostics}
    del inputs, c4, pdiag, deltas, labels, residual
    force_pytorch_attention()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    layer = model.model.layers[capture["block"]].get_submodule(capture["module"])
    if not torch.equal(layer.weight.detach().float().cpu(), weight):
        raise ValueError("dense checkpoint does not match captured weight")
    # Validation language windows, reserved separately from final evaluation.
    _, wiki = get_loaders("wikitext2", 1, 0, 128, MODEL)
    windows = [wiki.input_ids[:, i*128:(i+1)*128] for i in range(8)]
    reference = vocabulary_reference(model, windows, "cuda")
    dense_nll = float(np.mean([r["nll"] for r in reference]))
    if not np.isfinite(dense_nll) or dense_nll > np.log(100):
        raise RuntimeError("implausible dense perplexity; refuse pruning evaluation")
    with replaced_weight(layer, candidate):
        lm = language_metrics(model, windows, reference, "cuda")
        evaluated = behavior(model, tokenizer, rows, args.output, config_hash)
    atomic_json(report_path, {"complete": True, "config_sha256": config_hash,
        "behavior": evaluated, "language": lm, "local": local, "dense_nll": dense_nll,
        "memory": memory, "host_after": host_memory(),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})


if __name__ == "__main__":
    main()
