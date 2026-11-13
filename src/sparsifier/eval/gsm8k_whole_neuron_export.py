"""Export a completed triplet chain without another calibration pass.

Example: python -m sparsifier.eval.gsm8k_whole_neuron_export --source RUN --output ARTIFACT
This loads the declared base checkpoint on CPU and saves a self-contained
mixed-width model state plus an explicit width/index manifest. It runs no
GPU work and performs no calibration or recovery training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from sparsifier.eval.gsm8k import MODEL, REVISION
from sparsifier.eval.gsm8k_whole_neuron import restore_chain
from sparsifier.prune.whole_neuron_io import export_compact, file_hash


def main():
    from transformers import AutoModelForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.source/"config.json").read_text())
    if config["model"] != MODEL or config["revision"] != REVISION:
        raise ValueError("whole-neuron export base checkpoint mismatch")
    provenance = {"source_config_sha256": file_hash(args.source/"config.json"),
                  "source_report_sha256": file_hash(args.source/"report.json"),
                  "model": MODEL, "revision": REVISION}
    if (args.output/"manifest.json").exists():
        manifest = json.loads((args.output/"manifest.json").read_text())
        if manifest["provenance"] != provenance or file_hash(args.output/"model.pt") != manifest["weights_sha256"]:
            raise ValueError("existing compact export differs or is corrupt")
        print(json.dumps(manifest, indent=2))
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, use_kernels=False, attn_implementation="sdpa").eval()
    _, _, kept = restore_chain(model, args.source)
    manifest = export_compact(model, args.output, kept, provenance=provenance)
    print(json.dumps({"parameter_count": manifest["parameter_count"],
                      "artifact_total_bytes": sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()),
                      "latency": "not measured; benchmark the loaded compact artifact"}, indent=2))


if __name__ == "__main__":
    main()
