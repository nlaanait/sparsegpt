"""Select a batch policy for one worker before importing experiment drivers.

New workers default to batch 32. The v1 source module stays unchanged so active
workers retain their source hashes and resume policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys


def configure_batch_size(size):
    if type(size) is not int or size not in (8, 16, 32):
        raise ValueError("batch size must be 8, 16, or 32")
    loaded = [name for name in sys.modules if name.startswith("sparsifier.eval.gsm8k")
              and name not in (__name__, "sparsifier.eval.gsm8k_batch_runtime",
                               "sparsifier.eval.gsm8k_batching", "sparsifier.eval.gsm8k_batch_defaults")]
    if loaded:
        raise RuntimeError("configure batching before importing GSM8K drivers")
    from sparsifier.eval.gsm8k_batch_defaults import configure
    configure(size)


def reference_command(size, module, arguments, manifest):
    if module != "sparsifier.eval.gsm8k_decision_full":
        raise ValueError("reference preparation is only supported for full decision evaluation")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--reference", required=True)
    args, _ = parser.parse_known_args(arguments)
    return [sys.executable, "-u", "-m", "sparsifier.eval.gsm8k_batch_runtime",
            "--batch-size", str(size), "sparsifier.eval.gsm8k_decision",
            "--manifest", str(manifest), "--calibration-manifest", args.calibration_manifest,
            "--output", args.reference]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from sparsifier.eval.gsm8k_batch_defaults import DEFAULT_BATCH_SIZE
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--prepare-reference-manifest", type=Path)
    parser.add_argument("module", choices=("sparsifier.eval.gsm8k_decision",
                                         "sparsifier.eval.gsm8k_decision_full",
                                         "sparsifier.eval.gsm8k_downstream",
                                         "sparsifier.eval.gsm8k_sequential",
                                         "sparsifier.eval.gsm8k_whole_neuron",
                                         "sparsifier.eval.gsm8k_decision_pilot"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configure_batch_size(args.batch_size)
    from sparsifier.eval.gsm8k_batching import policy
    print(json.dumps({"event": "batching_policy", "batch_size": args.batch_size,
                      "batching": policy()}), flush=True)
    if args.prepare_reference_manifest:
        # Separate workers release the dense checkpoint before loading the
        # candidate. Both remain in the existing memory guard's process group.
        subprocess.run(reference_command(args.batch_size, args.module, args.arguments,
                                         args.prepare_reference_manifest), check=True)
    sys.argv = [args.module, *args.arguments]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
