#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEPENDENCY="${1:?pass the dense pilot gflow job ID}"
[[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || exit 2
NAME="qwen-9b-gsm8k-capture-v1"
OUT="$ROOT/results/tangent/gsm8k/capture-v1"
gbatch --gpus 1 --gpu-memory 64G --memory 80G --depends-on "$DEPENDENCY" \
  --name "$NAME" --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_capture \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --pilot-report "$ROOT/results/tangent/gsm8k/qwen-9b-gsm8k-dense-pilot-v2/report.json" \
  --output "$OUT"
