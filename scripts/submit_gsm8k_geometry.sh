#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SEED="${1:-0}"
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
NAME="qwen-9b-gsm8k-geometry-seed${SEED}-v2"
OUT="$ROOT/results/tangent/gsm8k/geometry-seed${SEED}-v2"
# CPU work is still scheduled exclusively on this unified-memory host.
gbatch --gpus 1 --gpu-memory 64G --memory 80G \
  --name "$NAME" --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_geometry \
  --capture "$ROOT/results/tangent/gsm8k/capture-v1" \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --output "$OUT" --seed "$SEED"
