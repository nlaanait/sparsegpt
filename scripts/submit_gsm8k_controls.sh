#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ARM="${1:?fisher or whole_chart}"
DEPENDENCY="${2:?dependency job ID required}"
[[ "$ARM" == fisher || "$ARM" == whole_chart ]] || exit 2
[[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || exit 2
NAME="qwen-9b-gsm8k-${ARM//_/-}-a1-s0-v1"
OUT="$ROOT/results/tangent/gsm8k/$NAME"
gbatch --gpus 1 --gpu-memory 64G --memory 80G --depends-on "$DEPENDENCY" \
  --name "$NAME" --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_controls \
  --capture "$ROOT/results/tangent/gsm8k/capture-v1" \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --sensitivity "$ROOT/results/tangent/gsm8k/sensitivity-v1" \
  --output "$OUT" --arm "$ARM"
