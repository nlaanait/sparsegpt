#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NAME="${1:?unique experiment name}"
ARM="${2:?arm}"
SPARSITY="${3:-0.5}"
BLOCKS="${4:-all}"
MODULES="${5:-mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
COUNT="${6:-32}"
[[ "$NAME" =~ ^qwen-9b-gsm8k-[a-z0-9-]+$ ]] || exit 2
OUT="$ROOT/results/tangent/gsm8k/$NAME"
gbatch --gpus 1 --gpu-memory 64G --memory 80G --name "$NAME" \
  --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_sequential \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --pilot-report "$ROOT/results/tangent/gsm8k/qwen-9b-gsm8k-dense-pilot-v2/report.json" \
  --c4-capture "$ROOT/results/tangent/gsm8k/capture-v1" \
  --output "$OUT" --arm "$ARM" --sparsity "$SPARSITY" --blocks "$BLOCKS" \
  --modules "$MODULES" --eval-count "$COUNT" --max-new-tokens 2048
