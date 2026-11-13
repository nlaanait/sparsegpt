#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SOURCE="${1:?source run name required}"
TOKENS="${2:-2048}"
[[ "$SOURCE" =~ ^qwen-9b-gsm8k-[a-z0-9-]+$ && "$TOKENS" =~ ^[1-9][0-9]*$ ]] || exit 2
NAME="${SOURCE}-budget${TOKENS}"
OUT="$ROOT/results/tangent/gsm8k/$NAME"
gbatch --gpus 1 --gpu-memory 64G --memory 80G \
  --name "$NAME" --project qwen-gsm8k --time 24:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_reevaluate \
  --source "$ROOT/results/tangent/gsm8k/$SOURCE" \
  --capture "$ROOT/results/tangent/gsm8k/capture-v1" \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --output "$OUT" --max-new-tokens "$TOKENS"
