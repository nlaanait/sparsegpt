#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NAME="${1:?unique experiment name}"
SOURCE="${2:-}"
[[ "$NAME" =~ ^qwen-9b-gsm8k-[a-z0-9-]+$ ]] || exit 2
EXTRA=()
if [[ -n "$SOURCE" ]]; then
  [[ "$SOURCE" =~ ^qwen-9b-gsm8k-[a-z0-9-]+$ ]] || exit 2
  EXTRA=(--source "$ROOT/results/tangent/gsm8k/$SOURCE")
fi
OUT="$ROOT/results/tangent/gsm8k/$NAME"
gbatch --gpus 1 --gpu-memory 64G --memory 80G --name "$NAME" \
  --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_original \
  --manifest "$ROOT/results/tangent/gsm8k/data/original-validation-v1.json" \
  --output "$OUT" --max-new-tokens 2048 "${EXTRA[@]}"
