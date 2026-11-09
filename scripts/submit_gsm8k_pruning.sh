#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ARM="${1:?arm required}"
ALPHA="${2:-1}"
DEPENDENCY="${3:-}"
SEED="${4:-0}"
[[ "$ARM" =~ ^[a-z][a-z0-9_]+$ && "$ALPHA" =~ ^[0-9]+([.][0-9]+)?$ && "$SEED" =~ ^[0-9]+$ ]] || exit 2
NAME="qwen-9b-gsm8k-${ARM//_/-}-a${ALPHA//./p}-s${SEED}-v1"
OUT="$ROOT/results/tangent/gsm8k/$NAME"
EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
  [[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || exit 2
  EXTRA=(--depends-on "$DEPENDENCY")
fi
gbatch --gpus 1 --gpu-memory 64G --memory 80G "${EXTRA[@]}" \
  --name "$NAME" --project qwen-gsm8k --time 12:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k_pruning \
  --capture "$ROOT/results/tangent/gsm8k/capture-v1" \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --atlas "$ROOT/results/tangent/gsm8k/geometry-seed0-v2/atlas.npz" \
  --output "$OUT" --arm "$ARM" --alpha "$ALPHA" --seed "$SEED" --eval-count 32
