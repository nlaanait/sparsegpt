#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PHASE="${1:?sensitivity or interventions required}"
DEPENDENCY="${2:-}"
EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
  [[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || exit 2
  EXTRA=(--depends-on "$DEPENDENCY")
fi
case "$PHASE" in
  sensitivity)
    MODULE=sparsifier.eval.gsm8k_sensitivity
    OUT="$ROOT/results/tangent/gsm8k/sensitivity-v1"
    ARGS=(--count 32)
    ;;
  interventions)
    MODULE=sparsifier.eval.gsm8k_interventions
    OUT="$ROOT/results/tangent/gsm8k/stage2-v1"
    ARGS=(--capture "$ROOT/results/tangent/gsm8k/capture-v1"
          --atlas "$ROOT/results/tangent/gsm8k/geometry-seed0-v2/atlas.npz"
          --sensitivity "$ROOT/results/tangent/gsm8k/sensitivity-v1"
          --singles 128 --group-pairs 8 --group-size 8 --eval-count 8)
    ;;
  *) exit 2 ;;
esac
NAME="qwen-9b-gsm8k-${PHASE}-v1"
gbatch --gpus 1 --gpu-memory 64G --memory 80G "${EXTRA[@]}" \
  --name "$NAME" --project qwen-gsm8k --time 24:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m "$MODULE" \
  --manifest "$ROOT/results/tangent/gsm8k/data/controlled-v1.json" \
  --output "$OUT" "${ARGS[@]}"
