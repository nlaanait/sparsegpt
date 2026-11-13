#!/usr/bin/env bash
# Exclusive GPU allocation plus unified-memory containment; no shared launches.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NAME="${1:-qwen-9b-gsm8k-dense-pilot}"
LIMIT="${2:-4}"
DEPENDENCY="${3:-}"
if [[ ! "$NAME" =~ ^[a-z][a-z0-9-]+$ || ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 lowercase-job-name nonnegative-example-limit" >&2
  exit 2
fi
OUT="$ROOT/results/tangent/gsm8k/$NAME"
DEPENDENCY_ARGS=()
if [[ -n "$DEPENDENCY" ]]; then
  [[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || exit 2
  DEPENDENCY_ARGS=(--depends-on "$DEPENDENCY")
fi
gflowd status
ginfo
# 80 GiB cgroup memory ceiling + 56 GiB CUDA allocator ceiling (overlapping
# unified-memory accounting, not additive). Watchdog stops at 96 GiB host use.
gbatch --gpus 1 --gpu-memory 64G --memory 80G "${DEPENDENCY_ARGS[@]}" \
  --name "$NAME" --project qwen-gsm8k --time 24:00:00 --auto-close -- \
  bash "$ROOT/scripts/run_gsm8k_guarded.sh" "$NAME" "$OUT/memory.jsonl" \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.gsm8k \
  --manifest "$ROOT/results/tangent/gsm8k/data/development.json" \
  --output "$OUT" --limit "$LIMIT" --max-new-tokens 512 --gpu-cap-gib 56
