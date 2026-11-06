#!/usr/bin/env bash
# Memory-gated single-layer Qwen3.5-0.8B smokes for each tangent technique.
# Refuses to start if host used memory already exceeds 90 GiB.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pixi run python - <<'PY'
from sparsifier.prune.memory_guard import ensure_host_memory_headroom, format_snapshot
state = ensure_host_memory_headroom(max_used_gib=90.0, min_available_gib=20.0)
print("memory_guard_ok", format_snapshot(state), flush=True)
PY

PRUNERS=(
  wanda
  tangent
  tangent_factor_union
  tangent_chart_obs
  tangent_fixed_union
  tangent_knockout
  tangent_editability
  tangent_exact_obs
)

for pruner in "${PRUNERS[@]}"; do
  pixi run python - <<'PY'
from sparsifier.prune.memory_guard import ensure_host_memory_headroom, format_snapshot
print(format_snapshot(ensure_host_memory_headroom(max_used_gib=90.0, min_available_gib=20.0)), flush=True)
PY
  echo "=== smoke pruner=${pruner} ==="
  pixi run python -u -m sparsifier Qwen/Qwen3.5-0.8B c4 \
    --seed 0 \
    --nsamples 8 \
    --sparsity 0.5 \
    --minlayer 0 \
    --maxlayer 1 \
    --blocksize 128 \
    --forward-batch 1 \
    --pytorch-attention \
    --skip-eval \
    --tangent-charts 4 \
    --tangent-rank 1 \
    --tangent-samples 64 \
    --tangent-factors 4 \
    --pruner "${pruner}"
done

# Note-faithful factor estimator (same pruner, BlockDictionary active sets).
pixi run python - <<'PY'
from sparsifier.prune.memory_guard import ensure_host_memory_headroom, format_snapshot
print(format_snapshot(ensure_host_memory_headroom(max_used_gib=90.0, min_available_gib=20.0)), flush=True)
PY
echo "=== smoke pruner=tangent_factor_union factor_source=block_dictionary ==="
pixi run python -u -m sparsifier Qwen/Qwen3.5-0.8B c4 \
  --seed 0 \
  --nsamples 8 \
  --sparsity 0.5 \
  --minlayer 0 \
  --maxlayer 1 \
  --blocksize 128 \
  --forward-batch 1 \
  --pytorch-attention \
  --skip-eval \
  --tangent-rank 1 \
  --tangent-samples 64 \
  --tangent-factors 4 \
  --tangent-active-factors 2 \
  --tangent-block-width 3 \
  --tangent-factor-source block_dictionary \
  --pruner tangent_factor_union

echo "all technique smokes finished"
