#!/usr/bin/env bash
# GH200 scheduling pattern:
#   1) fork causal-conv1d wheel build on gflow (shared 2G)
#   2) immediately queue a tangent smoke that uses --pytorch-attention
#      (no wait — PyTorch fallback, correct but slower)
#   3) queue the full matched pair after the wheel is ready, installing it
#      at job start via ensure_causal_conv1d.sh
#
# Usage:
#   ./scripts/schedule_tangent_gh200.sh
#   SKIP_FULL=1 ./scripts/schedule_tangent_gh200.sh   # build + smoke only
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

parse_id() {
  echo "$1" | grep -oE '[0-9]+' | head -1
}

echo "=== 1/3 causal-conv1d build (background) ==="
build_out="$(bash scripts/submit_causal_conv1d_build_gflow.sh)"
echo "$build_out"
BUILD_ID="$(echo "$build_out" | sed -n 's/^BUILD_JOB_ID=//p' | tail -1)"
[[ -n "$BUILD_ID" ]] || { echo "failed to get BUILD_JOB_ID" >&2; exit 1; }

SMOKE_OUT="results/tangent/qwen3.5-2b-c4-smoke-pytorch"
echo "=== 2/3 tangent smoke now (pytorch attention, no depends-on) ==="
smoke_out="$(
  gbatch \
    --gpus 1 \
    --gpu-memory 24G \
    --memory 48G \
    --name qwen-2b-tangent-c4-smoke-pytorch \
    --project tangent-sparsegpt \
    --time 2:00:00 \
    --auto-close \
    -- \
    pixi run python -u -m experiments.tangent_pruning_pair \
      --model Qwen/Qwen3.5-2B \
      --calibration c4 \
      --nsamples 8 \
      --sparsity 0.5 \
      --maxlayer 1 \
      --skip-eval \
      --pytorch-attention \
      --tangent-alpha 0.5 \
      --tangent-charts 4 \
      --tangent-rank 2 \
      --tangent-samples 64 \
      --output-dir "$SMOKE_OUT"
)"
echo "$smoke_out"
SMOKE_ID="$(parse_id "$smoke_out")"

if [[ "${SKIP_FULL:-0}" == "1" ]]; then
  echo
  gqueue
  echo "BUILD_JOB_ID=$BUILD_ID SMOKE_JOB_ID=$SMOKE_ID (full pair skipped)"
  exit 0
fi

FULL_OUT="results/tangent/qwen3.5-2b-c4-sp50"
echo "=== 3/3 full tangent pair after wheel (depends-on $BUILD_ID) ==="
full_out="$(
  gbatch \
    --gpus 1 \
    --gpu-memory 48G \
    --memory 64G \
    --name qwen-2b-tangent-c4-sp50 \
    --project tangent-sparsegpt \
    --time 8:00:00 \
    --auto-close \
    --depends-on "$BUILD_ID" \
    -- \
    bash -lc "
      set -euo pipefail
      ./scripts/ensure_causal_conv1d.sh
      pixi run python -u -m experiments.tangent_pruning_pair \
        --model Qwen/Qwen3.5-2B \
        --calibration c4 \
        --nsamples 128 \
        --sparsity 0.5 \
        --eval-dataset wikitext2 \
        --tangent-alpha 0.5 \
        --tangent-charts 8 \
        --tangent-rank 2 \
        --tangent-samples 512 \
        --output-dir '$FULL_OUT'
    "
)"
echo "$full_out"
FULL_ID="$(parse_id "$full_out")"

echo
gqueue
echo "BUILD_JOB_ID=$BUILD_ID  SMOKE_JOB_ID=$SMOKE_ID  FULL_JOB_ID=$FULL_ID"
echo "smoke report: $SMOKE_OUT/report.json"
echo "full report:  $FULL_OUT/report.json"
