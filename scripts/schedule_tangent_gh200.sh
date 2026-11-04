#!/usr/bin/env bash
# GH200: mid-size Qwen3.5-9B sparsity sweep — unstructured SparseGPT vs
# output-tangent, recording likelihood (nats) and perplexity on WikiText2.
#
# Starts immediately with --pytorch-attention (no wait on causal-conv1d).
# BUILD_FIRST=1 waits on a wheel build instead and runs with CUDA kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

BUILD_FIRST="${BUILD_FIRST:-0}"
BUILD_ID=""

if [[ "$BUILD_FIRST" == "1" ]]; then
  echo "=== causal-conv1d build first ==="
  build_out="$(bash scripts/submit_causal_conv1d_build_gflow.sh)"
  echo "$build_out"
  BUILD_ID="$(echo "$build_out" | sed -n 's/^BUILD_JOB_ID=//p' | tail -1)"
  [[ -n "$BUILD_ID" ]] || { echo "failed to get BUILD_JOB_ID" >&2; exit 1; }
  DEPENDS_ON="$BUILD_ID" PYTORCH_ATTENTION=0 \
    bash scripts/submit_tangent_sparsity_sweep_gflow.sh
else
  echo "=== Qwen3.5-9B unstructured vs tangent sparsity sweep (pytorch attention) ==="
  bash scripts/submit_tangent_sparsity_sweep_gflow.sh
fi

gqueue
echo "report: results/tangent/qwen3.5-9b-c4-sparsity-sweep/sweep_report.json"
