#!/usr/bin/env bash
# Queue a background causal-conv1d wheel build on gflow.
#
# Compile is CPU/nvcc heavy; request a tiny shared GPU slice so CUDA_HOME /
# device discovery works without blocking exclusive SparseGPT jobs.
#
# Prints: BUILD_JOB_ID=<n> for --depends-on chaining.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

NAME="${1:-causal-conv1d-cu130-aarch64-build}"

out="$(
  gbatch \
    --gpus 1 \
    --shared \
    --gpu-memory 2G \
    --memory 32G \
    --name "$NAME" \
    --project tangent-sparsegpt \
    --time 2:00:00 \
    --auto-close \
    -- \
    bash scripts/build_causal_conv1d_wheel.sh
)"
echo "$out"
id="$(echo "$out" | grep -oE '[0-9]+' | head -1 || true)"
if [[ -z "$id" ]]; then
  echo "ERROR: could not parse gbatch job id from: $out" >&2
  exit 1
fi
echo "BUILD_JOB_ID=$id"
echo "chain Qwen/kernel jobs with: --depends-on $id"
echo "meanwhile schedule fallback jobs without depends-on (or with --pytorch-attention)"
