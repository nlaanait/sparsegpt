#!/usr/bin/env bash
# GH200 smoke: one-layer Qwen3.5-2B ambient vs tangent pair via gflow.
# Small nsamples, skip eval — confirms CUDA + CLI + tangent path end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

NAME="${1:-qwen-2b-tangent-c4-smoke}"
OUT="results/tangent/qwen3.5-2b-c4-smoke"

gbatch \
  --gpus 1 \
  --gpu-memory 24G \
  --memory 48G \
  --name "$NAME" \
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
    --tangent-alpha 0.5 \
    --tangent-charts 4 \
    --tangent-rank 2 \
    --tangent-samples 64 \
    --output-dir "$OUT"

echo
gqueue
echo "logs: gjob log <id>   report: $OUT/report.json"
