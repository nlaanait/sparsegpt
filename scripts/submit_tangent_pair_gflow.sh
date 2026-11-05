#!/usr/bin/env bash
# Matched ambient SparseGPT vs output-tangent pair on Qwen3.5-2B (WikiText2).
# Caps sized for GH200 (96G HBM); exclusive --gpus 1 serializes with other jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

NAME="${1:-qwen-2b-tangent-c4-sp50}"
OUT="results/tangent/qwen3.5-2b-c4-sp50"

gbatch \
  --gpus 1 \
  --gpu-memory 48G \
  --memory 64G \
  --name "$NAME" \
  --project tangent-sparsegpt \
  --time 8:00:00 \
  --auto-close \
  -- \
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
    --output-dir "$OUT"

echo
gqueue
echo "logs: gjob log <id>   report: $OUT/report.json"
