#!/bin/bash
# Wait until the Qwen 3.8 driver is off the GPU, prove the E2B layer-wise
# forward, then run the four instruction-tuned Gemma 4 models.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p figures/gemma/logs
exec > figures/gemma/logs/driver.log 2>&1

echo "download E2B $(date)"
pixi run python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it')"

echo "waiting for Qwen $(date)"
while pgrep -f "qwen_36_38_sparsity.py" >/dev/null || pgrep -f "python -m sparsegpt.qwen" >/dev/null; do
  echo "still waiting $(date)"
  sleep 60
done

echo "gate $(date)"
pixi run python -u -m experiments.gemma_gate
echo "sweep $(date)"
pixi run python -u -m analysis.sweep
echo "done $(date)"
