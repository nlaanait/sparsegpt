#!/bin/bash
# Submit one gflow job per OLMo 2 Instruct size for the C4-calibrated
# SparseGPT sweep (WikiText2 + PTB). Jobs are chained: 1B → 7B → 13B → 32B.
# Dense WikiText2 smoke must already be cleared (driver uses --skip-smoke).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

submit() {
  local name="$1" gpu_mem="$2" host_mem="$3" time_limit="$4" model="$5" dep="${6:-}"
  local -a cmd=(
    gbatch
    --gpus 1
    --gpu-memory "$gpu_mem"
    --memory "$host_mem"
    --name "$name"
    --project olmo2-sparsegpt
    --time "$time_limit"
    --auto-close
  )
  if [[ -n "$dep" ]]; then
    cmd+=(--depends-on "$dep")
  fi
  cmd+=(
    --
    pixi run python -u -m drivers.olmo2_sparsity
    --models "$model"
    --skip-smoke
  )
  "${cmd[@]}"
}

echo "submitting OLMo 2 SparseGPT sweeps via gflow $(date)"

out1=$(submit \
  olmo2-1b-instruct-c4-sparsegpt-sweep 16G 32G 8:00:00 \
  allenai/OLMo-2-0425-1B-Instruct)
echo "$out1"
id1=$(echo "$out1" | grep -oE '[0-9]+' | head -1)

out7=$(submit \
  olmo2-7b-instruct-c4-sparsegpt-sweep 32G 48G 16:00:00 \
  allenai/OLMo-2-1124-7B-Instruct "$id1")
echo "$out7"
id7=$(echo "$out7" | grep -oE '[0-9]+' | head -1)

out13=$(submit \
  olmo2-13b-instruct-c4-sparsegpt-sweep 48G 64G 24:00:00 \
  allenai/OLMo-2-1124-13B-Instruct "$id7")
echo "$out13"
id13=$(echo "$out13" | grep -oE '[0-9]+' | head -1)

out32=$(submit \
  olmo2-32b-instruct-c4-sparsegpt-sweep 96G 96G 48:00:00 \
  allenai/OLMo-2-0325-32B-Instruct "$id13")
echo "$out32"

echo
gqueue
echo "done $(date)"
