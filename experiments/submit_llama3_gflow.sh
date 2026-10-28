#!/bin/bash
# One shared gflow job per Llama 3.1 / 3.2 size. Calibration is C4;
# eval is WikiText2 and PTB. Jobs are not chained.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${HOME}/.pixi/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

submit() {
  local name="$1" gpu_mem="$2" host_mem="$3" time_limit="$4" model="$5"
  gbatch \
    --gpus 1 \
    --shared \
    --gpu-memory "$gpu_mem" \
    --memory "$host_mem" \
    --name "$name" \
    --project llama3-sparsegpt \
    --time "$time_limit" \
    --auto-close \
    -- bash -lc "export PATH=\"\$HOME/.pixi/bin:\$PATH\"; cd /home/ubuntu/sparsegpt && pixi run --frozen -e gh200 python -u llama3_sparsity.py --models ${model}"
}

echo "submitting Llama 3 SparseGPT sweeps via gflow $(date)"
submit llama3-1b-base-c4-sparsegpt-sweep 16G 32G 8:00:00 meta-llama/Llama-3.2-1B
submit llama3-3b-base-c4-sparsegpt-sweep 24G 48G 16:00:00 meta-llama/Llama-3.2-3B
submit llama3-8b-base-c4-sparsegpt-sweep 32G 64G 24:00:00 meta-llama/Llama-3.1-8B
submit llama3-11b-vision-c4-sparsegpt-sweep 48G 64G 24:00:00 meta-llama/Llama-3.2-11B-Vision
submit llama3-70b-base-c4-sparsegpt-sweep 64G 224G 72:00:00 meta-llama/Llama-3.1-70B
submit llama3-90b-vision-c4-sparsegpt-sweep 64G 320G 96:00:00 meta-llama/Llama-3.2-90B-Vision
echo
gqueue
echo "done $(date)"
