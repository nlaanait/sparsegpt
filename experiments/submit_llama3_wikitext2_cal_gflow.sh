#!/bin/bash
# One shared gflow job per base Llama 3.1 / 3.2 size.
# Calibration is WikiText2; eval is C4 and PTB. Jobs are not chained.
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
    -- bash -lc "export PATH=\"\$HOME/.pixi/bin:\$PATH\"; cd /home/ubuntu/sparsegpt && pixi run --frozen -e gh200 python -u llama3_wikitext2_cal_sparsity.py --models ${model}"
}

echo "submitting Llama 3 WikiText2-calibrated sweeps via gflow $(date)"
submit llama3-1b-base-wikitext2-sparsegpt-sweep 16G 32G 8:00:00 meta-llama/Llama-3.2-1B
submit llama3-3b-base-wikitext2-sparsegpt-sweep 24G 48G 16:00:00 meta-llama/Llama-3.2-3B
submit llama3-8b-base-wikitext2-sparsegpt-sweep 32G 64G 24:00:00 meta-llama/Llama-3.1-8B
submit llama3-11b-vision-wikitext2-sparsegpt-sweep 48G 64G 24:00:00 meta-llama/Llama-3.2-11B-Vision
submit llama3-70b-base-wikitext2-sparsegpt-sweep 64G 224G 72:00:00 meta-llama/Llama-3.1-70B
submit llama3-90b-vision-wikitext2-sparsegpt-sweep 64G 320G 96:00:00 meta-llama/Llama-3.2-90B-Vision
echo
gqueue
echo "done $(date)"
