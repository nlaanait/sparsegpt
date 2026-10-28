#!/bin/bash
# Submit one shared gflow job per model for likelihood-metric collection.
# Dense + SparseGPT ladder (0.1..0.8). Prune stays in-memory — no sparse_opt
# checkpoints. Jobs are independent (--shared, no depends-on) so the scheduler
# can pack anything that fits residual GPU memory.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${HOME}/.pixi/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

submit() {
  local name="$1" project="$2" gpu_mem="$3" host_mem="$4" time_limit="$5"
  local family="$6" model="$7"
  gbatch \
    --gpus 1 \
    --shared \
    --gpu-memory "$gpu_mem" \
    --memory "$host_mem" \
    --name "$name" \
    --project "$project" \
    --time "$time_limit" \
    --auto-close \
    -- bash -lc "export PATH=\"\$HOME/.pixi/bin:\$PATH\"; cd /home/ubuntu/sparsegpt && pixi run --frozen -e gh200 python -u collect_likelihood_metrics.py --family ${family} --models ${model}"
}

echo "submitting likelihood-metric jobs (no ckpt, shared, no chain) $(date)"
df -h / | tail -1

# OLMo 2 Instruct (C4 → wt/ptb)
submit olmo2-1b-instruct-c4-likelihood  olmo2-likelihood 16G 32G 12:00:00 olmo2 allenai/OLMo-2-0425-1B-Instruct
submit olmo2-7b-instruct-c4-likelihood  olmo2-likelihood 32G 48G 24:00:00 olmo2 allenai/OLMo-2-1124-7B-Instruct
submit olmo2-13b-instruct-c4-likelihood olmo2-likelihood 48G 64G 36:00:00 olmo2 allenai/OLMo-2-1124-13B-Instruct
submit olmo2-32b-instruct-c4-likelihood olmo2-likelihood 96G 96G 72:00:00 olmo2 allenai/OLMo-2-0325-32B-Instruct

# Llama 3 (C4 → wt/ptb)
submit llama3-1b-base-c4-likelihood        llama3-likelihood 16G 32G 12:00:00 llama3 meta-llama/Llama-3.2-1B
submit llama3-3b-base-c4-likelihood        llama3-likelihood 24G 48G 24:00:00 llama3 meta-llama/Llama-3.2-3B
submit llama3-8b-base-c4-likelihood        llama3-likelihood 32G 64G 36:00:00 llama3 meta-llama/Llama-3.1-8B
submit llama3-11b-vision-c4-likelihood     llama3-likelihood 48G 64G 36:00:00 llama3 meta-llama/Llama-3.2-11B-Vision
submit llama3-70b-base-c4-likelihood       llama3-likelihood 64G 224G 96:00:00 llama3 meta-llama/Llama-3.1-70B
submit llama3-90b-vision-c4-likelihood     llama3-likelihood 64G 320G 120:00:00 llama3 meta-llama/Llama-3.2-90B-Vision

# Qwen3.5 (C4 → wt/ptb)
submit qwen35-0.8b-c4-likelihood qwen-likelihood 16G 32G 12:00:00 qwen3.5 Qwen/Qwen3.5-0.8B
submit qwen35-2b-c4-likelihood   qwen-likelihood 24G 48G 18:00:00 qwen3.5 Qwen/Qwen3.5-2B
submit qwen35-4b-c4-likelihood   qwen-likelihood 32G 64G 24:00:00 qwen3.5 Qwen/Qwen3.5-4B
submit qwen35-9b-c4-likelihood   qwen-likelihood 48G 64G 36:00:00 qwen3.5 Qwen/Qwen3.5-9B
submit qwen35-27b-c4-likelihood  qwen-likelihood 64G 96G 72:00:00 qwen3.5 Qwen/Qwen3.5-27B

# Qwen 3.6 / 3.8 (C4 → wt/ptb)
submit qwen36-27b-c4-likelihood qwen-likelihood 64G 96G 72:00:00 qwen Qwen/Qwen3.6-27B
submit qwen38-27b-c4-likelihood qwen-likelihood 64G 96G 72:00:00 qwen Qwen/Qwen3.8-27B

# Gemma 4 base (C4 → wt/ptb)
submit gemma-e2b-c4-likelihood      gemma-likelihood 24G 48G 24:00:00 gemma google/gemma-4-E2B
submit gemma-e4b-c4-likelihood      gemma-likelihood 32G 64G 36:00:00 gemma google/gemma-4-E4B
submit gemma-12b-c4-likelihood     gemma-likelihood 48G 64G 48:00:00 gemma google/gemma-4-12B
submit gemma-26b-a4b-c4-likelihood gemma-likelihood 64G 96G 72:00:00 gemma google/gemma-4-26B-A4B
submit gemma-31b-c4-likelihood     gemma-likelihood 64G 96G 72:00:00 gemma google/gemma-4-31B

echo
gqueue
echo "done $(date)"
df -h / | tail -1
