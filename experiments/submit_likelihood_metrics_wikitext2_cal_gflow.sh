#!/bin/bash
# WikiText2-calibrated likelihood-metric jobs (eval C4 + PTB).
# Same shared / no-ckpt / no-chain policy as the C4 submit script.
# Gemma counterpart is the -it WikiText2-cal family (no base wt-cal spec).
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

echo "submitting WikiText2-cal likelihood-metric jobs $(date)"
df -h / | tail -1

# OLMo 2 Instruct (WikiText2 → c4/ptb)
submit olmo2-1b-instruct-wikitext2-likelihood  olmo2-likelihood 16G 32G 12:00:00 olmo2_wikitext2_cal allenai/OLMo-2-0425-1B-Instruct
submit olmo2-7b-instruct-wikitext2-likelihood  olmo2-likelihood 32G 48G 24:00:00 olmo2_wikitext2_cal allenai/OLMo-2-1124-7B-Instruct
submit olmo2-13b-instruct-wikitext2-likelihood olmo2-likelihood 48G 64G 36:00:00 olmo2_wikitext2_cal allenai/OLMo-2-1124-13B-Instruct
submit olmo2-32b-instruct-wikitext2-likelihood olmo2-likelihood 96G 96G 72:00:00 olmo2_wikitext2_cal allenai/OLMo-2-0325-32B-Instruct

# Llama 3 (WikiText2 → c4/ptb)
submit llama3-1b-base-wikitext2-likelihood        llama3-likelihood 16G 32G 12:00:00 llama3_wikitext2_cal meta-llama/Llama-3.2-1B
submit llama3-3b-base-wikitext2-likelihood        llama3-likelihood 24G 48G 24:00:00 llama3_wikitext2_cal meta-llama/Llama-3.2-3B
submit llama3-8b-base-wikitext2-likelihood        llama3-likelihood 32G 64G 36:00:00 llama3_wikitext2_cal meta-llama/Llama-3.1-8B
submit llama3-11b-vision-wikitext2-likelihood     llama3-likelihood 48G 64G 36:00:00 llama3_wikitext2_cal meta-llama/Llama-3.2-11B-Vision
submit llama3-70b-base-wikitext2-likelihood       llama3-likelihood 64G 224G 96:00:00 llama3_wikitext2_cal meta-llama/Llama-3.1-70B
submit llama3-90b-vision-wikitext2-likelihood     llama3-likelihood 64G 320G 120:00:00 llama3_wikitext2_cal meta-llama/Llama-3.2-90B-Vision

# Qwen3.5 (WikiText2 → c4/ptb)
submit qwen35-0.8b-wikitext2-likelihood qwen-likelihood 16G 32G 12:00:00 qwen3.5_wikitext2_cal Qwen/Qwen3.5-0.8B
submit qwen35-2b-wikitext2-likelihood   qwen-likelihood 24G 48G 18:00:00 qwen3.5_wikitext2_cal Qwen/Qwen3.5-2B
submit qwen35-4b-wikitext2-likelihood   qwen-likelihood 32G 64G 24:00:00 qwen3.5_wikitext2_cal Qwen/Qwen3.5-4B
submit qwen35-9b-wikitext2-likelihood   qwen-likelihood 48G 64G 36:00:00 qwen3.5_wikitext2_cal Qwen/Qwen3.5-9B
submit qwen35-27b-wikitext2-likelihood  qwen-likelihood 56G 96G 72:00:00 qwen3.5_wikitext2_cal Qwen/Qwen3.5-27B

# Qwen 3.6 / 3.8 (WikiText2 → c4/ptb)
submit qwen36-27b-wikitext2-likelihood qwen-likelihood 56G 96G 72:00:00 qwen_wikitext2_cal Qwen/Qwen3.6-27B
submit qwen38-27b-wikitext2-likelihood qwen-likelihood 56G 96G 72:00:00 qwen_wikitext2_cal Qwen/Qwen3.8-27B

# Gemma 4 Instruct (WikiText2 → c4/ptb)
submit gemma-e4b-it-wikitext2-likelihood  gemma-likelihood 32G 64G 36:00:00 gemma_it_wikitext2_cal google/gemma-4-E4B-it
submit gemma-12b-it-wikitext2-likelihood  gemma-likelihood 48G 64G 48:00:00 gemma_it_wikitext2_cal google/gemma-4-12B-it
submit gemma-31b-it-wikitext2-likelihood  gemma-likelihood 64G 96G 72:00:00 gemma_it_wikitext2_cal google/gemma-4-31B-it

echo
gqueue
echo "done $(date)"
df -h / | tail -1
