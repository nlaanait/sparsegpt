#!/usr/bin/env bash
# Mid-size sparsity sweep: unstructured SparseGPT vs output-tangent (likelihood).
# Default model is OLMo-2-7B-Instruct: Qwen3.5 mid-size OOMs on GH200 under FLA.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

MODEL="${MODEL:-allenai/OLMo-2-1124-7B-Instruct}"
NAME="${1:-olmo2-7b-tangent-c4-sparsity-sweep}"
OUT="${OUT:-results/tangent/olmo2-7b-c4-sparsity-sweep}"
FORWARD_BATCH="${FORWARD_BATCH:-8}"
NSAMPLES="${NSAMPLES:-128}"
# GH200 driver 12.8: use the restored pixi env (torch 2.10+cu128 + causal-conv1d wheel).
PIXI_ENV="${PIXI_ENV:-gh200}"
# Only set for Qwen linear-attn fallbacks; OLMo does not need it.
PYTORCH_ATTENTION="${PYTORCH_ATTENTION:-0}"
DEPENDS_ON="${DEPENDS_ON:-}"
extra=()
if [[ -n "$DEPENDS_ON" ]]; then extra+=(--depends-on "$DEPENDS_ON"); fi
attn=()
if [[ "$PYTORCH_ATTENTION" == "1" ]]; then attn+=(--pytorch-attention); fi

gbatch \
  --gpus 1 --gpu-memory 48G --memory 64G \
  --name "$NAME" --project tangent-sparsegpt --time 36:00:00 --auto-close \
  "${extra[@]}" -- \
  bash -lc "cd \$HOME/sparsegpt && pixi run -e '$PIXI_ENV' python -u -m experiments.tangent_sparsity_sweep --model '$MODEL' --calibration c4 --nsamples $NSAMPLES --forward-batch $FORWARD_BATCH --eval-dataset wikitext2 --tangent-alpha 0.5 --tangent-charts 8 --tangent-rank 2 --tangent-samples 512 --output-dir '$OUT' ${attn[*]}"

echo
gqueue
echo "report: $OUT/sweep_report.json"
