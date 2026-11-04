#!/usr/bin/env bash
# Mid-size Qwen sparsity sweep: unstructured SparseGPT vs output-tangent.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

# 9B / 4B with fla linear-attn need tiny forward batches on GH200.
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
NAME="${1:-qwen-4b-tangent-c4-sparsity-sweep}"
OUT="${OUT:-results/tangent/qwen3.5-4b-c4-sparsity-sweep}"
PYTORCH_ATTENTION="${PYTORCH_ATTENTION:-1}"
FORWARD_BATCH="${FORWARD_BATCH:-1}"
NSAMPLES="${NSAMPLES:-64}"
DEPENDS_ON="${DEPENDS_ON:-}"
extra=()
if [[ -n "$DEPENDS_ON" ]]; then extra+=(--depends-on "$DEPENDS_ON"); fi
attn=()
if [[ "$PYTORCH_ATTENTION" == "1" ]]; then attn+=(--pytorch-attention); fi

gbatch \
  --gpus 1 --gpu-memory 96G --memory 96G \
  --name "$NAME" --project tangent-sparsegpt --time 36:00:00 --auto-close \
  "${extra[@]}" -- \
  bash -lc "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd \$HOME/sparsegpt && pixi run python -u -m experiments.tangent_sparsity_sweep --model '$MODEL' --calibration c4 --nsamples $NSAMPLES --forward-batch $FORWARD_BATCH --eval-dataset wikitext2 --tangent-alpha 0.5 --tangent-charts 8 --tangent-rank 2 --tangent-samples 512 --output-dir '$OUT' ${attn[*]}"

echo
gqueue
echo "report: $OUT/sweep_report.json"
