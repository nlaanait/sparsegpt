#!/usr/bin/env bash
# Mid-size Qwen3.5-9B: unstructured SparseGPT vs output-tangent across a
# sparsity sweep (likelihood + perplexity on WikiText2). One gbatch job,
# resumable via results/tangent/.../sweep_report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! ginfo >/dev/null 2>&1; then
  echo "gflowd is not reachable; start with: gflowd up" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
NAME="${1:-qwen-9b-tangent-c4-sparsity-sweep}"
OUT="${OUT:-results/tangent/qwen3.5-9b-c4-sparsity-sweep}"
# Optional: DEPENDS_ON=<build-job-id> to wait for causal-conv1d; otherwise use
# pytorch attention fallback so the sweep can start immediately.
PYTORCH_ATTENTION="${PYTORCH_ATTENTION:-1}"
DEPENDS_ON="${DEPENDS_ON:-}"

extra=()
if [[ -n "$DEPENDS_ON" ]]; then
  extra+=(--depends-on "$DEPENDS_ON")
fi

inner=(
  pixi run python -u -m experiments.tangent_sparsity_sweep
  --model "$MODEL"
  --calibration c4
  --nsamples 128
  --eval-dataset wikitext2
  --tangent-alpha 0.5
  --tangent-charts 8
  --tangent-rank 2
  --tangent-samples 512
  --output-dir "$OUT"
)
if [[ "$PYTORCH_ATTENTION" == "1" ]]; then
  inner+=(--pytorch-attention)
fi

gbatch \
  --gpus 1 \
  --gpu-memory 48G \
  --memory 64G \
  --name "$NAME" \
  --project tangent-sparsegpt \
  --time 36:00:00 \
  --auto-close \
  "${extra[@]}" \
  -- \
  "${inner[@]}"

echo
gqueue
echo "report: $OUT/sweep_report.json"
