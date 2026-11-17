#!/usr/bin/env bash
# Run inside a gflow allocation. Propagate cancellation to the systemd cgroup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$1"
TELEMETRY="$2"
shift 2
GPU_CAP="${GSM8K_GPU_CAP_GIB:-56}"
HOST_STOP="${GSM8K_HOST_STOP_GIB:-80}"
HOST_MAX="${GSM8K_HOST_MEMORY_MAX:-80G}"
if [[ ! "$NAME" =~ ^[a-z][a-z0-9-]+$ ]]; then
  echo "invalid systemd unit name" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "refusing GPU worker without scheduler-provided CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
cleanup() { systemctl --user stop "$NAME.service" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
systemd-run --user --wait --pipe --collect --unit="$NAME" \
  -p MemoryMax="$HOST_MAX" -p MemorySwapMax=0 -p MemoryAccounting=yes \
  -p KillMode=control-group -p RuntimeMaxSec=86400 \
  --working-directory="$ROOT" --setenv="PYTHONPATH=$ROOT/src:$ROOT" \
  --setenv="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --setenv="SPARSIFIER_GPU_CAP_GIB=$GPU_CAP" \
  --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=8 --setenv=MKL_NUM_THREADS=8 \
  "$ROOT/.pixi/envs/default/bin/python" -u -m sparsifier.eval.memory_guard \
  --gpu-cap-gib "$GPU_CAP" --host-stop-gib "$HOST_STOP" --telemetry "$TELEMETRY" -- "$@" &
SERVICE_CLIENT=$!
wait "$SERVICE_CLIENT"
