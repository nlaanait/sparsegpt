#!/usr/bin/env bash
# Build causal-conv1d for the current Pixi torch (cu130 / cp312 / aarch64) and
# drop a wheel under ~/.pixi/wheels/cu130. Does not mutate pixi.toml.
#
# Intended to run under gbatch while other GPU jobs that tolerate the PyTorch
# attention fallback proceed. Jobs that want the CUDA op should --depends-on
# this job and call scripts/ensure_causal_conv1d.sh at start.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

WHEEL_DIR="${CAUSAL_CONV1D_WHEEL_DIR:-${HOME}/.pixi/wheels/cu130}"
VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"
SRC="${CAUSAL_CONV1D_SOURCE:-}" # empty → PyPI sdist; or git+https://...
LOCK="${CAUSAL_CONV1D_LOCK:-/tmp/causal-conv1d-build.lock}"
READY="${WHEEL_DIR}/READY"

mkdir -p "$WHEEL_DIR"

detect_arch_list() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    printf '%s\n' "$TORCH_CUDA_ARCH_LIST"
    return
  fi
  # GH200 Hopper → 9.0; override for other SKUs.
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  case "$cap" in
    9.0*) echo "9.0+PTX" ;;
    10.*) echo "10.0+PTX" ;;
    12.*) echo "12.1+PTX" ;;
    "")   echo "9.0+PTX" ;; # GH200 default when nvidia-smi unavailable in build jail
    *)    echo "${cap}+PTX" ;;
  esac
}

need_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc: $(nvcc --version | tail -1)"
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    echo "nvcc: $(nvcc --version | tail -1)"
    return 0
  fi
  echo "ERROR: nvcc not found. Install CUDA toolkit matching the driver (CUDA 13.x on GH200)," >&2
  echo "       or set CUDA_HOME to a toolkit that provides nvcc." >&2
  exit 1
}

existing="$(ls -1 "$WHEEL_DIR"/causal_conv1d-*-cp312-*-linux_aarch64.whl 2>/dev/null | head -1 || true)"
if [[ -n "$existing" && -f "$READY" && "${CAUSAL_CONV1D_FORCE:-0}" != "1" ]]; then
  echo "wheel already present: $existing"
  echo "READY: $READY"
  exit 0
fi

need_nvcc
ARCH_LIST="$(detect_arch_list)"
echo "TORCH_CUDA_ARCH_LIST=$ARCH_LIST"
echo "wheel dir: $WHEEL_DIR"
echo "version/source: ${SRC:-causal-conv1d==$VERSION}"

# flock so a manual rebuild and a gbatch rebuild do not clobber each other
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another causal-conv1d build holds $LOCK; waiting"
  flock 9
fi

pixi run bash -lc "
set -euo pipefail
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST='${ARCH_LIST}'
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
PY
SPEC='${SRC:-causal-conv1d==${VERSION}}'
TMP=\$(mktemp -d)
trap 'rm -rf \"\$TMP\"' EXIT
python -m pip wheel --no-build-isolation --no-deps -w \"\$TMP\" \"\$SPEC\"
whl=\$(ls -1 \"\$TMP\"/causal_conv1d-*.whl | head -1)
test -n \"\$whl\"
# Normalize name toward the pixi.toml expectation when version matches 1.7.0.
base=\$(basename \"\$whl\")
dest='${WHEEL_DIR}'/\$base
cp -f \"\$whl\" \"\$dest\"
# Stable alias used by ensure + bootstrap path rewrites.
if [[ \"\$base\" == causal_conv1d-${VERSION}-* ]]; then
  ln -sfn \"\$base\" '${WHEEL_DIR}/causal_conv1d-${VERSION}-cp312-cp312-linux_aarch64.whl'
fi
printf '%s\\n' \"\$dest\" > '${READY}'
echo \"wrote \$dest\"
ls -lh '${WHEEL_DIR}'
"

echo "build complete: $(cat "$READY")"
