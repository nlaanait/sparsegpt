#!/usr/bin/env bash
# Idempotent install of a built causal-conv1d wheel into the Pixi env.
# Safe to call at the start of gbatch jobs that --depends-on the wheel build.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

WHEEL_DIR="${CAUSAL_CONV1D_WHEEL_DIR:-${HOME}/.pixi/wheels/cu130}"
READY="${WHEEL_DIR}/READY"
LOCK="${CAUSAL_CONV1D_INSTALL_LOCK:-/tmp/causal-conv1d-install.lock}"

if pixi run python -c "import causal_conv1d; print(causal_conv1d.__file__)" >/dev/null 2>&1; then
  echo "causal_conv1d already importable"
  exit 0
fi

whl=""
if [[ -f "$READY" ]]; then
  whl="$(tr -d '[:space:]' <"$READY")"
fi
if [[ -z "$whl" || ! -f "$whl" ]]; then
  whl="$(ls -1 "$WHEEL_DIR"/causal_conv1d-*-linux_aarch64.whl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$whl" || ! -f "$whl" ]]; then
  echo "ERROR: no causal-conv1d wheel under $WHEEL_DIR (READY missing)." >&2
  echo "       Run scripts/submit_causal_conv1d_build_gflow.sh first." >&2
  exit 1
fi

exec 9>"$LOCK"
flock 9
if pixi run python -c "import causal_conv1d" >/dev/null 2>&1; then
  echo "causal_conv1d already importable (after wait)"
  exit 0
fi

echo "installing $whl into pixi env"
pixi run python -m pip install --no-deps --force-reinstall "$whl"
pixi run python -c "import causal_conv1d; print('causal_conv1d OK', causal_conv1d.__file__)"
