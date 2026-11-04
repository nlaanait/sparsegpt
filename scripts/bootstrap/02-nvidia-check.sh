#!/usr/bin/env bash
# Verify NVIDIA driver / NVML. Does not install drivers (bare-metal / cloud
# images usually ship them). Fail soft so CPU-only bootstrap can continue.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

STRICT="${BOOTSTRAP_REQUIRE_GPU:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  msg="nvidia-smi not found; install NVIDIA drivers before GPU jobs"
  if [[ "$STRICT" == "1" ]]; then
    die "$msg (set BOOTSTRAP_REQUIRE_GPU=0 to continue without a GPU)"
  fi
  warn "$msg"
  exit 0
fi

log "nvidia-smi:"
nvidia-smi || die "nvidia-smi failed"

if ! ldconfig -p 2>/dev/null | grep -q 'libnvidia-ml\.so'; then
  warn "libnvidia-ml.so not visible via ldconfig; gflow GPU detection may fail"
else
  log "NVML library present"
fi

log "GPU check OK"
