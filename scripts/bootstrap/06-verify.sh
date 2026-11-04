#!/usr/bin/env bash
# Post-bootstrap checks: tools, GPU, unit tests, CPU tangent reference.
# Optional GPU smoke via gbatch is in scripts/submit_tangent_smoke_gflow.sh.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_path_exports
WORKDIR="${BOOTSTRAP_WORKDIR:-$REPO_ROOT}"
cd "$WORKDIR"

fail=0

check() {
  local name="$1"
  shift
  if "$@"; then
    log "OK  $name"
  else
    warn "FAIL $name"
    fail=1
  fi
}

check "arch is aarch64" bash -c '[[ "$(uname -m)" == "aarch64" ]]'
check "pixi" pixi --version
check "gflowd" gflowd --version
check "gbatch" gbatch --help
check "gflowd reachable" ginfo
check "nvidia-smi" nvidia-smi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU inventory:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

check "pixi cuda-check" pixi run cuda-check
check "pytest geometry + sparsegpt" \
  pixi run python -m pytest -q tests/test_geometry.py tests/test_prune_geometry.py tests/test_sparsegpt.py tests/test_prune_sparsegpt.py

# Multi-seed synthetic tangent study (CPU, no HF download).
OUT="${BOOTSTRAP_TANGENT_REF_OUT:-/tmp/tangent-pruning-reference.json}"
check "tangent-reference" \
  pixi run python examples/tangent_pruning_experiment.py --seeds 0 1 2 --output "$OUT"

if [[ "$fail" -ne 0 ]]; then
  die "one or more verify checks failed"
fi

log "verify passed"
log "next: scripts/submit_tangent_smoke_gflow.sh   # small GPU pair via gbatch"
log "  or: scripts/submit_tangent_pair_gflow.sh     # Qwen3.5-2B matched pair"
