#!/usr/bin/env bash
# Bootstrap a bare-metal Ubuntu GH200 node for tangent-aware-pruning.
#
# Usage (on the node, from a clone or a copied scripts/ tree):
#   ./scripts/bootstrap/bootstrap.sh
#   BOOTSTRAP_WORKDIR=/data/sparsegpt ./scripts/bootstrap/bootstrap.sh
#   CAUSAL_CONV1D_WHEEL=/path/to/wheel.whl ./scripts/bootstrap/bootstrap.sh
#
# Phases:
#   1 system apt packages
#   2 NVIDIA / NVML check
#   3 pixi
#   4 gflow (runqd) + gflowd
#   5 repo checkout + pixi install
#   6 verify (tests + CPU tangent reference)
#
# Skip phases with BOOTSTRAP_SKIP=1,2 or only run some with BOOTSTRAP_ONLY=3,4
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ONLY="${BOOTSTRAP_ONLY:-}"
SKIP="${BOOTSTRAP_SKIP:-}"

should_run() {
  local id="$1"
  if [[ -n "$ONLY" ]]; then
    [[ ",$ONLY," == *",$id,"* ]]
    return
  fi
  if [[ -n "$SKIP" ]]; then
    [[ ",$SKIP," != *",$id,"* ]]
    return
  fi
  return 0
}

run_phase() {
  local id="$1"
  local script="$2"
  if ! should_run "$id"; then
    log "skip phase $id ($script)"
    return 0
  fi
  log "==== phase $id: $script ===="
  bash "${BOOTSTRAP_DIR}/${script}"
}

log "GH200 bootstrap starting (arch=$(detect_arch), root=$REPO_ROOT)"
run_phase 1 01-system-packages.sh
run_phase 2 02-nvidia-check.sh
run_phase 3 03-pixi.sh
run_phase 4 04-gflow.sh
run_phase 5 05-repo-env.sh
run_phase 6 06-verify.sh
log "bootstrap complete"
