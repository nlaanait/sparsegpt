#!/usr/bin/env bash
# Checkout tangent-aware-pruning (or keep the current tree) and `pixi install`
# for GH200 (linux-aarch64 + CUDA 13.0).
#
# The workspace pixi.toml pins linux-aarch64 / cu130 and optionally a local
# causal-conv1d wheel under ~/.pixi/wheels/cu130. On a fresh GH200 that wheel
# is usually missing; this script stages a copy from CAUSAL_CONV1D_WHEEL or
# strips the path dep so `pixi install` can proceed (Qwen3.5 linear-attn
# kernels degrade without it; tangent smokes and most families still run).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_path_exports
need_cmd pixi
need_cmd git

REPO_URL="${BOOTSTRAP_REPO_URL:-https://github.com/nlaanait/sparsegpt.git}"
WORKDIR="${BOOTSTRAP_WORKDIR:-$REPO_ROOT}"
CUDA_TAG="${BOOTSTRAP_CUDA_TAG:-13.0}"
EXPECT_ARCH="${BOOTSTRAP_EXPECT_ARCH:-aarch64}"

# Prefer an explicit BOOTSTRAP_BRANCH. Otherwise stay on the current checkout
# (e.g. cursor/gpu-node-bootstrap-*) so a single-branch clone of the bootstrap
# PR does not force-fetch tangent-aware-pruning and fail.
if [[ -n "${BOOTSTRAP_BRANCH:-}" ]]; then
  BRANCH="$BOOTSTRAP_BRANCH"
elif [[ -d "${WORKDIR}/.git" ]]; then
  BRANCH="$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD)"
else
  BRANCH="tangent-aware-pruning"
fi

arch="$(detect_arch)"
if [[ "$arch" != "$EXPECT_ARCH" ]]; then
  die "expected $EXPECT_ARCH (GH200 Grace) but uname -m resolved to $arch"
fi
if [[ "$(pixi_platform)" != "linux-aarch64" ]]; then
  die "pixi platform is $(pixi_platform); GH200 bootstrap expects linux-aarch64"
fi

prepare_tree() {
  if [[ -d "${WORKDIR}/.git" ]]; then
    log "using existing git tree at $WORKDIR (want branch $BRANCH)"
    cd "$WORKDIR"
    # Single-branch clones do not create refs/remotes/origin/<other>; write the
    # ref explicitly so checkout works for any BOOTSTRAP_BRANCH.
    if ! git fetch origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"; then
      git fetch origin || die "git fetch origin failed"
    fi
    local cur
    cur="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$cur" != "$BRANCH" ]]; then
      if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
      elif git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        git checkout -B "$BRANCH" "origin/$BRANCH"
      else
        die "branch $BRANCH not found locally or on origin after fetch"
      fi
    fi
    git pull --ff-only origin "$BRANCH" 2>/dev/null || true
  else
    log "cloning $REPO_URL ($BRANCH) into $WORKDIR"
    mkdir -p "$(dirname "$WORKDIR")"
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
  fi
  log "HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
}

toml_causal_wheel_path() {
  python3 - <<'PY' "${WORKDIR}/pixi.toml"
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text()
m = re.search(r'causal-conv1d\s*=\s*\{[^}]*path\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "")
PY
}

rewrite_toml_wheel_path() {
  local new_path="$1"
  local toml="${WORKDIR}/pixi.toml"
  [[ -f "$toml" ]] || die "missing $toml"
  if [[ ! -f "${toml}.bootstrap.bak" ]]; then
    cp -f "$toml" "${toml}.bootstrap.bak"
  fi
  python3 - <<'PY' "$toml" "$new_path"
from pathlib import Path
import re, sys
path, new = Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
new_text, n = re.subn(
    r'(causal-conv1d\s*=\s*\{[^}]*path\s*=\s*")([^"]+)(")',
    lambda m: f"{m.group(1)}{new}{m.group(3)}",
    text,
    count=1,
)
if not n:
    raise SystemExit("causal-conv1d path pin not found in pixi.toml")
path.write_text(new_text)
print(f"pointed causal-conv1d path at {new}")
PY
}

stage_causal_conv1d_wheel() {
  local pinned
  pinned="$(toml_causal_wheel_path)"
  local dest_dir="${HOME}/.pixi/wheels/cu130"
  local dest_whl="${dest_dir}/causal_conv1d-1.7.0-cp312-cp312-linux_aarch64.whl"
  local src="${CAUSAL_CONV1D_WHEEL:-}"

  mkdir -p "$dest_dir"

  # Prefer an explicit wheel, then whatever pixi.toml already points at.
  if [[ -z "$src" && -n "$pinned" && -f "$pinned" ]]; then
    src="$pinned"
  fi
  if [[ -z "$src" && -f /home/nlaanait/.pixi/wheels/cu130/causal_conv1d-1.7.0-cp312-cp312-linux_aarch64.whl ]]; then
    src=/home/nlaanait/.pixi/wheels/cu130/causal_conv1d-1.7.0-cp312-cp312-linux_aarch64.whl
  fi

  if [[ -n "$src" && -f "$src" ]]; then
    if [[ "$src" != "$dest_whl" ]]; then
      log "staging causal-conv1d wheel from $src -> $dest_whl"
      cp -f "$src" "$dest_whl"
    else
      log "causal-conv1d wheel already at $dest_whl"
    fi
    if [[ -n "$pinned" && "$pinned" != "$dest_whl" ]]; then
      rewrite_toml_wheel_path "$dest_whl"
    fi
    return 0
  fi

  if [[ -f "$dest_whl" ]]; then
    log "causal-conv1d wheel already at $dest_whl"
    if [[ -n "$pinned" && "$pinned" != "$dest_whl" ]]; then
      rewrite_toml_wheel_path "$dest_whl"
    fi
    return 0
  fi

  warn "no causal-conv1d aarch64 wheel found"
  warn "set CAUSAL_CONV1D_WHEEL=/path/to/causal_conv1d-*-linux_aarch64.whl to keep Qwen3.5 CUDA kernels"
  return 1
}

patch_pixi_for_missing_wheel() {
  # Drop the absolute path dep and the flash-linear-attention `conv1d` extra
  # (that extra re-pulls causal-conv1d from PyPI and compiles it during
  # `pixi install`, which fails when system nvcc ≠ torch's CUDA).
  # Backup once at pixi.toml.bootstrap.bak.
  local toml="${WORKDIR}/pixi.toml"
  [[ -f "$toml" ]] || die "missing $toml"
  if [[ ! -f "${toml}.bootstrap.bak" ]]; then
    cp -f "$toml" "${toml}.bootstrap.bak"
  fi

  python3 - <<'PY' "$toml"
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text()
changed = []
new, n = re.subn(
    r'(?m)^causal-conv1d\s*=\s*\{[^}]*\}\s*\n',
    '',
    text,
)
if n:
    changed.append(f"removed {n} causal-conv1d path pin(s)")
    text = new
# Keep fla, but do not pull/build causal-conv1d via the conv1d extra.
def _drop_conv1d(m: re.Match[str]) -> str:
    parts = [x.strip().strip("\"'") for x in m.group(2).split(",")]
    parts = [p for p in parts if p and p != "conv1d"]
    return m.group(1) + ", ".join(f'"{p}"' for p in parts) + m.group(3)

new, n = re.subn(
    r'(flash-linear-attention\s*=\s*\{[^}]*extras\s*=\s*\[)([^\]]*)(\])',
    _drop_conv1d,
    text,
    count=1,
)
if n:
    changed.append("dropped flash-linear-attention conv1d extra")
    text = new
path.write_text(text)
print("; ".join(changed) if changed else f"no causal-conv1d / conv1d edits in {path}")
PY
}

ensure_hf_cache() {
  mkdir -p "${HOME}/.cache/huggingface/hub"
  if [[ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
    log "HF token present in environment (gated models OK)"
  else
    warn "no HF_TOKEN / HUGGING_FACE_HUB_TOKEN; public models only until you export one"
  fi
}

pixi_install() {
  cd "$WORKDIR"
  log "pixi install (platform=$(pixi_platform), cuda=$CUDA_TAG)"
  # Lockfile is authored for linux-aarch64+cu130 on the Grace host; GH200 matches.
  pixi install
  log "pixi env ready"
  pixi run cuda-check || warn "cuda-check failed inside pixi (driver/toolkit mismatch?)"
}

prepare_tree
ensure_hf_cache

if stage_causal_conv1d_wheel; then
  log "keeping causal-conv1d path dependency"
else
  if [[ "${BOOTSTRAP_ALLOW_STRIP_CONV1D:-1}" == "1" ]]; then
    patch_pixi_for_missing_wheel
  else
    die "causal-conv1d wheel required (BOOTSTRAP_ALLOW_STRIP_CONV1D=0)"
  fi
fi

pixi_install
log "repo env ready at $WORKDIR"
