#!/usr/bin/env bash
# Shared helpers for bare-metal Ubuntu GPU node bootstrap.
set -euo pipefail

BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BOOTSTRAP_DIR}/../.." && pwd)"

log()  { printf '[bootstrap] %s\n' "$*"; }
warn() { printf '[bootstrap] WARN: %s\n' "$*" >&2; }
die()  { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

ensure_path_exports() {
  # pixi / uv / pipx / cargo / local bin
  export PATH="${HOME}/.pixi/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
}

append_shell_path() {
  local marker="# sparsegpt-gpu-bootstrap PATH"
  local line='export PATH="$HOME/.pixi/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
  local rc="${HOME}/.bashrc"
  if [[ -f "${HOME}/.zshrc" ]]; then
    rc="${HOME}/.zshrc"
  fi
  if [[ -f "$rc" ]] && grep -Fq "$marker" "$rc"; then
    return 0
  fi
  {
    echo
    echo "$marker"
    echo "$line"
  } >>"$rc"
  log "appended PATH exports to $rc"
}

detect_arch() {
  local m
  m="$(uname -m)"
  case "$m" in
    aarch64|arm64) echo aarch64 ;;
    x86_64|amd64)  echo x86_64 ;;
    *) die "unsupported architecture: $m" ;;
  esac
}

pixi_platform() {
  # pixi.toml platform names
  case "$(detect_arch)" in
    aarch64) echo linux-aarch64 ;;
    x86_64)  echo linux-64 ;;
  esac
}

have_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    return 0
  fi
  command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null
}

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "need root or passwordless sudo to run: $*"
  fi
}
