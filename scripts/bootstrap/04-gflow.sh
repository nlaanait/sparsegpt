#!/usr/bin/env bash
# Install gflow (PyPI package name: runqd) and start gflowd.
# Prefer uv tool install; fall back to pipx, then pip --user.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_path_exports
append_shell_path

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  log "installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
  ensure_path_exports
  need_cmd uv
}

install_runqd() {
  if command -v gflowd >/dev/null 2>&1 && command -v gbatch >/dev/null 2>&1; then
    log "gflow already installed: $(gflowd --version 2>/dev/null || echo ok)"
    return 0
  fi

  install_uv
  log "installing runqd (gflow) via uv tool"
  uv tool install runqd
  ensure_path_exports
  need_cmd gflowd
  need_cmd gbatch
  need_cmd gqueue
  need_cmd gjob
  need_cmd ginfo
  need_cmd gcancel
  log "gflow installed: $(gflowd --version)"
}

configure_daemon() {
  mkdir -p "${HOME}/.config/gflow"
  if [[ ! -f "${HOME}/.config/gflow/gflow.toml" ]]; then
    log "initializing gflow config"
    gflowd init || true
  fi

  # Prefer systemd user service when a user bus exists (typical after login /
  # linger). Otherwise gflowd up/start falls back to tmux or a detached process.
  if command -v systemctl >/dev/null 2>&1 && systemctl --user status >/dev/null 2>&1; then
    if command -v loginctl >/dev/null 2>&1; then
      loginctl enable-linger "$(id -un)" 2>/dev/null || warn "could not enable linger (daemon may stop on logout)"
    fi
    if gflowd service install >/dev/null 2>&1; then
      log "installed gflowd systemd user service"
    else
      warn "gflowd service install failed; falling back to gflowd up"
    fi
  fi

  if ginfo >/dev/null 2>&1; then
    log "gflowd already reachable"
  else
    log "starting gflowd"
    gflowd up || gflowd start || die "failed to start gflowd"
  fi

  gflowd status || true
  log "ginfo:"
  ginfo || warn "ginfo failed (GPU detection may need nvidia-smi / NVML)"
}

configure_cursor_mcp() {
  # Optional: wire gflow MCP for Cursor agents on this node.
  local mcp="${HOME}/.cursor/mcp.json"
  mkdir -p "${HOME}/.cursor"
  if [[ -f "$mcp" ]] && grep -q '"gflow"' "$mcp" 2>/dev/null; then
    log "Cursor MCP already mentions gflow ($mcp)"
    return 0
  fi
  if [[ -f "$mcp" ]]; then
    warn "refusing to overwrite existing $mcp; add gflow mcp manually: gflow mcp serve"
    return 0
  fi
  cat >"$mcp" <<'EOF'
{
  "mcpServers": {
    "gflow": {
      "command": "gflow",
      "args": ["mcp", "serve"]
    }
  }
}
EOF
  log "wrote $mcp for gflow MCP"
}

install_runqd
configure_daemon
configure_cursor_mcp
log "gflow ready"
