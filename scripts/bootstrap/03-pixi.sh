#!/usr/bin/env bash
# Install Pixi into ~/.pixi (official installer).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_path_exports

if command -v pixi >/dev/null 2>&1; then
  log "pixi already installed: $(pixi --version)"
else
  log "installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
  ensure_path_exports
  need_cmd pixi
  log "pixi installed: $(pixi --version)"
fi

append_shell_path
log "pixi ready"
