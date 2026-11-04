#!/usr/bin/env bash
# Apt packages needed by pixi builds, gflow, Hugging Face datasets, and git.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

log "installing Ubuntu packages"
run_root apt-get update -y
run_root DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  git-lfs \
  jq \
  libssl-dev \
  pkg-config \
  python3 \
  python3-venv \
  python3-pip \
  rsync \
  tmux \
  unzip \
  wget \
  xz-utils \
  zlib1g-dev

if ! command -v git-lfs >/dev/null 2>&1; then
  warn "git-lfs missing after apt install"
else
  git lfs install --skip-repo >/dev/null 2>&1 || true
fi

log "system packages ready"
