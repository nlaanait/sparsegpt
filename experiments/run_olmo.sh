#!/bin/bash
# Dense WikiText2 smoke for all OLMo 2 Instruct sizes, then the C4-calibrated
# sparsity sweep (WikiText2 + PTB). Does not start itself; run explicitly.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/olmo2/logs
exec > results/olmo2/logs/driver.log 2>&1

echo "olmo2 smoke+sweep $(date)"
    pixi run python -u -m drivers.olmo2_sparsity
echo "done $(date)"
