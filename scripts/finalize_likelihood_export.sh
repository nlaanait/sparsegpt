#!/bin/bash
# Re-stage latest likelihood JSON after jobs finish, commit, and push.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${HOME}/.pixi/bin:${PATH}"

branch=$(git branch --show-current)
echo "branch=$branch"
echo "queue:"
gqueue || true

git add figures/*/likelihood figures/*_wikitext2_cal/likelihood 2>/dev/null || true
git add figures/gemma_it_wikitext2_cal/likelihood 2>/dev/null || true

if git diff --cached --quiet; then
  echo "no new likelihood artifacts to commit"
else
  git commit -m "Update likelihood artifacts after remaining gflow jobs."
fi

echo "Attempting push to origin/$branch ..."
git push -u origin "$branch"
