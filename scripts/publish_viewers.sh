#!/usr/bin/env bash
# Publish docs/viewers/ (+ the baseline-viewer redirect) to the gh-pages branch.
# Surgical: replaces ONLY viewers/ and baseline-viewer/ on gh-pages; leaves
# dashboard/, investigations/, and the docs mirror untouched.
#
# Regenerate first:
#   PYTHONPATH=.:/path/to/bigraph-viz2/py .venv/bin/python scripts/regenerate_viewers.py
# Then publish:
#   bash scripts/publish_viewers.sh
set -euo pipefail
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS_ROOT"

if [ ! -f docs/viewers/index.html ]; then
  echo "::error:: docs/viewers/index.html missing — run scripts/regenerate_viewers.py first" >&2
  exit 1
fi

git fetch origin gh-pages
WT="$(mktemp -d)"
git worktree add "$WT" gh-pages
trap 'git worktree remove --force "$WT" 2>/dev/null || true' EXIT

rm -rf "$WT/viewers" "$WT/baseline-viewer"
cp -R docs/viewers "$WT/viewers"
cp -R docs/baseline-viewer "$WT/baseline-viewer"

cd "$WT"
git add viewers baseline-viewer
if git diff --cached --quiet; then
  echo "no viewer changes to publish"
else
  git -c user.name="github-actions[bot]" \
      -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
      commit -m "publish composite viewers hub"
  git push origin gh-pages
  echo "published docs/viewers/ -> gh-pages:viewers/"
fi
