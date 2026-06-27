#!/bin/zsh
# Build the two-state 3D model (birth + pre-division) at the pinned parameters.
# Run from the v2ecoli worktree root. See REPRODUCE.md for the full context.
set -e

# ── config ───────────────────────────────────────────────────────────────
PARSIMONY_HOME=${PARSIMONY_HOME:-/Users/eranagmon/code/parsimony}
PY=${PY:-/Users/eranagmon/code/v2ecoli/.venv/bin/python}
TOP_N=${TOP_N:-400}
REUSE_STRUCT=${REUSE_STRUCT:-/Users/eranagmon/code/v2e-pdmp-refresh/out/ecoli3d_expanded/structures}
PARCA_CACHE=${PARCA_CACHE:-/Users/eranagmon/code/v2ecoli/out/cache}
export PARSIMONY_HOME

# ── prerequisites ────────────────────────────────────────────────────────
[ -x "$PARSIMONY_HOME/target/release/parsimony" ] || {
  echo "build the rust binary first: (cd $PARSIMONY_HOME && cargo build --release -p parsimony-cli)"; exit 1; }
mkdir -p out
[ -e out/cache ] || ln -s "$PARCA_CACHE" out/cache   # ParCa cache
# pre-seed structures cache to avoid net fetch (uncached species still fetch)
mkdir -p out/ecoli3d out/ecoli3d-div
[ -e out/ecoli3d/structures ] || cp -R "$REUSE_STRUCT" out/ecoli3d/structures
[ -e out/ecoli3d-div/structures ] || ln -s ../ecoli3d/structures out/ecoli3d-div/structures
rm -rf .parsimony/cache   # recipe-keyed — clear so geometry rebuilds

# ── build ────────────────────────────────────────────────────────────────
echo "==== BIRTH (snapshot) top_n=$TOP_N ===="; date
$PY -m v2ecoli.structural.build --out out/ecoli3d     --state snapshot --top-n "$TOP_N"
echo "==== DIVISION top_n=$TOP_N ===="; date
$PY -m v2ecoli.structural.build --out out/ecoli3d-div --state division --top-n "$TOP_N"
echo "ALL_BUILDS_DONE"; date
