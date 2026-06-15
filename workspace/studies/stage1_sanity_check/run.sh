#!/usr/bin/env bash
# Stage 1 sanity-check study — full pipeline orchestrator.
#
# Phases:
#   1. Full ParCa rerun with overrides     (~4-8 hr)
#   2. Gzip-promote + build cache (acetate) (~2 min)
#   3. Multi-gen sim, 1 seed × 5 gens       (~12 hr at 136-min doubling)
#   4. Generate diagnostic plots + report   (~minutes)
#
# Pause/resume between phases is fine — each writes its outputs to disk
# and the next phase reads from them. Comment out completed phases to
# skip.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

SIM_DATA_DIR="out/sim_data_stage1_final"
CACHE_DIR="out/cache_stage1"
REPORT_OUT="docs/stage1_sanity_check.html"
CPUS="${CPUS:-8}"
GENERATIONS="${GENERATIONS:-5}"
SEED="${SEED:-0}"
MAX_DURATION="${MAX_DURATION:-12000}"

echo "============================================================"
echo "  Stage 1 sanity-check study"
echo "  cpus=$CPUS  gens=$GENERATIONS  seed=$SEED  max_dur=$MAX_DURATION"
echo "============================================================"

# --- Phase 1 — ParCa rerun with in-memory overrides --------------------
# STAGE1_OVERRIDE_SKIP=d_period is the canonical config.  See README.md
# "Why D=30 is dropped" — the d_period+replisome_rate pair breaks Step 6.
echo "[$(date +%H:%M:%S)] Phase 1: ParCa rerun (~4-8 hr)"
STAGE1_OVERRIDE_SKIP=d_period v2ecoli-parca \
    --mode full \
    --cpus "$CPUS" \
    -o "$SIM_DATA_DIR" \
    --overrides-module studies.stage1_sanity_check.overrides

# --- Phase 2 — Promote + build cache at acetate condition -------------
echo "[$(date +%H:%M:%S)] Phase 2: gzip-promote + build_cache (acetate)"
gzip -c "$SIM_DATA_DIR/parca_state.pkl" > "$SIM_DATA_DIR/parca_state.pkl.gz"
python scripts/build_cache.py \
    --fixture "$SIM_DATA_DIR/parca_state.pkl.gz" \
    --cache   "$CACHE_DIR" \
    --condition acetate

# --- Phase 3 — Multi-gen simulation -----------------------------------
echo "[$(date +%H:%M:%S)] Phase 3: multi-gen sim (1 seed × $GENERATIONS gens)"
python studies/stage1_sanity_check/run_sim.py \
    --generations  "$GENERATIONS" \
    --seed         "$SEED" \
    --cache-dir    "$CACHE_DIR" \
    --max-duration "$MAX_DURATION" \
    --db-path      out/stage1_sanity_runs.db

echo "[$(date +%H:%M:%S)] Done. SQLite history at out/stage1_sanity_runs.db"
echo "                    Generate report next via Phase 4 plotting script."
