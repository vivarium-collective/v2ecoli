#!/usr/bin/env bash
# Focused TE sweep: fill in the 15x..30x inflection region with 5 seeds
# per multiplier so the autorepression-correlation r value gets a stable
# multi-seed pool. Runs serially in background. ~22 min wall.
#
# Output: appends to studies/dnaa-01-expression-dynamics/runs.db.
# Each sim is named baseline-te{N}x-seed{S}.

set -e
cd /Users/eranagmon/code/v2ecoli

RUNNER="studies/dnaa-01-expression-dynamics/sims/run_baseline.py"
LOG="investigations/dnaa-replication/overnight-2026-05-17/sweep_log.txt"
echo "=== focused-sweep launched $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

run_one() {
  local mult=$1
  local seed=$2
  local name="baseline-te${mult}x-seed${seed}"
  echo "[$(date -u +%H:%M:%S)] start $name" >> "$LOG"
  if .venv/bin/python "$RUNNER" \
       --name "$name" \
       --seed "$seed" \
       --duration_min 10 \
       --dnaa_te_multiplier "$mult" \
       --heavy_tf_listener \
       >> "$LOG" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE $name" >> "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $name" >> "$LOG"
  fi
}

# 15× × 5 seeds, 25× × 5 seeds, 30× × 5 seeds (15 new). Then top up 10× and 20× to 5 seeds each (4 more).
for mult in 15 25 30; do
  for seed in 0 1 2 3 4; do
    run_one "$mult" "$seed"
  done
done

# Top up existing
for mult in 10 20; do
  for seed in 3 4; do
    run_one "$mult" "$seed"
  done
done

echo "=== sweep finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
