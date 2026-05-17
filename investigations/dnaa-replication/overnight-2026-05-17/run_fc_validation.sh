#!/usr/bin/env bash
# Phase 2: validate (TE=20×, fc=0.7) sweet spot with additional seeds.
# Also probe adjacent fc values (0.6, 0.8) to map the optimum region.
# 5 sims at fc=0.7 (seeds 2,3,4), 3 sims at fc=0.6 (seeds 0,1,2), 3 sims at fc=0.8 (seeds 0,1,2).
# Total: 9 sims, ~13 min wall.

set -e
cd /Users/eranagmon/code/v2ecoli

RUNNER="investigations/dnaa-replication/overnight-2026-05-17/run_baseline_with_fc.py"
LOG="investigations/dnaa-replication/overnight-2026-05-17/fc_test_log.txt"
echo "=== fc-validation launched $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

run_one() {
  local te=$1
  local fc=$2
  local seed=$3
  local name="baseline-te${te}x-fc${fc}-seed${seed}"
  echo "[$(date -u +%H:%M:%S)] start $name" >> "$LOG"
  if .venv/bin/python "$RUNNER" \
       --name "$name" \
       --seed "$seed" \
       --duration_min 10 \
       --dnaa_te_multiplier "$te" \
       --dnaa_autorep_multiplier "$fc" \
       >> "$LOG" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] DONE $name" >> "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $name" >> "$LOG"
  fi
}

# Sweet-spot top-up: TE=20×, fc=0.7, seeds 2,3,4
for seed in 2 3 4; do
  run_one 20 0.7 "$seed"
done

# Adjacent: TE=20×, fc∈{0.6, 0.8}, seeds 0,1,2
for fc in 0.6 0.8; do
  for seed in 0 1 2; do
    run_one 20 "$fc" "$seed"
  done
done

echo "=== fc-validation finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
