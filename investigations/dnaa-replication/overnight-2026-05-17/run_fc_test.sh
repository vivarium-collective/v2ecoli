#!/usr/bin/env bash
# Quick test of the joint (TE × fc) sweep.
# Start with a small grid: TE=20× × fc∈{0.3, 0.5, 0.7, 1.0, 2.0} × 2 seeds = 10 sims.
# If fc<1.0 substantially improves the picture, then we run a full sweep.

set -e
cd /Users/eranagmon/code/v2ecoli

RUNNER="investigations/dnaa-replication/overnight-2026-05-17/run_baseline_with_fc.py"
LOG="investigations/dnaa-replication/overnight-2026-05-17/fc_test_log.txt"
echo "=== fc-test launched $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

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

# Pilot: TE=20× × fc ∈ {0.3, 0.5, 0.7, 1.0, 2.0} × seeds 0,1
for fc in 0.3 0.5 0.7 2.0; do
  for seed in 0 1; do
    run_one 20 "$fc" "$seed"
  done
done

# Also try TE=30× × fc=0.3 (since 30× alone collapses)
for seed in 0 1; do
  run_one 30 0.3 "$seed"
done

echo "=== fc-test finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
