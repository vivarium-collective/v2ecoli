#!/usr/bin/env bash
# Sequentially run v8 (carry-forward relax) with three STUCK_THRESHOLD_S values.
set -u
cd "$(dirname "$0")/.."

for THRESH in 180 300 600; do
    EXP="dnaa5_adaptive_v8_thresh${THRESH}_succ_gen3_seed2"
    OUT="out/${EXP}_parquet"
    LOG="out/${EXP}_run.log"
    echo "[sweep] starting STUCK_THRESHOLD_S=${THRESH} @ $(date)" | tee "$LOG"
    V2ECOLI_DNAA_STUCK_THRESHOLD_S="$THRESH" uv run scripts/run_condition_multigen_parquet.py \
        --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic \
        --out-dir "$OUT" \
        --experiment-id "$EXP" \
        --generations 12 \
        --max-min 200 \
        --seed 2 \
        --resume-dill out/steady_state_inputs/succinate_default_gen3_start_dnaa3.dill \
        >> "$LOG" 2>&1
    echo "[sweep] done STUCK_THRESHOLD_S=${THRESH} @ $(date)" | tee -a "$LOG"
done
echo "[sweep] all three thresholds complete @ $(date)"
