#!/usr/bin/env bash
# Canonical launch script for the dnaa5 stepped-Adair milestone.
#
# Reproduces the 12-gen seed=1/seed=4 lineages whose stats/PDF were sent
# to Haochen and Suckjoon. All 19 env vars are load-bearing:
#
# - GRADIENT_GATE=1 populates the bulk-DnaA-ATP rolling window and enables
#   the gradient_rising computation; without it, gradient_rising defaults to
#   True and both COOP_GRADIENT_GATE and ADAPTIVE_KHALF become no-ops.
# - COOP_GRADIENT_GATE=1 uses gradient_rising to collapse the Adair ladder
#   to Langmuir K_d_max when bulk isn't rising — the primary cooperativity
#   gate for the parent oriC.
# - POST_INIT_UNLOCK_S=60 is the SeqA proxy on daughter oriCs.
# - Sat-init fires when the cluster hits 8/8 sustained for 1 s.
#
# Usage: bash scripts/run_milestone_dnaa5_stepped_adair.sh [SEED [SEED ...]]
#        (default seeds: 1 4)

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-1 4}")

for SEED in "${SEEDS[@]}"; do
    EXP=dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed${SEED}_12gen
    V2ECOLI_DNAA_ADAIR_KD=1 \
    V2ECOLI_DNAA_ADAIR_KDS_NM=100,100,50,10,3,3,3,3 \
    V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100 \
    V2ECOLI_DNAA_ADAIR_KD_MIN_NM=3 \
    V2ECOLI_DNAA_ADAPTIVE_KHALF=1 \
    V2ECOLI_DNAA_COOP_GRADIENT_GATE=1 \
    V2ECOLI_DNAA_COOP_STUCK_GATE=0 \
    V2ECOLI_DNAA_GRADIENT_GATE=1 \
    V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S=0.05 \
    V2ECOLI_DNAA_GRADIENT_WINDOW_S=120 \
    V2ECOLI_DNAA_HILL_CONC=0 \
    V2ECOLI_DNAA_HILL_KD=0 \
    V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN=0.025 \
    V2ECOLI_DNAA_KHALF_STUCK_THRESHOLD_S=300 \
    V2ECOLI_DNAA_KINETIC_ORIC_LOW=0 \
    V2ECOLI_DNAA_POST_INIT_UNLOCK_S=60 \
    V2ECOLI_DNAA_RELAX_SNAP=0 \
    V2ECOLI_SATURATION_SUSTAINED_S=1 \
    V2ECOLI_SATURATION_TRIGGERED_INIT=1 \
    nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
      --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic \
      --out-dir out/${EXP}_parquet \
      --experiment-id "$EXP" \
      --generations 12 --max-min 180 --seed "$SEED" \
      --resume-dill out/dnaa5_v1.5_hillKd_h4_K3_seed4/gen_dills/gen5.dill \
      > "out/${EXP}_run.log" 2>&1 &
    echo "launched seed=$SEED PID $!"
done
wait
