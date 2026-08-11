#!/usr/bin/env bash
# Basal (minimal glucose) V=1.5 sat-init 12-gen from V=1.0 mass-clock gen10 handoff.
# Same milestone config as scripts/run_milestone_dnaa5_stepped_adair.sh but on the
# basal cache + basal handoff dill. RIDA left OFF (default).
#
# Usage: bash scripts/run_basal_v1.5_satinit_from_burnin.sh [SEED [SEED ...]]

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-1 2}")

for SEED in "${SEEDS[@]}"; do
    EXP=basal_v1.5_satinit_fromBurninGen10v1.0_seed${SEED}_12gen
    V2ECOLI_DNAA_ADAIR_KD=1 \
    V2ECOLI_DNAA_ADAIR_KDS_NM=100,100,50,10,3,3,3,3 \
    V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100 \
    V2ECOLI_DNAA_ADAIR_KD_MIN_NM=3 \
    V2ECOLI_DNAA_COOP_GRADIENT_GATE=1 \
    V2ECOLI_DNAA_GRADIENT_GATE=1 \
    V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S=0.05 \
    V2ECOLI_DNAA_GRADIENT_WINDOW_S=120 \
    V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN=0.025 \
    V2ECOLI_DNAA_POST_INIT_UNLOCK_S=60 \
    V2ECOLI_SATURATION_SUSTAINED_S=1 \
    V2ECOLI_SATURATION_TRIGGERED_INIT=1 \
    V2ECOLI_DNAA_RIDA_ENABLED=0 \
    V2ECOLI_RIDA_COMPLEX_FORK_GATE=0 \
    nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
      --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic_basal \
      --out-dir out/${EXP}_parquet \
      --experiment-id "$EXP" \
      --generations 12 --max-min 180 --seed "$SEED" \
      --resume-dill out/basal_v1.0_massclock_burnin_seed0_10gen/gen_dills/gen10.dill \
      > "out/${EXP}_run.log" 2>&1 &
    echo "launched seed=$SEED PID $!"
done
wait
