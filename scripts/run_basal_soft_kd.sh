#!/usr/bin/env bash
# Basal V=1.5 sat-init 12-gen with a SOFTER K_d ladder to test whether
# reduced cooperativity gives the ~60 s per-oriC Δt spread seen in Haochen's
# experiments. No chromblock. Same handoff dill as baseline.
#
# Usage: bash scripts/run_basal_soft_kd.sh <TAG> <KDS_NM> <KD_MIN_NM> <SEED> [<SEED> ...]
#   TAG       e.g. "soft1"  (goes into the experiment name)
#   KDS_NM    comma-separated K_d ladder, e.g. "100,100,50,30,20,20,20,20"
#   KD_MIN_NM min value in the ladder (used as the Langmuir floor)
#   SEED      one or more sim seeds

set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: TAG KDS_NM KD_MIN_NM SEED [SEED ...]}"
KDS="${2:?}"
KD_MIN="${3:?}"
shift 3
SEEDS=("$@")
[ "${#SEEDS[@]}" -eq 0 ] && { echo "no seeds given"; exit 1; }

for SEED in "${SEEDS[@]}"; do
    EXP=basal_v1.5_satinit_${TAG}_fromBurninGen10v1.0_seed${SEED}_12gen
    V2ECOLI_DNAA_ADAIR_KD=1 \
    V2ECOLI_DNAA_ADAIR_KDS_NM="$KDS" \
    V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100 \
    V2ECOLI_DNAA_ADAIR_KD_MIN_NM="$KD_MIN" \
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
    V2ECOLI_DNAA_CHROM_BLOCK_FRAC=0.0 \
    nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
      --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic_basal \
      --out-dir out/${EXP}_parquet \
      --experiment-id "$EXP" \
      --generations 12 --max-min 180 --seed "$SEED" \
      --resume-dill out/basal_v1.0_massclock_burnin_seed0_10gen/gen_dills/gen10.dill \
      > "out/${EXP}_run.log" 2>&1 &
    echo "launched tag=$TAG kds=$KDS seed=$SEED PID $!"
done
wait
