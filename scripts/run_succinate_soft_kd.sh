#!/usr/bin/env bash
# Succinate variant of run_basal_soft_kd.sh — validates the soft K_d ramp
# discovered on basal against the milestone succinate lineage.
#
# Uses the canonical deep-burn-in dill (milestone_fromGen8_seed5_gen5.dill)
# to start from a proper stepped-Adair steady state.
#
# Usage: bash scripts/run_succinate_soft_kd.sh <TAG> <KDS_NM> <KD_MIN_NM> <SEED> [<SEED> ...]

set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: TAG KDS_NM KD_MIN_NM SEED [SEED ...]}"
KDS="${2:?}"
KD_MIN="${3:?}"
shift 3
SEEDS=("$@")
[ "${#SEEDS[@]}" -eq 0 ] && { echo "no seeds given"; exit 1; }

for SEED in "${SEEDS[@]}"; do
    EXP=succinate_v1.5_satinit_${TAG}_fromMilestoneGen5_seed${SEED}_12gen
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
      --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic \
      --out-dir out/${EXP}_parquet \
      --experiment-id "$EXP" \
      --generations 12 --max-min 180 --seed "$SEED" \
      --resume-dill out/steady_state_inputs/milestone_fromGen8_seed5_gen5.dill \
      > "out/${EXP}_run.log" 2>&1 &
    echo "launched tag=$TAG kds=$KDS seed=$SEED PID $!"
done
wait
