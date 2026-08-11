#!/usr/bin/env bash
# Soft-grad K_d ladder (100,80,60,40,25,15,8,3 nM) + chromosomal DnaA-box
# blocking. Combines the winning soft-grad ramp with the chromblock
# perturbation to test whether chromosomal boxes matter for synchrony under
# a softer cooperativity.
#
# Usage: bash scripts/run_basal_softgrad_chromblock.sh <FRAC> <SEED> [<SEED> ...]

set -euo pipefail
cd "$(dirname "$0")/.."

FRAC="${1:?usage: FRAC SEED [SEED ...]}"
shift
SEEDS=("$@")
[ "${#SEEDS[@]}" -eq 0 ] && { echo "no seeds given"; exit 1; }

TAG="softgrad_chromblock$(python3 -c "print(int(float('${FRAC}')*100))")"

for SEED in "${SEEDS[@]}"; do
    EXP=basal_v1.5_satinit_${TAG}_fromBurninGen10v1.0_seed${SEED}_12gen
    V2ECOLI_DNAA_ADAIR_KD=1 \
    V2ECOLI_DNAA_ADAIR_KDS_NM=100,80,60,40,25,15,8,3 \
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
    V2ECOLI_DNAA_CHROM_BLOCK_FRAC="$FRAC" \
    V2ECOLI_DNAA_CHROM_BLOCK_SEED="$SEED" \
    nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
      --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic_basal \
      --out-dir out/${EXP}_parquet \
      --experiment-id "$EXP" \
      --generations 12 --max-min 180 --seed "$SEED" \
      --resume-dill out/basal_v1.0_massclock_burnin_seed0_10gen/gen_dills/gen10.dill \
      > "out/${EXP}_run.log" 2>&1 &
    echo "launched frac=$FRAC seed=$SEED PID $!"
done
wait
