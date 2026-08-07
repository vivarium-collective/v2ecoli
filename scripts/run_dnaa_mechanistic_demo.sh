#!/usr/bin/env bash
# Mechanistic DnaA-oriC replication-initiation demonstration runner.
#
# Runs the stepped-Adair / sat-init / daughter POST_INIT_UNLOCK mechanism
# (ported from feat/aim2-dnaa-oric-box-binding @ b577de4b) on a stock v2ecoli
# cache. Because the reference "apo+ATP kinetic" cache that seeded the bulk
# DnaA-ATP pool was never committed upstream, we drive the pool with the
# transparent demonstration production flux V2ECOLI_DNAA_ATP_PRODUCTION_PER_S
# (molecules/s) so the mechanism can be exercised and characterised.
#
# Usage:
#   scripts/run_dnaa_mechanistic_demo.sh CACHE_DIR EXP_ID SEED GENS MAX_MIN PROD_PER_S [OUT_DIR]
#
# Env vars below are the 19 load-bearing mechanism knobs + the production flux.
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_DIR="${1:?cache dir}"
EXP="${2:?experiment id}"
SEED="${3:-4}"
GENS="${4:-6}"
MAXMIN="${5:-180}"
PROD="${6:-6.0}"
OUT_DIR="${7:-out/${EXP}_parquet}"

echo "[demo] cache=$CACHE_DIR exp=$EXP seed=$SEED gens=$GENS max-min=$MAXMIN production=$PROD/s"

PYTHONUNBUFFERED=1 \
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
V2ECOLI_DNAA_ATP_PRODUCTION_PER_S="$PROD" \
.venv/bin/python -u scripts/run_condition_multigen_parquet.py \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --experiment-id "$EXP" \
  --generations "$GENS" --max-min "$MAXMIN" --seed "$SEED"
