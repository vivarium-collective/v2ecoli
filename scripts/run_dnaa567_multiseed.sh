#!/usr/bin/env bash
# Multi-seed robustness for the dnaa-5/6/7 headline chain.
# Reproduces the seed-1 operating points (documented in the study yamls) at a new
# seed, and PERSISTS the full env recipe into each run log — the original seed-1
# runs were launched inline and never recorded their DNAA_*/RIDA_* env.
#
# Usage: scripts/run_dnaa567_multiseed.sh <seed>
# Runs the three studies sequentially for one seed. Launch one process per seed
# to parallelise across seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:?usage: run_dnaa567_multiseed.sh <seed>}"
PY=.venv/bin/python
RUNNER=scripts/run_condition_multigen_parquet.py
CACHE=out/cache_dnaa4_s06_F05
GENS=8
PERT="TU00259[c]=0.0015"

run() {  # $1=name ; recipe passed via the calling env
  local name="$1"
  echo "================ $name (seed $SEED) ================"
  echo "RECIPE: $(env | grep -E '^(DNAA_|RIDA_|DDAH_|DARS)' | sort | tr '\n' ' ')"
  $PY "$RUNNER" \
    --cache-dir "$CACHE" --generations "$GENS" --seed "$SEED" \
    --perturbation "$PERT" \
    --experiment-id "$name" --out-dir "out/$name"
}

# ---- dnaa-5: cooperative oriC-low switch (n=4, K=30), MASS trigger, RIDA/DDAH/DARS off ----
DNAA_ORIC_COOPERATIVITY_N=4 DNAA_ORIC_HALF_SAT_nM=30 \
RIDA_RATE_MULTIPLIER=0 DDAH_RATE_MULTIPLIER=0 DARS1_RATE_MULTIPLIER=0 DARS2_RATE_MULTIPLIER=0 \
  run "dnaa5_coop_n4_k30_seed${SEED}_8gen"

# ---- dnaa-6: mechanistic trigger on the oriC-LOW cooperative switch, RIDA/DDAH/DARS off ----
DNAA_INITIATION_TRIGGER=mechanistic DNAA_INIT_TRIGGER_POOL=low DNAA_INIT_LOW_THRESHOLD=6 \
DNAA_ORIC_COOPERATIVITY_N=4 DNAA_ORIC_HALF_SAT_nM=30 \
RIDA_RATE_MULTIPLIER=0 DDAH_RATE_MULTIPLIER=0 DARS1_RATE_MULTIPLIER=0 DARS2_RATE_MULTIPLIER=0 \
  run "dnaa6_mech_low_coop_n4_seed${SEED}_8gen"

# ---- dnaa-7: same mechanistic trigger + RIDA operating point (0.6), DDAH/DARS on ----
DNAA_INITIATION_TRIGGER=mechanistic DNAA_INIT_TRIGGER_POOL=low DNAA_INIT_LOW_THRESHOLD=6 \
DNAA_ORIC_COOPERATIVITY_N=4 DNAA_ORIC_HALF_SAT_nM=30 \
RIDA_RATE_MULTIPLIER=0.6 DDAH_RATE_MULTIPLIER=1.0 DARS1_RATE_MULTIPLIER=1.0 DARS2_RATE_MULTIPLIER=1.0 \
  run "dnaa7_ridacal_r06_seed${SEED}_8gen"

echo "================ DONE seed $SEED ================"
