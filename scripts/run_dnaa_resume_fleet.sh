#!/usr/bin/env bash
# Authentic DnaA-oriC reproduction fleet — resumes the REAL reference burn-in
# dill (dnaa5_v1.5_hillKd_h4_K3_seed4_gen5.dill, extracted from
# origin/feat/aim2-dnaa-oric-box-binding) instead of the constant-production
# stand-in. The dill carries the real oriC labelling (24 low / 9 high boxes)
# and a DnaA-ATP pool replenished by real DnaA translation, so the production
# flux is turned OFF here.
#
# NOTE: all runs resume the SAME seed-4 burn-in state; --seed varies only the
# forward stochastic trajectory (the matching apo+ATP kinetic cache and per-seed
# dills were never committed).
set -uo pipefail
cd "$(dirname "$0")/.."

MAXPAR="${MAXPAR:-3}"
GENS="${GENS:-6}"
MAXMIN="${MAXMIN:-110}"
SEEDS=(1 2 3 4)
DILL=out/steady_state_inputs/dnaa5_v1.5_hillKd_h4_K3_seed4_gen5.dill

MECH_ENV=(
  V2ECOLI_DNAA_ADAIR_KD=1
  V2ECOLI_DNAA_ADAIR_KDS_NM=100,100,50,10,3,3,3,3
  V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100
  V2ECOLI_DNAA_ADAIR_KD_MIN_NM=3
  V2ECOLI_DNAA_ADAPTIVE_KHALF=1
  V2ECOLI_DNAA_COOP_GRADIENT_GATE=1
  V2ECOLI_DNAA_COOP_STUCK_GATE=0
  V2ECOLI_DNAA_GRADIENT_GATE=1
  V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S=0.05
  V2ECOLI_DNAA_GRADIENT_WINDOW_S=120
  V2ECOLI_DNAA_HILL_CONC=0
  V2ECOLI_DNAA_HILL_KD=0
  V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN=0.025
  V2ECOLI_DNAA_KHALF_STUCK_THRESHOLD_S=300
  V2ECOLI_DNAA_KINETIC_ORIC_LOW=0
  V2ECOLI_DNAA_POST_INIT_UNLOCK_S=60
  V2ECOLI_DNAA_RELAX_SNAP=0
  V2ECOLI_SATURATION_SUSTAINED_S=1
  V2ECOLI_SATURATION_TRIGGERED_INIT=1
  V2ECOLI_DNAA_ATP_PRODUCTION_PER_S=0   # OFF — the dill has a real, translation-fed pool
)

run_one() {  # mode seed
  local mode="$1" seed="$2"
  local exp="dnaa5_resume_${mode}_seed${seed}"
  local out="out/${exp}_parquet" log="out/${exp}.log"
  rm -rf "$out"
  if [ "$mode" = mech ]; then
    env "${MECH_ENV[@]}" PYTHONUNBUFFERED=1 \
      .venv/bin/python -u scripts/run_condition_multigen_parquet.py \
        --cache-dir out/cache_dnaa5_oric --out-dir "$out" --experiment-id "$exp" \
        --generations "$GENS" --max-min "$MAXMIN" --seed "$seed" \
        --resume-dill "$DILL" > "$log" 2>&1
  else  # mass-clock control: same burn-in, mechanism OFF
    PYTHONUNBUFFERED=1 \
      .venv/bin/python -u scripts/run_condition_multigen_parquet.py \
        --cache-dir out/cache_dnaa5_oric --out-dir "$out" --experiment-id "$exp" \
        --generations "$GENS" --max-min "$MAXMIN" --seed "$seed" \
        --resume-dill "$DILL" > "$log" 2>&1
  fi
  echo "[resume-fleet] DONE $exp (exit $?)"
}

[ -f "$DILL" ] || { echo "MISSING dill: $DILL"; exit 1; }
JOBS=()
for s in "${SEEDS[@]}"; do JOBS+=("mech $s" "ctrl $s"); done
echo "[resume-fleet] ${#JOBS[@]} runs, max ${MAXPAR} concurrent, gens=${GENS}"
for job in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  # shellcheck disable=SC2086
  set -- $job
  ( run_one "$1" "$2" ) &
  echo "[resume-fleet] launched dnaa5_resume_$1_seed$2"
  sleep 2
done
wait
echo "[resume-fleet] ALL COMPLETE"
