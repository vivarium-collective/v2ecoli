#!/usr/bin/env bash
# Overnight fleet for the DnaA-oriC mechanistic-initiation investigation.
#
# Launches three study conditions, each over 4 seeds, with a concurrency cap:
#   1. succinate  mechanism (sat-init, production flux 0.2/s)
#   2. succinate  mass-clock CONTROL (inherited criticalMass heuristic)
#   3. basal      mechanism (sat-init, production flux 0.2/s)
#
# The control provides the synchronous baseline for the asynchrony contrast:
# under the mass clock a cell's oriCs fire together, under sat-init they fire
# with a spread. Production rate 0.2/s was calibrated so succinate divides at
# its natural ~85 min doubling with bulk DnaA-ATP peaking ~23 nM (reference
# sawtooth band) and clean one-initiation-per-cycle.
set -uo pipefail
cd "$(dirname "$0")/.."

MAXPAR="${MAXPAR:-4}"
GENS="${GENS:-6}"
PROD="${PROD:-0.2}"
SEEDS=(1 2 3 4)

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
)

run_one() {  # mode cache exp seed maxmin
  local mode="$1" cache="$2" exp="$3" seed="$4" maxmin="$5"
  local out="out/${exp}_parquet" log="out/${exp}.log"
  rm -rf "$out"
  if [ "$mode" = mech ]; then
    env "${MECH_ENV[@]}" \
        V2ECOLI_DNAA_ATP_PRODUCTION_PER_S="$PROD" \
        PYTHONUNBUFFERED=1 \
        .venv/bin/python -u scripts/run_condition_multigen_parquet.py \
          --cache-dir "$cache" --out-dir "$out" --experiment-id "$exp" \
          --generations "$GENS" --max-min "$maxmin" --seed "$seed" \
          > "$log" 2>&1
  else  # mass-clock control: no mechanism env, no production flux
    PYTHONUNBUFFERED=1 \
        .venv/bin/python -u scripts/run_condition_multigen_parquet.py \
          --cache-dir "$cache" --out-dir "$out" --experiment-id "$exp" \
          --generations "$GENS" --max-min "$maxmin" --seed "$seed" \
          > "$log" 2>&1
  fi
  echo "[fleet] DONE $exp (exit $?)"
}
# run_one and MECH_ENV are inherited directly by the ( ) & subshells below.

# Build the job list: "mode cache exp seed maxmin"
JOBS=()
for s in "${SEEDS[@]}"; do
  JOBS+=("mech  out/cache_dnaa5_oric        dnaa5_succ_mech_seed${s}   ${s} 100")
  JOBS+=("ctrl  out/cache_dnaa5_oric        dnaa5_succ_ctrl_seed${s}   ${s} 100")
  JOBS+=("mech  out/cache_dnaa5_oric_basal  dnaa5_basal_mech_seed${s}  ${s} 130")
done

echo "[fleet] ${#JOBS[@]} lineages, max ${MAXPAR} concurrent, gens=${GENS} prod=${PROD}/s"
pids=()
for job in "${JOBS[@]}"; do
  # throttle
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  # shellcheck disable=SC2086
  set -- $job
  ( run_one "$1" "$2" "$3" "$4" "$5" ) &
  echo "[fleet] launched $3 (seed $4)"
  sleep 2
done
wait
echo "[fleet] ALL COMPLETE"
