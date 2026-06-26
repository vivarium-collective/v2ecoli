#!/usr/bin/env bash
# run_local_4x4x5.sh — LOCAL v2ecoli↔vEcoli comparison across all 5 conditions,
# both engines, with MATCHED-INITIAL-STATE seeding on the v2ecoli side (so rows
# reflect genuine dynamics, not stochastic low-copy sampling luck — see the
# SpoT/ppGpp artifact analysis). Produces per-condition zarr stores + a report
# card. This is the local dress-rehearsal for the cloud 4x4x5 → 16x16x5.
#
# Layout (matches comparison_report_card.py --pbg-vs-pbg):
#   <OUT>/<condition>/v2ecoli_seed<NN>.zarr   (matched-initial-state)
#   <OUT>/<condition>/vecoli_seed<NN>.zarr    (genuine vEcoli, vivarium-process)
#
# Usage: bash scripts/run_local_4x4x5.sh [SEEDS] [GENS] [MAXSTEPS] [OUT]
#   defaults: SEEDS=4 GENS=2 MAXSTEPS=4000 OUT=out/local_4x4x5
#   (MAXSTEPS caps steps/generation; a full generation divides ~2700 steps.)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${1:-4}"
GENS="${2:-4}"
# PER-GENERATION step budget (a generous, non-binding safety cap; real division by
# mass ends each generation well before it). The two runners interpret --max-steps
# DIFFERENTLY: v2ecoli's run_multigen_xarray treats it as a TOTAL across all gens,
# while vEcoli's run_vivarium_ecoli_pbg_multigen treats it as PER-generation. So we
# pass v2ecoli GENS×PER_GEN (total) and vEcoli PER_GEN (per-gen) — same per-gen
# budget for both, both divide naturally for GENS generations.
PER_GEN="${3:-15000}"
OUT="${4:-out/local_4x4x5}"
V2_MAXSTEPS=$(( GENS * PER_GEN ))   # v2ecoli: total cap
VE_MAXSTEPS="$PER_GEN"              # vEcoli: per-generation cap
# v2ecoli runs on v2's ParCa; genuine vEcoli runs on the UPSTREAM ParCa. They are
# SEPARATE caches — feeding the vEcoli engine v2's simData makes its FBA go
# negative (InvalidBoundaryError). The matched-initial-state reference is the
# upstream simData (so v2 starts from genuine-vEcoli initial counts).
CACHE="${V2E_CACHE:-out/cache_full}"                                  # v2ecoli ParCa
VE_CACHE="${V2E_VECOLI_CACHE:-out/compare_harness/vecoli_parca}"      # upstream ParCa
REF_SD="${V2E_REF_SIMDATA:-$VE_CACHE/simData.cPickle}"
: "${V2E_VECOLI_DIR:=/Users/eranagmon/code/vEcoli-upstream}"
export V2E_VECOLI_DIR PYTHONHASHSEED=0 PYTHONPATH="$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
# Override with V2E_CONDITIONS="basal acetate" for a partial/smoke run.
read -r -a CONDITIONS <<< "${V2E_CONDITIONS:-basal with_aa succinate no_oxygen acetate}"

echo "=== local 4x4x5: seeds=$SEEDS gens=$GENS per_gen=$PER_GEN (v2 total=$V2_MAXSTEPS, vE per-gen=$VE_MAXSTEPS) out=$OUT ==="
echo "    cache=$CACHE  ref_simData=$REF_SD  vecoli_dir=$V2E_VECOLI_DIR"
mkdir -p "$OUT"

run_engine() {  # composite cache_dir cond maxsteps extra_flags...
  local composite="$1" cache="$2" cond="$3" maxsteps="$4"; shift 4
  echo ">>> $composite / $cond  (cache=$cache, max_steps=$maxsteps)"
  "$PY" scripts/run_comparison_ensemble.py \
    --composite "$composite" --condition "$cond" --cache-dir "$cache" \
    --n-seeds "$SEEDS" --seed-start 0 --max-generations "$GENS" \
    --max-steps "$maxsteps" --chunk 60 --mode serial \
    --out-root "$OUT/$cond" "$@"
}

for cond in "${CONDITIONS[@]}"; do
  run_engine v2ecoli "$CACHE"    "$cond" "$V2_MAXSTEPS" --match-initial-state --match-vecoli-simdata "$REF_SD"
  run_engine vecoli  "$VE_CACHE" "$cond" "$VE_MAXSTEPS" --vecoli-source vivarium-process
done

# Build the per-condition map {cond: [<dir>, <dir>]} — both engines share the
# condition dir (report card --pbg-vs-pbg reads v2ecoli_seed*/vecoli_seed* there).
CJSON=$("$PY" - "$OUT" "${CONDITIONS[@]}" <<'PY'
import json, sys
out = sys.argv[1]; conds = sys.argv[2:]
print(json.dumps({c: [f"{out}/{c}", f"{out}/{c}"] for c in conds}))
PY
)
echo "=== report card (multi-seed local) ==="
"$PY" scripts/comparison_report_card.py --local-pbg-dir "$CJSON" \
  --local-pbg-seeds "$SEEDS" --only all -o "$OUT/report" || \
  echo "[warn] report card failed; per-condition zarr stores are under $OUT/<cond>/"
echo "=== done: stores under $OUT/<condition>/, report under $OUT/report ==="
