#!/usr/bin/env bash
# Run the MetabolismRedux 5-condition comparison ensemble (6 seeds x 1 gen) on the
# mini via Ray. Both engines per condition; genuine vEcoli via vivarium-process.
# Verify progress via the on-disk zarr stores, NOT this log (Ray buffers).
set -u
REPO="${REPO:-$HOME/code/v2e-redux-mini}"
VECOLI="${V2E_VECOLI_DIR:-$HOME/code/vEcoli}"
PY="$HOME/code/v2ecoli/.venv/bin/python"
OUTROOT="${OUTROOT:-$REPO/out/redux_5cond}"
SEEDS="${SEEDS:-6}"
RAYTHREADS="${V2E_RAY_THREADS:-4}"
CONDS=(basal with_aa succinate no_oxygen acetate)

cd "$REPO" || exit 1
mkdir -p "$OUTROOT"
export PYTHONPATH="$REPO" V2E_VECOLI_DIR="$VECOLI" V2E_RAY_THREADS="$RAYTHREADS"

run_engine() {  # $1=composite $2=cond $3=cachedir  [extra flags...]
  local comp="$1" cond="$2" cache="$3"; shift 3
  echo "[$(date +%H:%M:%S)] === $comp $cond (seeds=$SEEDS) ==="
  "$PY" scripts/run_comparison_ensemble.py \
    --composite "$comp" --condition "$cond" --cache-dir "$cache" \
    --n-seeds "$SEEDS" --max-generations 1 --max-steps 15000 --chunk 60 \
    --mode ray --from-vecoli-config "configs/metabolism_redux_${cond}.json" \
    "$@" --out-root "$OUTROOT/$cond" 2>&1
}

for cond in "${CONDS[@]}"; do
  echo "########## CONDITION $cond ##########"
  run_engine v2ecoli "$cond" "$REPO/out/cache_full" \
    --match-vecoli-simdata "$REPO/out/compare_harness/vecoli_parca/simData.cPickle"
  run_engine vecoli "$cond" "$REPO/out/compare_harness/vecoli_parca" \
    --vecoli-source vivarium-process
  echo "[$(date +%H:%M:%S)] condition $cond done; stores:"
  ls -d "$OUTROOT/$cond"/*.zarr 2>/dev/null | wc -l
done
echo "[$(date +%H:%M:%S)] ALL CONDITIONS COMPLETE"
