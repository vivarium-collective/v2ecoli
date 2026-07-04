#!/usr/bin/env bash
# Reproducible vEcoli<->v2ecoli growth-condition comparison sweep (post-fix).
#
# Runs the config-driven comparison harness across the 5 nutrient conditions
# (basal / with_aa / succinate / no_oxygen / acetate) at Chris-style population
# sampling (whatever n_init_sims x generations the configs specify), using a
# v2ecoli ParCa cache per condition built from a SINGLE shared sim_data so the
# only difference is the model, not the fit.
#
# Prereqs (env-overridable):
#   VECOLI_REPO     vEcoli checkout (default ~/code/vEcoli) — must hold the
#                   configs/cond_*.json and out/kb/simData.cPickle
#   VECOLI_SIMDATA  vEcoli simData.cPickle (default $VECOLI_REPO/out/kb/...)
#   PARCA_STATE     v2ecoli full ParCa state to build caches from. If unset and
#                   absent, a full v2ecoli-parca is run (~hours).
#   JAVA_HOME       a JDK 17 (Nextflow needs it). Auto-detected via brew if unset.
#   OUT             output dir (default out/cond_sweep)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

VECOLI_REPO=${VECOLI_REPO:-$HOME/code/vEcoli}
VECOLI_SIMDATA=${VECOLI_SIMDATA:-$VECOLI_REPO/out/kb/simData.cPickle}
OUT=${OUT:-out/cond_sweep}
PARCA_STATE=${PARCA_STATE:-$OUT/parca/parca_state.pkl}
JAVA_HOME=${JAVA_HOME:-$(/usr/libexec/java_home -v 17 2>/dev/null || echo /opt/homebrew/opt/openjdk@17)}
mkdir -p "$OUT"

echo ">> VECOLI_REPO=$VECOLI_REPO  JAVA_HOME=$JAVA_HOME  OUT=$OUT"
"$JAVA_HOME/bin/java" -version 2>&1 | head -1

# 1. Full-mode v2ecoli ParCa (only if we don't already have a state) -----------
if [ ! -f "$PARCA_STATE" ]; then
  echo ">> running full-mode v2ecoli ParCa (the translation-efficiency fix is in the code) ..."
  .venv/bin/v2ecoli-parca --mode full -c 8 -o "$OUT/parca" --cache-dir "$OUT/parca_cache"
  PARCA_STATE="$OUT/parca/parca_state.pkl"
fi

# 2. Build one per-condition v2 cache from that single shared fit ---------------
echo ">> building per-condition caches from $PARCA_STATE ..."
$PY - "$PARCA_STATE" "$OUT" <<'PY'
import sys, pickle, gzip
from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
from v2ecoli.core import save_sim_input
def ls(p):
    try: return load_parca_state(p)
    except Exception: return pickle.load((gzip.open if p.endswith('.gz') else open)(p, 'rb'))
state, out = sys.argv[1], sys.argv[2]
sd = hydrate_sim_data_from_state(ls(state))
MEDIA = {'basal': 'minimal', 'with_aa': 'minimal_plus_amino_acids',
         'succinate': 'minimal_succinate', 'no_oxygen': 'minimal_minus_oxygen',
         'acetate': 'minimal_acetate'}
for cond, media in MEDIA.items():
    save_sim_input(sd, bundle_dir=f'{out}/cache_{cond}', condition=cond, fixed_media=media)
    print(f'  built {out}/cache_{cond}')
PY

# 3. Run the harness across all 5 conditions, one matched cache each ------------
echo ">> running the 5-condition comparison sweep ..."
JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" $PY scripts/compare_harness.py \
  --config \
    "$VECOLI_REPO/configs/cond_basal.json" \
    "$VECOLI_REPO/configs/cond_with_aa.json" \
    "$VECOLI_REPO/configs/cond_succinate.json" \
    "$VECOLI_REPO/configs/cond_no_oxygen.json" \
    "$VECOLI_REPO/configs/cond_acetate.json" \
  --vecoli-repo "$VECOLI_REPO" \
  --vecoli-simdata "$VECOLI_SIMDATA" \
  --v2-cache \
    "$OUT/cache_basal" "$OUT/cache_with_aa" "$OUT/cache_succinate" \
    "$OUT/cache_no_oxygen" "$OUT/cache_acetate" \
  --workdir "$OUT/work" \
  -o "$OUT/conditions_report.html"

echo ">> DONE. Report: $OUT/conditions_report.html"
echo ">> Per-condition verdicts:"
for c in basal with-aa succinate no-oxygen acetate; do
  f="out/compare/report_card_verdict_cond-$c.json"
  [ -f "$f" ] && printf "   %-11s " "$c" && \
    $PY -c "import json;print(json.load(open('$f'))['overall'])" 2>/dev/null || true
done
