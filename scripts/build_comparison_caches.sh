#!/usr/bin/env bash
# Build BOTH condition-complete ParCa caches for the v2ecoli <-> vEcoli pbg-vs-pbg
# comparison, deterministically and idempotently — the single source of truth. No
# --mode fast fixtures (basal-only), no AI-in-the-loop. Re-running is a no-op when
# both caches are already present.
#
# Why TWO caches (not one shared): the two engines have DIVERGENT ParCa formats —
# v2ecoli needs initial_state.json + v2-specific fields (e.g. pool_label); upstream
# needs its kb/simData.cPickle layout. Confirmed empirically: v2 cannot load the
# upstream cache ("missing initial_state.json" / "no field pool_label"). So each
# engine gets its OWN full ParCa, both built here from code.
#
# Why FULL (not the shipped fixture): models/parca/parca_state.pkl.gz is --mode FAST
# (reduced TF condition set). On it, v2ecoli's per-condition regen finds no condition
# data and silently runs BASAL for every media — which read as a 44-336% v2-vs-vEcoli
# "divergence" that was pure setup artifact. --mode full fits all conditions.
#
# Usage:  bash scripts/build_comparison_caches.sh
# Env:    V2_CACHE (default out/cache_full), UP_CACHE (default the upstream cache),
#         V2E_VECOLI_DIR (the pristine upstream vEcoli fork), V2_PARCA_CPUS (default 8).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONPATH="$PWD"

V2_CACHE="${V2_CACHE:-out/cache_full}"
UP_CACHE="${UP_CACHE:-out/compare_harness/vecoli_parca}"
FORK="${V2E_VECOLI_DIR:-/Users/eranagmon/code/vEcoli-upstream}"
V2_PARCA_CPUS="${V2_PARCA_CPUS:-8}"
CONDS=(basal with_aa succinate no_oxygen acetate)

echo "[build-caches] (1/3) v2ecoli FULL ParCa -> $V2_CACHE"
if [ -f "$V2_CACHE/simData.cPickle" ] && [ -f "$V2_CACHE/initial_state.json" ]; then
  echo "       present — skipping (rm -rf $V2_CACHE to rebuild)"
else
  # Two deterministic steps (parca_run's --cache-dir only writes the ParCa STATE,
  # not the cache bundle): (a) run the full ParCa -> parca_state.pkl (~2.5 min, v2
  # wholecell Cython is compiled so parallel is safe); (b) materialise the cache
  # bundle (initial_state.json + simData.cPickle, all conditions) from it.
  PSTATE=out/sim_data_full/parca_state.pkl
  if [ ! -f "$PSTATE" ]; then
    .venv/bin/python scripts/parca_run.py --mode full -o out/sim_data_full --cpus "$V2_PARCA_CPUS"
  fi
  gzip -kf "$PSTATE"                       # build_cache expects a gzipped fixture
  .venv/bin/python scripts/build_cache.py --fixture "$PSTATE.gz" --cache "$V2_CACHE"
fi

echo "[build-caches] (2/3) upstream-vEcoli FULL ParCa -> $UP_CACHE  (serial --cpus 1)"
# --cpus 1 is REQUIRED, not a perf knob: the pristine upstream checkout ships
# uncompiled Cython, so parallel pool workers re-import it and respawn-loop (the
# >1h hang in sms-api #147 / F3). The serial path keeps the main-process compiled pin.
if [ -f "$UP_CACHE/simData.cPickle" ]; then
  echo "       present — skipping (rm -rf $UP_CACHE to rebuild)"
else
  V2E_VECOLI_DIR="$FORK" .venv/bin/python scripts/build_upstream_parca.py \
    --cpus 1 --copy-to "$UP_CACHE"
fi

echo "[build-caches] (3/3) verifying v2 ParCa condition-completeness (fail loud)"
.venv/bin/python - "$V2_CACHE" "${CONDS[@]}" <<'PY'
import sys, pickle
cache, conds = sys.argv[1], sys.argv[2:]
with open(f"{cache}/simData.cPickle", "rb") as f:
    sd = pickle.load(f)
fitted = set(getattr(sd, "condition_to_doubling_time", {}) or {})
print(f"[build-caches] v2 fitted conditions ({len(fitted)}): {sorted(fitted)}")
missing = [c for c in conds if c not in fitted]
if missing:
    raise SystemExit(
        f"[build-caches] FAIL: v2 ParCa {cache} is missing conditions {missing} "
        f"(is it --mode fast?). The comparison needs a condition-complete cache.")
print("[build-caches] OK — all comparison conditions are fitted.")
PY

echo "[build-caches] DONE. Comparison caches ready (single source of truth):"
echo "    v2ecoli : $V2_CACHE"
echo "    upstream: $UP_CACHE"
echo "  Run the comparison with --cache-dir $V2_CACHE (v2ecoli) / $UP_CACHE (vecoli)."
