#!/usr/bin/env bash
# Rebuild the dnaa-1 (Mechanism-A, V=1.7e-3) ParCa cache and run the canonical
# 7-generation seed-0 succinate lineage, regenerating the parquet that was lost
# from studies/dnaa-1-expression/parquet-runs/. Mirrors rebuild_dnaa3_parca_cache.sh;
# only the dnaA promoter perturbation (TU00259[c]=0.0017, the 1.7e-3 canonical),
# generations (7), seed (0), and out paths differ.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY="$HOME/code/v2ecoli/.venv/bin/python"
SD=out/sim_data_dnaa1
CACHE=out/cache_dnaa1_mechA_1.7e-3
DILL=out/steady_state_inputs/succinate_default_gen3_start.dill
OUT=studies/dnaa-1-expression/parquet-runs/dnaa1-mechA-1.7e-3-7gen
EXP=dnaa1_mechA_1p7e-3_7gen

[ -f "$DILL" ] || { echo "FATAL: resume dill missing: $DILL"; exit 1; }

echo "================ [1/3] ParCa --mode full ================"
rm -rf "$SD"; mkdir -p "$SD"
$PY scripts/parca_run.py --mode full --cpus 8 -o "$SD"
gzip -c "$SD/parca_state.pkl" > "$SD/parca_state.pkl.gz"

echo "================ [2/3] build cache (dnaa-1 mechA perturbation TU00259[c]=0.0017) ================"
$PY - "$SD" "$CACHE" <<'PYEOF'
import sys, os, shutil, json
sys.path.insert(0, ".")
from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
sd_dir, cache = sys.argv[1], sys.argv[2]
if os.path.exists(cache):
    shutil.rmtree(cache)
sim = hydrate_sim_data_from_state(load_parca_state(f"{sd_dir}/parca_state.pkl.gz"))
sim.genetic_perturbations = {"TU00259[c]": 0.0017}   # mechA canonical V=1.7e-3
save_sim_input(sim, cache, condition="succinate", fixed_media="minimal_succinate")
write_cache_version(cache, repo_root=".")
print("cache built:", cache, "| inputs_hash:",
      json.load(open(f"{cache}/cache_version.json")).get("inputs_hash", "?")[:16])
PYEOF

echo "================ [3/3] run canonical 7-gen seed-0 lineage ================"
rm -rf "$OUT"
V2ECOLI_EMIT_UNIQUE=1 $PY scripts/run_condition_multigen_parquet.py \
  --cache-dir "$CACHE" --resume-dill "$DILL" \
  --start-gen 1 --generations 7 --max-min 120 --seed 0 \
  --out-dir "$OUT" --experiment-id "$EXP" --dill-dir "$OUT/gen_dills"
echo "DONE: parquet at $OUT"
