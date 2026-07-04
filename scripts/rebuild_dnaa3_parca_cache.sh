#!/usr/bin/env bash
# dnaa-3 in-sim build: rebuild ParCa + cache for the composite WITH the
# DnaaBoxBinding step wired (early layer). Adding the step changed the composite
# inputs_hash, so the dnaa-2 cache is stale — this builds a matching cache, then
# runs a 1-gen validation to confirm the step FIRES + emits occupancy + the cycle
# is preserved.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=../../../.venv/bin/python
SD=out/sim_data_dnaa3
CACHE=out/cache_dnaa3
DILL=out/steady_state_inputs/succinate_default_gen3_start.dill

echo "================ [1/4] ParCa --mode full ================"
rm -rf "$SD"; mkdir -p "$SD"
$PY scripts/parca_run.py --mode full --cpus 8 -o "$SD"
gzip -c "$SD/parca_state.pkl" > "$SD/parca_state.pkl.gz"

echo "================ [2/4] build cache (dnaa-2 perturbation) ================"
$PY - "$SD" "$CACHE" <<'PYEOF'
import sys, os, shutil
sys.path.insert(0, ".")
from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
sd_dir, cache = sys.argv[1], sys.argv[2]
if os.path.exists(cache): shutil.rmtree(cache)
sim = hydrate_sim_data_from_state(load_parca_state(f"{sd_dir}/parca_state.pkl.gz"))
sim.genetic_perturbations = {"TU00259[c]": 1e-3}
save_sim_input(sim, cache, condition="succinate", fixed_media="minimal_succinate")
write_cache_version(cache, repo_root=".")
import json
print("cache built:", cache, "| inputs_hash:",
      json.load(open(f"{cache}/cache_version.json")).get("inputs_hash", "?")[:16])
PYEOF

echo "================ [3/4] validation run (1 gen, emit unique) ================"
rm -rf out/dnaa3_insim_validate
V2ECOLI_EMIT_UNIQUE=1 $PY scripts/run_condition_multigen_parquet.py \
  --cache-dir "$CACHE" --resume-dill "$DILL" \
  --start-gen 1 --generations 1 --max-min 90 --seed 1 \
  --out-dir out/dnaa3_insim_validate --experiment-id dnaa3_insim --dill-dir out/dnaa3_insim_validate/gen_dills

echo "================ [4/4] check: did the step FIRE + emit occupancy? cycle ok? ================"
$PY - <<'PYEOF'
import polars as pl, glob
fs=sorted(glob.glob('out/dnaa3_insim_validate/**/history/**/*.pq', recursive=True))
fs=[f for f in fs if '/agent_id=0/' in f]
if not fs: print("NO parquet — run failed"); raise SystemExit
d=pl.scan_parquet(fs[-1]).tail(1).collect(); cols=d.columns
bind=[c for c in cols if 'dnaA_binding' in c]
g=lambda c: d['listeners__dnaA_binding__'+c][0] if 'listeners__dnaA_binding__'+c in cols else 'MISSING'
print("dnaA_binding columns present:", len(bind))
print("  free DnaA-ATP nM:", g('free_DnaA_ATP_nM'))
print("  oriC bound/total:", g('oric__n_bound'), "/", g('oric__n_total'), "| low-aff occ:", g('oric__low_affinity_occupied'))
print("  chromosomal:", g('chromosome__occupied_count'), "/", g('chromosome__n_total'))
print("CYCLE: oriC =", d['listeners__replication_data__number_of_oric'][0],
      "| cell_mass =", round(d['listeners__mass__cell_mass'][0],1))
fired = g('chromosome__n_total') not in ('MISSING', 0)
print("=> STEP FIRED + populated:" , fired)
PYEOF
echo "================ DONE ================"
