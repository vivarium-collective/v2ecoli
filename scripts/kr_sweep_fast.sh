#!/usr/bin/env bash
# FAST k_r sweep — confirm the dnaa-2 fraction-drift lever WITHOUT ParCa rebuilds.
# Reuses ONE ParCa state (out/sim_data_dnaa2/parca_state.pkl.gz) and overrides the
# DnaA-ADP reverse rate (MONOMER0-4565_RXN, equilibrium rxn index 27) post-hydration
# via eq.set_rev_rate, then rebuilds only the cache + runs a 2-gen lineage per value.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=../../../.venv/bin/python
SD=out/sim_data_dnaa2/parca_state.pkl.gz
DILL=out/steady_state_inputs/succinate_default_gen3_start.dill
RESULTS=out/kr_sweep_results.tsv
echo -e "k_r\tgen2_mean_%DnaA-ATP\tgen2_totDnaA\tgen2_in_band" > "$RESULTS"

for KR in 1e-7 1e-6 1e-5; do
  tag=$(echo "$KR" | tr '.-' 'pm')
  echo "================ k_r = $KR ================"
  CACHE="out/cache_kr_$tag"; RUN="out/kr_run_$tag"; rm -rf "$CACHE" "$RUN"
  # 1) hydrate ParCa state, override k_r, build cache
  $PY - "$SD" "$CACHE" "$KR" <<'PY'
import sys, os, shutil
sys.path.insert(0,".")
from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
sd_gz, cache, kr = sys.argv[1], sys.argv[2], float(sys.argv[3])
if os.path.exists(cache): shutil.rmtree(cache)
sim = hydrate_sim_data_from_state(load_parca_state(sd_gz))
eq = sim.process.equilibrium
idx = [i for i,r in enumerate(eq.rxn_ids) if "MONOMER0-4565" in str(r)][0]
eq.rates_rev[idx] = kr   # direct array set (set_rev_rate uses a different complex-idx scheme)
assert abs(eq.rates_rev[idx]-kr) < kr*1e-6, "override did not stick"
print(f"  overrode k_r[idx={idx}] -> {eq.rates_rev[idx]:.1e}")
sim.genetic_perturbations = {"TU00259[c]": 1e-3}
save_sim_input(sim, cache, condition="succinate", fixed_media="minimal_succinate")
write_cache_version(cache, repo_root=".")
print("  cache built:", cache)
PY
  # 2) 2-gen run from the burned-in dill
  $PY scripts/run_condition_multigen_parquet.py --cache-dir "$CACHE" \
      --resume-dill "$DILL" --start-gen 1 --generations 2 --max-min 90 --seed 1 \
      --out-dir "$RUN" --experiment-id "kr_$tag" --dill-dir "$RUN/gen_dills"
  # 3) measure gen-2 mean DnaA-ATP fraction + total DnaA
  $PY - "$RUN" "$KR" "$RESULTS" <<'PY'
import sys, glob, polars as pl, numpy as np
run, kr, results = sys.argv[1], sys.argv[2], sys.argv[3]
fs=glob.glob(f"{run}/**/history/**/lineage_seed=1/**/*.pq", recursive=True)
ids=pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
def i(m): return ids.index(m)
bc=pl.col("bulk__count")
df=(pl.scan_parquet(fs, hive_partitioning=True).filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
    .select(["generation",bc.list.get(i("MONOMER0-160[c]")).alias("a"),bc.list.get(i("MONOMER0-4565[c]")).alias("d"),
             bc.list.get(i("PD03831[c]")).alias("p")]).collect())
g=df.filter(pl.col("generation")==2)
if len(g)==0: g=df.filter(pl.col("generation")==df["generation"].max())
tot=(g["a"]+g["d"]+g["p"]).to_numpy(); frac=(g["a"]/(g["a"]+g["d"]+g["p"])).to_numpy()
inb = "yes" if 0.2<=frac.mean()<=0.5 else ("LOW" if frac.mean()<0.2 else "HIGH")
open(results,"a").write(f"{kr}\t{frac.mean():.3f}\t{int(tot.mean())}\t{inb}\n")
print(f"  k_r={kr}: gen-2 mean %DnaA-ATP={frac.mean():.3f}  totDnaA={int(tot.mean())}  band={inb}")
PY
done
echo "================ SWEEP DONE ================"
column -t "$RESULTS"
