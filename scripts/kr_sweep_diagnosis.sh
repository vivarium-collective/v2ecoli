#!/usr/bin/env bash
# k_r sweep — confirm the dnaa-2 diagnosis lever.
# Hypothesis (from the seed-1 8-gen diagnosis): the DnaA-ATP fraction sits at the
# LOW edge of [0.2,0.5] at the proper DnaA abundance because DnaA-ADP dissociation
# is slow (k_r = 1e-7/s) — a faster k_r should re-charge more apo→DnaA-ATP and lift
# the fraction. This sweeps k_r (reverse rate of MONOMER0-4565_RXN), rebuilding
# ParCa+cache per value, and measures the steady-gen DnaA-ATP fraction.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=../../../.venv/bin/python
TSV=v2ecoli/processes/parca/reconstruction/ecoli/flat/equilibrium_reaction_rates.tsv
DILL=out/steady_state_inputs/succinate_default_gen3_start.dill
RESULTS=out/kr_sweep_results.tsv
cp "$TSV" "$TSV.orig"
trap 'cp "$TSV.orig" "$TSV"; rm -f "$TSV.orig"' EXIT
echo -e "k_r\tgen2_mean_%DnaA-ATP\tgen2_totDnaA" > "$RESULTS"

for KR in 1.0E-07 1.0E-06 1.0E-05; do
  tag=$(echo "$KR" | tr '.+' 'p_')
  echo "================ k_r = $KR ================"
  # 1) set both reverse-rate columns for MONOMER0-4565_RXN
  $PY - "$KR" <<'PY'
import sys
kr=sys.argv[1]
p="v2ecoli/processes/parca/reconstruction/ecoli/flat/equilibrium_reaction_rates.tsv"
lines=open(p).read().splitlines()
out=[]
for ln in lines:
    if ln.startswith('"MONOMER0-4565_RXN"'):
        c=ln.split('\t'); c[2]=kr; c[3]=kr; ln='\t'.join(c)
    out.append(ln)
open(p,'w').write('\n'.join(out)+'\n')
print("set MONOMER0-4565_RXN reverse_rate ->", kr)
PY
  # 2) ParCa (full)  3) gzip  4) cache  5) 2-gen run  (per k_r)
  SD="out/sim_data_kr_$tag"; CACHE="out/cache_kr_$tag"; RUN="out/kr_run_$tag"
  rm -rf "$SD" "$CACHE" "$RUN"; mkdir -p "$SD"
  $PY scripts/parca_run.py --mode full --cpus 8 -o "$SD"
  gzip -c "$SD/parca_state.pkl" > "$SD/parca_state.pkl.gz"
  $PY - "$SD" "$CACHE" <<'PY'
import sys, os, shutil
sys.path.insert(0,".")
from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
sd_dir, cache = sys.argv[1], sys.argv[2]
if os.path.exists(cache): shutil.rmtree(cache)
state=load_parca_state(f"{sd_dir}/parca_state.pkl.gz")
sim=hydrate_sim_data_from_state(state)
sim.genetic_perturbations={"TU00259[c]":1e-3}
save_sim_input(sim, cache, condition="succinate", fixed_media="minimal_succinate")
write_cache_version(cache, repo_root=".")
print("cache built:", cache)
PY
  $PY scripts/run_condition_multigen_parquet.py --cache-dir "$CACHE" \
      --resume-dill "$DILL" --start-gen 1 --generations 2 --max-min 90 --seed 1 \
      --out-dir "$RUN" --experiment-id "kr_$tag" --dill-dir "$RUN/gen_dills"
  # 6) measure gen-2 mean DnaA-ATP fraction + total DnaA
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
g2=df.filter(pl.col("generation")==2)
tot=(g2["a"]+g2["d"]+g2["p"]).to_numpy(); frac=(g2["a"]/(g2["a"]+g2["d"]+g2["p"])).to_numpy()
with open(results,"a") as f: f.write(f"{kr}\t{frac.mean():.3f}\t{int(tot.mean())}\n")
print(f"k_r={kr}: gen-2 mean %DnaA-ATP = {frac.mean():.3f}, totDnaA = {int(tot.mean())}")
PY
done
echo "================ SWEEP DONE ================"
cat "$RESULTS"
