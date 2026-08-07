import sys, glob, os, json, numpy as np, polars as pl
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("rda", "scripts/render_dnaa4_autoreg.py")
rda = importlib.util.module_from_spec(spec); spec.loader.exec_module(rda)
import dnaa_observables as dnaa
L = dnaa.L

RUNS = {
  "langmuir_n1":  "out/dnaa4_s06_F05_seed1_12gen",         # n=1 Langmuir baseline
  "coop_n4":      "out/dnaa5_coop_n4_k30_seed1_8gen",      # operating point
  "coop_n6":      "out/dnaa5_coop_n6_k30_seed1_8gen",      # too strong
  "coop_n4_s0":   "out/dnaa5_coop_n4_k30_seed0_8gen",
  "coop_n4_s2":   "out/dnaa5_coop_n4_k30_seed2_8gen",
}
def occ_frame(run):
    fs = sorted(glob.glob(f"{run}/**/history/**/*.pq", recursive=True))
    cols = ["generation","global_time", L+"number_of_oric",
            L+"oriC_low_bound_atp", L+"oriC_low_free", dnaa.MASS_COL]
    df = rda._frame(run)  # gives total_dnaa, atp_fraction, n_oric, abs_min
    # occupancy needs the raw oriC_low cols; pull them directly
    raw = pl.scan_parquet(fs, hive_partitioning=True).select(
        ["generation","global_time", pl.col(L+"oriC_low_bound_atp").alias("lb"),
         pl.col(L+"oriC_low_free").alias("lf")]).collect().sort(["generation","global_time"])
    lb = raw["lb"].to_numpy().astype(float); lf = raw["lf"].to_numpy().astype(float)
    occ = lb/np.maximum(lb+lf,1e-9)
    return df, occ

out = {}
for tag, run in RUNS.items():
    if not glob.glob(f"{run}/**/history/**/*.pq", recursive=True):
        print(f"skip {tag}: no data"); continue
    df, occ = occ_frame(run)
    n = df.height
    # subsample to ~1500 pts for a light HTML
    idx = np.linspace(0, n-1, min(n,1500)).astype(int)
    g = df["generation"].to_numpy()
    ss = g >= 3
    out[tag] = {
      "t_min":   [round(float(x),2) for x in df["abs_min"].to_numpy()[idx]],
      "occ":     [round(float(x),4) for x in occ[idx]],
      "n_oric":  [int(x) for x in df["n_oric"].to_numpy()[idx]],
      "pool":    [round(float(x),1) for x in df["total_dnaa"].to_numpy()[idx]],
      "atpfr":   [round(float(x),4) for x in df["atp_fraction"].to_numpy()[idx]],
      "occ_max": round(float(occ.max()),3),
      "occ_g3p_mean": round(float(occ[ss].mean()),3),
      "pool_g3p_mean": round(float(df["total_dnaa"].to_numpy()[ss].mean()),1),
      "atpfr_g3p_mean": round(float(df["atp_fraction"].to_numpy()[ss].mean()),3),
    }
    print(f"{tag}: occ_max={out[tag]['occ_max']} pool_g3p={out[tag]['pool_g3p_mean']} atpfr={out[tag]['atpfr_g3p_mean']}")
json.dump(out, open("/tmp/dnaa5_data.json","w"))
print("wrote /tmp/dnaa5_data.json")
