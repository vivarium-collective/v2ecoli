import glob
import numpy as np
import polars as pl

L = "listeners__replication_data__"
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
BAND = (300, 800)


def per_gen(run, gens=range(9, 15)):
    fs = sorted(glob.glob(f"{run}/**/history/**/*.pq", recursive=True))
    if not fs:
        return None
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {k: ids.index(v) for k, v in BULK.items()}
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        ["generation"] + BOUND_ATP + BOUND_ADP
        + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
    ).collect()
    a = lambda d, c: np.asarray(d[c].to_list(), float)
    out = {}
    for g in gens:
        d = df.filter(pl.col("generation") == g)
        if d.height == 0:
            continue
        tot = a(d, "apo") + a(d, "atp") + a(d, "adp") + sum(a(d, c) for c in BOUND_ATP) + sum(a(d, c) for c in BOUND_ADP)
        out[g] = (tot.mean(), tot.min(), tot.max())
    return out


runs = {
    "s=0.70 (baseline, seed0)": "out/dnaa4_s07_seed0_16gen",
    "s=0.65 (sweep)": "out/dnaa4_band_s065",
    "s=0.60 (sweep)": "out/dnaa4_band_s060",
}
print(f"Band [{BAND[0]},{BAND[1]}] — per-gen DnaA mean (trough/peak), gens 9-14\n")
for label, run in runs.items():
    pg = per_gen(run)
    if not pg:
        print(f"{label}: no data"); continue
    means = [m for m, lo, hi in pg.values()]
    troughs = [lo for m, lo, hi in pg.values()]
    peaks = [hi for m, lo, hi in pg.values()]
    mean_inband = all(BAND[0] <= m <= BAND[1] for m in means)
    trough_ok = min(troughs) >= BAND[0]
    peak_ok = max(peaks) <= BAND[1]
    print(f"{label}:")
    for g, (m, lo, hi) in pg.items():
        flag = "" if BAND[0] <= m <= BAND[1] else "  <-- mean OUT"
        print(f"   gen{g}: mean={m:6.0f}  trough={lo:6.0f}  peak={hi:6.0f}{flag}")
    print(f"   => per-gen MEAN all in-band: {mean_inband} | min trough={min(troughs):.0f} (>=300: {trough_ok}) | max peak={max(peaks):.0f} (<=800: {peak_ok})")
    print(f"   => CLEAN PASS (mean in-band AND trough>=300 AND peak<=800): {mean_inband and trough_ok and peak_ok}\n")
