"""dnaa-3 — ALL readout plots ("what we will measure" section), Rashmi 2026-06-05.

One multi-panel figure over the succinate lineage covering every dnaa-3 readout:
  1. total DnaA concentration (count / cell MASS — Rashmi 2026-06-05)
  2. DnaA-ATP fraction (+ Boesen [0.2,0.5] band)
  3. oriC count (cell-cycle integrity, 1↔2)
  4. DnaA bound by pool: high-aff ATP / high-aff ADP / oriC-low ATP (free vs bound)
  5. total DnaA boxes (free vs bound), doubling across replication
  6. cell mass (periodic doubling)

Occupancy is the dnaa-3 fast-equilibrium binding model P=C/(C+K_d) evaluated on the
run's free DnaA-ATP/ADP concentration (the in-sim listener is pending its firing fix;
same model, same numbers). Pools: chromosomal_high 302 + oriC_high 3 + promoter_high
2 (K_d≈1 nM, ATP+ADP) ; oriC_low 8 (K_d≈100 nM, ATP only).
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AVOGADRO = 6.02214076e23
DENSITY_G_PER_L = 1100.0
N_HIGH = 302 + 3 + 2     # chromosomal + oriC-high + promoter (ATP+ADP, K_d 1 nM)
N_LOW = 8                # oriC-low (ATP only, K_d 100 nM)
KD_HIGH, KD_LOW = 1.0, 100.0


def load(run, gens):
    fs = []
    for g in gens:
        fs += sorted(glob.glob(f"{run}/**/history/**/generation={g}/**/*.pq", recursive=True))
    from collections import defaultdict
    by = defaultdict(list)
    for f in fs:
        by[f.split("generation=")[1].split("/")[0]].append(f)  # group nothing; keep all
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    def bi(m): return ids.index(m)
    bc = pl.col("bulk__count")
    df = (pl.scan_parquet(fs, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(["global_time", "generation",
                   bc.list.get(bi("PD03831[c]")).alias("apo"),
                   bc.list.get(bi("MONOMER0-160[c]")).alias("atp"),
                   bc.list.get(bi("MONOMER0-4565[c]")).alias("adp"),
                   pl.col("listeners__mass__cell_mass").alias("mass"),
                   pl.col("listeners__replication_data__number_of_oric").alias("oric")])
          .sort(["generation", "global_time"]).collect())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa2_seed1_8gen")
    ap.add_argument("--gens", default="2,3,4,5,6,7,8")
    ap.add_argument("--out", default="studies/dnaa-3-box-binding/charts/dnaa3_readouts")
    args = ap.parse_args()
    gens = [int(g) for g in args.gens.split(",")]
    df = load(args.run, gens)

    gt = df["global_time"].to_numpy(); gen = df["generation"].to_numpy()
    off = np.zeros(len(gt)); run = 0.0
    for g in sorted(set(gen.tolist())):
        m = gen == g; off[m] = run; run += gt[m].max() + 1.0   # cumulative across gens (+1s gap)
    t = (gt + off) / 60
    apo = df["apo"].to_numpy(); atp = df["atp"].to_numpy(); adp = df["adp"].to_numpy()
    mass = df["mass"].to_numpy(); oric = df["oric"].to_numpy()
    total = apo + atp + adp
    V_L = mass * 1e-15 / DENSITY_G_PER_L
    atp_nM = atp / (V_L * AVOGADRO) * 1e9
    adp_nM = adp / (V_L * AVOGADRO) * 1e9
    c_high = atp_nM + adp_nM
    p_high = c_high / (c_high + KD_HIGH)
    p_low = atp_nM / (atp_nM + KD_LOW)
    frac_atp_of_bound = np.divide(atp_nM, c_high, out=np.zeros_like(atp_nM), where=c_high > 0)
    # per-pool bound counts
    high_bound = N_HIGH * p_high
    high_bound_atp = high_bound * frac_atp_of_bound
    high_bound_adp = high_bound * (1 - frac_atp_of_bound)
    low_bound = N_LOW * p_low
    total_boxes = N_HIGH + N_LOW
    total_bound = high_bound + low_bound

    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("dnaa-3 — DnaA-box binding readouts on the succinate lineage "
                 "(fast-equilibrium model on the dnaa-2 baseline)", fontsize=13)

    a = ax[0, 0]; a.plot(t, total / mass, color="#0f172a"); a.set_title("Total DnaA concentration (count / cell mass)")
    a.set_ylabel("count / fg"); a.grid(alpha=0.25)

    a = ax[0, 1]; fr = atp / total
    a.plot(t, fr, color="#7c3aed"); a.axhspan(0.2, 0.5, color="#22c55e", alpha=0.12)
    a.axhline(0.2, color="#16a34a", lw=0.7, ls="--"); a.axhline(0.5, color="#16a34a", lw=0.7, ls="--")
    a.set_title("DnaA-ATP fraction (Boesen [0.2,0.5] band)"); a.set_ylabel("fraction"); a.grid(alpha=0.25)

    a = ax[1, 0]; a.step(t, oric, where="post", color="#16a34a")
    a.set_title("oriC count (1↔2, one initiation/gen)"); a.set_ylabel("oriC"); a.set_yticks([1, 2, 3, 4]); a.grid(alpha=0.25)

    a = ax[1, 1]
    a.plot(t, high_bound_atp, color="#2563eb", label="high-aff · bound-ATP")
    a.plot(t, high_bound_adp, color="#f59e0b", label="high-aff · bound-ADP")
    a.plot(t, low_bound, color="#dc2626", label="oriC-low · bound-ATP")
    a.set_title("DnaA bound, by pool & nucleotide"); a.set_ylabel("boxes bound"); a.legend(fontsize=7); a.grid(alpha=0.25)

    a = ax[2, 0]
    a.plot(t, total_bound, color="#475569", label="bound")
    a.plot(t, total_boxes - total_bound, color="#dc2626", label="FREE")
    a.axhline(total_boxes, color="#94a3b8", lw=0.7, ls=":")
    a.set_title(f"DnaA boxes: free vs bound (of {total_boxes})"); a.set_ylabel("boxes")
    a.set_xlabel("lineage time (min)"); a.legend(fontsize=8); a.grid(alpha=0.25)

    a = ax[2, 1]; a.plot(t, mass, color="#0f172a")
    a.set_title("cell mass (periodic doubling)"); a.set_ylabel("fg"); a.set_xlabel("lineage time (min)"); a.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}.png/.svg | free boxes range: {int((total_boxes-total_bound).min())}-{int((total_boxes-total_bound).max())}")


if __name__ == "__main__":
    main()
