"""dnaa-3 — sequestration-capacity VARIANTS (for Haochen 2026-06-06).

Haochen flagged that the model over-binds the oriC low-affinity boxes: free DnaA
should be <100 nM but the self-consistent model at the strict-consensus box count
(307) sits at ~186 nM. The DEFAULT model is UNCHANGED (307 consensus boxes); this
figure is an exploratory sensitivity sweep, NOT a recalibration — it shows how the
high-affinity sequestration capacity (the number of genomic DnaA boxes, uncertain:
~300 strict consensus → ~450 incl. weak/secondary sites) moves free DnaA and the
oriC-low occupancy into the regime Haochen expects.

Each variant changes ONLY the high-affinity box count; K_d (1 nM high / 100 nM
oriC-low), DnaA abundance, and the trajectory are the validated dnaa-2 lineage.
Concentrations are nM = count / cell VOLUME (Haochen pt 5). Steady gens only.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, polars as pl
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NA = 6.02214076e23
KD_HIGH_nM, KD_LOW_nM = 1.0, 100.0
N_LOW = 8
DEFAULT = 307                       # strict TTWTNCACA consensus — the unchanged default
VARIANTS = [307, 360, 400, 450]     # default + weak/secondary-site sensitivity
COLORS = {307: "#0f172a", 360: "#2563eb", 400: "#16a34a", 450: "#9333ea"}


def load(run, gens):
    fs = []
    for g in gens:
        fs += sorted(glob.glob(f"{run}/**/history/**/generation={g}/**/*.pq", recursive=True))
    fs = [f for f in fs if "/agent_id=0" in f]
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    bi = lambda m: ids.index(m); bc = pl.col("bulk__count")
    return (pl.scan_parquet(fs, hive_partitioning=True)
            .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
            .select(["generation", "global_time",
                     bc.list.get(bi("MONOMER0-160[c]")).alias("atp"),
                     bc.list.get(bi("MONOMER0-4565[c]")).alias("adp"),
                     pl.col("listeners__mass__volume").alias("vol_fL"),
                     pl.col("listeners__replication_data__number_of_oric").alias("oric")])
            .sort(["generation", "global_time"]).collect())


def solve(D, V_L, oric, n_high_base):
    """Self-consistent free DnaA (count) for a given high-aff box count."""
    F = np.empty(len(D))
    for i in range(len(D)):
        nh = n_high_base * (1.6 if oric[i] >= 2 else 1.0)   # chromosomal partly replicated
        nl = N_LOW * oric[i]
        Kh = KD_HIGH_nM * 1e-9 * V_L[i] * NA
        Kl = KD_LOW_nM * 1e-9 * V_L[i] * NA
        F[i] = brentq(lambda f: f + nh * f / (f + Kh) + nl * f / (f + Kl) - D[i], 1e-9, D[i])
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa2_seed1_8gen")
    ap.add_argument("--gens", default="4,5,6")
    ap.add_argument("--out", default="studies/dnaa-3-box-binding/charts/dnaa3_box_variants")
    args = ap.parse_args()
    gens = [int(g) for g in args.gens.split(",")]
    df = load(args.run, gens)

    gt = df["global_time"].to_numpy(); gen = df["generation"].to_numpy()
    off = np.zeros(len(gt)); run = 0.0
    for g in sorted(set(gen.tolist())):
        m = gen == g; off[m] = run; run += gt[m].max() + 1.0
    t = (gt + off) / 60
    D = (df["atp"].to_numpy() + df["adp"].to_numpy()).astype(float)
    V_L = df["vol_fL"].to_numpy() * 1e-15
    oric = df["oric"].to_numpy()

    res = {}
    for nh in VARIANTS:
        F = solve(D, V_L, oric, nh)
        F_nM = F / (V_L * NA) * 1e9
        Kl = KD_LOW_nM * 1e-9 * V_L * NA
        res[nh] = (F_nM, F / (F + Kl))

    # fine sweep for the summary curve
    sweep_n = np.arange(307, 471, 12)
    sweep_free, sweep_occ = [], []
    for nh in sweep_n:
        F = solve(D, V_L, oric, int(nh))
        sweep_free.append((F / (V_L * NA) * 1e9).mean())
        Kl = KD_LOW_nM * 1e-9 * V_L * NA
        sweep_occ.append((F / (F + Kl)).mean())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("dnaa-3 — sequestration-capacity VARIANTS (for Haochen) · DEFAULT stays 307 consensus boxes · "
                 "free DnaA = count / cell VOLUME (nM), steady gens", fontsize=12)

    a = ax[0]
    for nh in VARIANTS:
        lbl = f"{nh} boxes" + (" (default)" if nh == DEFAULT else "")
        a.plot(t, res[nh][0], color=COLORS[nh], lw=1.4, label=lbl)
    a.axhline(KD_LOW_nM, color="#dc2626", ls="--", lw=1, label="100 nM (oriC-low K_d / Haochen target)")
    a.set_title("Free DnaA (nM) over time, per box count"); a.set_ylabel("free DnaA (nM)")
    a.set_xlabel("steady-gen time (min)"); a.legend(fontsize=7); a.grid(alpha=0.25)

    a = ax[1]
    for nh in VARIANTS:
        a.plot(t, res[nh][1], color=COLORS[nh], lw=1.4, label=f"{nh}")
    a.axhline(0.5, color="#94a3b8", ls=":", lw=1)
    a.set_title("oriC low-affinity occupancy over time"); a.set_ylabel("fraction bound")
    a.set_ylim(0, 1); a.set_xlabel("steady-gen time (min)"); a.legend(fontsize=7, title="boxes"); a.grid(alpha=0.25)

    a = ax[2]
    a.plot(sweep_n, sweep_free, color="#0f172a", lw=1.6)
    a.axhspan(0, KD_LOW_nM, color="#22c55e", alpha=0.10)
    a.axhline(KD_LOW_nM, color="#dc2626", ls="--", lw=1)
    for nh in VARIANTS:
        a.scatter([nh], [res[nh][0].mean()], color=COLORS[nh], zorder=5, s=40)
        a.annotate(f"{nh}", (nh, res[nh][0].mean()), textcoords="offset points", xytext=(4, 4), fontsize=8)
    a.axvline(DEFAULT, color="#0f172a", ls=":", lw=0.8)
    a.set_title("Mean free DnaA vs box count (green = <100 nM window)")
    a.set_ylabel("mean free DnaA (nM)"); a.set_xlabel("high-affinity box count"); a.grid(alpha=0.25)

    note = ("Default (307 consensus) → free DnaA ~186 nM (over-binds oriC-low, as Haochen noted). "
            "A modest increase in sequestration capacity (weak/secondary genomic DnaA sites; literature ~300-450) "
            "moves free DnaA below 100 nM — ~360 reaches the edge, ~400 sits comfortably in-window. "
            "Shown as variants for discussion; the canonical model is unchanged.")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, color="#7c2d12", wrap=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=130, bbox_inches="tight")
    print("wrote", args.out, "| mean free DnaA nM:", {nh: round(float(res[nh][0].mean()), 0) for nh in VARIANTS})


if __name__ == "__main__":
    main()
