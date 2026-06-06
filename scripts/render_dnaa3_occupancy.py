"""dnaa-3 — DnaA-box OCCUPANCY chromosome view (bound / unbound per box).

The chromosome view Rashmi asked for (2026-06-05): DnaA-box occupancy around the
circular chromosome at several timepoints across a succinate cell cycle, grouped
by regulatory region. Each box is a marker — FILLED = DnaA-bound, OPEN = free —
with a per-region bound/total fraction. Red ■ = ter. A bottom panel plots TOTAL
DnaA CONCENTRATION (all forms / cell mass) over the cycle (Rashmi 2026-06-05).

Occupancy is the dnaa-3 FAST-EQUILIBRIUM binding model evaluated on the validated
dnaa-2 baseline trajectory: per timestep, the free DnaA-ATP/ADP CONCENTRATION
(bulk count / cell volume) drives per-pool occupancy P = C/(C+K_d). This is the
dnaa-3 mechanism (the in-sim version is being wired); the current WCM leaves the
DnaA_box.DnaA_bound flag unset, so we compute the design's predicted occupancy
directly from concentration + the per-pool K_d.

Regions: oriC (3 high-aff R1/R2/R4 + 8 low-aff) + dnaA-promoter (2) + chromosomal
(302). datA / DARS1 / DARS2 are out of scope at this stage (Rashmi 2026-06-05).

Usage:
  .venv/bin/python scripts/render_dnaa3_occupancy.py \\
      --run out/dnaa2_seed1_8gen --generation 2 \\
      --out studies/dnaa-3-box-binding/charts/dnaa3_box_occupancy
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AVOGADRO = 6.02214076e23
DENSITY_G_PER_L = 1100.0  # E. coli cell density ≈ 1.1 g/mL

# dnaa-3 box catalog (4 affinity pools, 315 sites). K_d in nM; which nucleotide
# forms bind. oriC has 3 high + 8 low = 11 boxes (the "oriC k/11" in the ref).
POOLS = {
    "oriC_high":       dict(n=3,   kd=1.0,   forms="ATP+ADP", region="oriC"),
    "oriC_low":        dict(n=8,   kd=100.0, forms="ATP",     region="oriC"),
    "promoter_high":   dict(n=2,   kd=1.0,   forms="ATP+ADP", region="dnaA_promoter"),
    "chromosomal_high":dict(n=302, kd=1.0,   forms="ATP+ADP", region="chromosomal"),
}
REGION_COLOR = {"oriC": "#16a34a", "dnaA_promoter": "#0891b2", "chromosomal": "#94a3b8"}


def load(run: str, generation: int):
    fs = sorted(glob.glob(f"{run}/**/history/**/generation={generation}/**/*.pq", recursive=True))
    if not fs:
        raise SystemExit(f"no parquet for generation={generation} under {run}")
    # pick the single lineage (agent_id) with the most timesteps for a clean cycle
    from collections import defaultdict
    by_agent = defaultdict(list)
    for f in fs:
        a = f.split("agent_id=")[1].split("/")[0]
        by_agent[a].append(f)
    agent = max(by_agent, key=lambda a: len(by_agent[a]))
    fs = sorted(by_agent[agent])
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    def bi(m): return ids.index(m)
    bc = pl.col("bulk__count")
    df = (pl.scan_parquet(fs, hive_partitioning=True)
          .select(["global_time",
                   bc.list.get(bi("PD03831[c]")).alias("apo"),
                   bc.list.get(bi("MONOMER0-160[c]")).alias("atp"),
                   bc.list.get(bi("MONOMER0-4565[c]")).alias("adp"),
                   pl.col("listeners__mass__cell_mass").alias("mass"),
                   pl.col("listeners__mass__volume").alias("vol_fL"),
                   pl.col("listeners__replication_data__number_of_oric").alias("oric")])
          .sort("global_time").collect())
    return df


def conc_nM(count, mass_fg):
    """bulk count + cell mass(fg) → concentration in nM."""
    volume_L = mass_fg * 1e-15 / DENSITY_G_PER_L
    return count / (volume_L * AVOGADRO) * 1e9


from scipy.optimize import brentq
def pool_occupancy(atp_ct, adp_ct, vol_L, oric=1):
    """SELF-CONSISTENT occupancy: the high-affinity boxes sequester DnaA, so free
    DnaA (and thus oriC-low occupancy) is much lower than the naive bulk value
    (Haochen 2026-06-06). oriC boxes double when oriC has doubled (replication)."""
    D = float(atp_ct + adp_ct)  # binding-competent DnaA (apo≈0)
    n = {name: p["n"] * (oric if p["region"] == "oriC" else 1) for name, p in POOLS.items()}
    Kct = {name: p["kd"] * 1e-9 * vol_L * AVOGADRO for name, p in POOLS.items()}
    if D <= 0:
        return {k: 0.0 for k in POOLS}
    f = lambda F: F + sum(n[k] * F / (F + Kct[k]) for k in POOLS) - D
    F = brentq(f, 1e-9, D)
    return {k: F / (F + Kct[k]) for k in POOLS}


def region_counts(P, oric=1):
    """Expected (bound, total) per drawn region. oriC boxes DOUBLE when oriC has
    doubled at replication (Haochen 2026-06-06)."""
    reg = {"oriC": [0.0, 0], "dnaA_promoter": [0.0, 0], "chromosomal": [0.0, 0]}
    for name, p in POOLS.items():
        mult = oric if p["region"] == "oriC" else 1
        reg[p["region"]][0] += P[name] * p["n"] * mult
        reg[p["region"]][1] += p["n"] * mult
    return {r: (int(round(v[0])), v[1]) for r, v in reg.items()}


def _cluster(ax, center_xy, n_total, n_bound, color, cols=3, dr=0.11):
    """Draw a stacked cluster of n_total boxes; first n_bound filled (bound),
    the rest FREE — drawn larger with a bold red ring so they stand out even when
    most of the pool is saturated (Rashmi 2026-06-05: couldn't see unbound boxes)."""
    cx, cy = center_xy
    for k in range(n_total):
        r, c = divmod(k, cols)
        x = cx + (c - (cols - 1) / 2) * dr
        y = cy + r * dr
        if k < n_bound:
            ax.plot(x, y, "o", ms=8, mfc=color, mec=color, zorder=4)            # BOUND: solid
        else:
            ax.plot(x, y, "o", ms=9, mfc="white", mec="#dc2626", mew=2.2, zorder=6)  # FREE: bold red ring


def _lbl(nb, nt):
    return f"{nb} bound\n{nt - nb} free"

def draw_circle(ax, P, t_min, oric=1):
    rc = region_counts(P, oric)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.sin(th), np.cos(th), color="#cbd5e1", lw=2, zorder=1)
    # oriC cluster (top) — the regulatory pool where the free boxes live; drawn big
    _cluster(ax, (0.0, 1.20), rc["oriC"][1], rc["oriC"][0], REGION_COLOR["oriC"], cols=4)
    repl = "  (×2 replicating)" if oric >= 2 else ""
    ax.text(0, 1.74, f"oriC{repl} ({rc['oriC'][0]}/{rc['oriC'][1]})\n{_lbl(*rc['oriC'])}", ha="center",
            va="center", fontsize=8.5, color=REGION_COLOR["oriC"], fontweight="bold")
    # dnaA_promoter cluster (upper-left)
    _cluster(ax, (-1.15, 0.70), rc["dnaA_promoter"][1], rc["dnaA_promoter"][0],
             REGION_COLOR["dnaA_promoter"], cols=2)
    ax.text(-1.4, 1.08, f"dnaA_promoter\n{_lbl(*rc['dnaA_promoter'])}", ha="center", va="center",
            fontsize=7.5, color=REGION_COLOR["dnaA_promoter"], fontweight="bold")
    # chromosomal: ring of dots; FREE ones drawn larger + bold red ring so they pop
    nb, nt = rc["chromosomal"]
    ang = np.linspace(0.20, 2 * np.pi - 0.20, nt)  # skip the oriC/ter poles
    sel = np.ones(nt, bool)
    if nb < nt:  # mark the (nt-nb) free positions, spread around the ring
        sel[np.linspace(0, nt - 1, nt - nb).round().astype(int)] = False
    for a, b in zip(ang, sel):
        x, y = np.sin(a), np.cos(a)
        if b:
            ax.plot(x, y, "o", ms=2.6, mfc=REGION_COLOR["chromosomal"], mec=REGION_COLOR["chromosomal"], zorder=2)
        else:
            ax.plot(x, y, "o", ms=6, mfc="white", mec="#dc2626", mew=1.8, zorder=6)  # free: bold red ring
    ax.text(1.32, -0.25, f"chromosomal\n{_lbl(nb, nt)}", ha="center", va="center",
            fontsize=7.5, color="#64748b", fontweight="bold")
    ax.plot(0, -1, "s", ms=11, mfc="#dc2626", mec="#dc2626", zorder=5)  # ter
    ax.set_title(f"t = {t_min:.1f} min", fontsize=10)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.45, 1.92); ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa2_seed1_8gen")
    ap.add_argument("--generation", type=int, default=5)   # later steady gen (Haochen pt 2)
    ap.add_argument("--out", default="studies/dnaa-3-box-binding/charts/dnaa3_box_occupancy")
    ap.add_argument("--n-frames", type=int, default=5)
    args = ap.parse_args()

    df = load(args.run, args.generation)
    t0 = df["global_time"][0]
    tmin = ((df["global_time"] - t0) / 60).to_numpy()
    apo = df["apo"].to_numpy(); atp = df["atp"].to_numpy(); adp = df["adp"].to_numpy()
    mass = df["mass"].to_numpy(); vol_L = df["vol_fL"].to_numpy() * 1e-15; oric = df["oric"].to_numpy()
    idx = np.linspace(0, len(df) - 1, args.n_frames).round().astype(int)

    fig = plt.figure(figsize=(3.2 * args.n_frames, 5.4))
    gs = fig.add_gridspec(2, args.n_frames, height_ratios=[3.0, 1.15], hspace=0.34, wspace=0.04)
    fig.suptitle("dnaa-3 — DnaA-box occupancy by region across the succinate cell cycle\n"
                 "SELF-CONSISTENT fast-equilibrium binding (boxes sequester free DnaA; later/steady gens) · "
                 "filled = DnaA-bound, open = free · red ■ = ter · oriC doubles at replication · datA/DARS out of scope",
                 fontsize=12, y=1.0)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, k])
        ii = int(i)
        draw_circle(ax, pool_occupancy(atp[ii], adp[ii], vol_L[ii], int(oric[ii])), tmin[ii], int(oric[ii]))

    axc = fig.add_subplot(gs[1, :])
    # Rashmi 2026-06-05: total DnaA CONCENTRATION = count / cell MASS (not volume).
    conc_total = (apo + atp + adp) / mass  # all DnaA forms (count) / cell mass (fg)
    axc.plot(tmin, conc_total, color="#0f172a", lw=1.6)
    axc.scatter(tmin[idx], conc_total[idx], color="#dc2626", zorder=5, s=28)
    axc.set_xlabel("cell-cycle time (min)"); axc.set_ylabel("total DnaA conc.\n(count / fg)")
    axc.set_title("Total DnaA concentration — all forms (apo + ATP + ADP + bound) / cell MASS (Rashmi 2026-06-05)", fontsize=9)
    axc.grid(alpha=0.25)

    handles = [
        Line2D([], [], marker="o", ls="", mfc="#475569", mec="#475569", ms=9, label="DnaA-bound box"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#dc2626", mew=2.2, ms=9, label="FREE box (red ring)"),
        Line2D([], [], marker="s", ls="", mfc="#dc2626", mec="#dc2626", ms=9, label="ter"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}.png/.svg")
    m = int(idx[len(idx)//2])
    P = pool_occupancy(atp[m], adp[m], vol_L[m], int(oric[m]))
    bulk_atp_nM = atp[m] / (vol_L[m] * AVOGADRO) * 1e9
    print(f"mid-cycle bulk [DnaA-ATP] {bulk_atp_nM:.0f} nM | oriC={int(oric[m])} | self-consistent occupancy:",
          {k: round(v, 2) for k, v in P.items()})


if __name__ == "__main__":
    main()
