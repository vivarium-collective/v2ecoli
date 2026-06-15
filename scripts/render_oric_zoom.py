"""oriC-zoom figure: the origin of replication charging up with DnaA-ATP.

Left  — stacked occupancy of the oriC boxes (3 high-affinity + 8 low-affinity)
        by bound form over one cell cycle: the origin fills with DnaA-ATP toward
        initiation, then doubles (oriC x1 -> x2) as it fires.
Right — a snapshot of the origin-box array at the pre-initiation moment, each
        box colored by what it is bound to (free / DnaA-ATP / DnaA-ADP).

Usage: render_oric_zoom.py <run_dir> <generation> <out.png> [title]
"""
import sys
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "listeners__replication_data__"
P_ORIC_HI, P_ORIC_LO = 1, 2
F_FREE, F_ATP, F_ADP = 0, 1, 2
C_FREE, C_ATP, C_ADP = "#cbd5e1", "#16a34a", "#f59e0b"

run_dir = sys.argv[1]
gen = int(sys.argv[2])
out = sys.argv[3]
title = sys.argv[4] if len(sys.argv) > 4 else f"oriC charging with DnaA-ATP toward initiation (gen {gen})"

fs = sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))
df = pl.scan_parquet(fs, hive_partitioning=True).select(
    ["generation", "global_time", L + "number_of_oric",
     L + "dnaa_box_coordinates", L + "dnaa_box_pool_label", L + "dnaa_box_bound_form"]
).collect().sort(["generation", "global_time"]).filter(pl.col("generation") == gen)

t = df["global_time"].to_numpy() / 60.0
n_oric = df[L + "number_of_oric"].to_numpy()

# per-tick counts of oriC-region boxes (high+low) by bound form
free_c, atp_c, adp_c = [], [], []
for i in range(df.height):
    pl_ = np.asarray(df[L + "dnaa_box_pool_label"][i], int)
    bf = np.asarray(df[L + "dnaa_box_bound_form"][i], int)
    m = (pl_ == P_ORIC_HI) | (pl_ == P_ORIC_LO)
    free_c.append(int(np.sum(bf[m] == F_FREE)))
    atp_c.append(int(np.sum(bf[m] == F_ATP)))
    adp_c.append(int(np.sum(bf[m] == F_ADP)))
free_c, atp_c, adp_c = map(np.array, (free_c, atp_c, adp_c))

# initiation ticks: where number_of_oric increases
init_idx = np.where(np.diff(n_oric) > 0)[0] + 1
# pre-initiation snapshot: tick just before the first initiation
i_pre = int(init_idx[0] - 1) if len(init_idx) else int(np.argmax(atp_c))

fig = plt.figure(figsize=(13.5, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 0.05, 1.0], wspace=0.25)

# ---- left: stacked occupancy timeline ----
axL = fig.add_subplot(gs[0, 0])
axL.stackplot(t, free_c, atp_c, adp_c, colors=[C_FREE, C_ATP, C_ADP],
              labels=["free", "bound DnaA-ATP", "bound DnaA-ADP"], alpha=0.95)
for j, ii in enumerate(init_idx):
    axL.axvline(t[ii], color="#dc2626", lw=1.4, ls="--",
                label="initiation (oriC doubles)" if j == 0 else None)
axL.axvline(t[i_pre], color="#1e293b", lw=1.0, ls=":", label="pre-initiation snapshot")
axL.set_xlabel("time in generation (min)")
axL.set_ylabel("oriC boxes (3 high-aff + 8 low-aff)")
axL.set_title("oriC charging: origin boxes fill with DnaA-ATP, then fire")
axL.set_xlim(t.min(), t.max())
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2)
axL.grid(alpha=0.2)

# ---- right: origin-box array at pre-initiation ----
axR = fig.add_subplot(gs[0, 2])
pl_ = np.asarray(df[L + "dnaa_box_pool_label"][i_pre], int)
bf = np.asarray(df[L + "dnaa_box_bound_form"][i_pre], int)
co = np.asarray(df[L + "dnaa_box_coordinates"][i_pre], float)
cmap = {F_FREE: C_FREE, F_ATP: C_ATP, F_ADP: C_ADP}
hi = np.where(pl_ == P_ORIC_HI)[0]
lo = np.where(pl_ == P_ORIC_LO)[0]
hi = hi[np.argsort(co[hi])]
lo = lo[np.argsort(co[lo])]
for col, rows, y, lab in [("high", hi, 1.0, "high-affinity (R-boxes)"),
                          ("low", lo, 0.0, "low-affinity (I/τ-sites)")]:
    n = len(rows)
    xs = np.linspace(-0.45 * max(n, 1), 0.45 * max(n, 1), n) if n else []
    for x, r in zip(xs, rows):
        axR.scatter([x], [y], s=520 if col == "high" else 320,
                    c=cmap[int(bf[r])], edgecolors="#1e293b",
                    linewidths=1.2 if col == "high" else 0.7, zorder=3)
    axR.text(0, y + 0.36, lab, ha="center", fontsize=9, color="#334155")
n_atp = int(np.sum(bf[(pl_ == P_ORIC_HI) | (pl_ == P_ORIC_LO)] == F_ATP))
n_tot = int(np.sum((pl_ == P_ORIC_HI) | (pl_ == P_ORIC_LO)))
axR.set_title(f"oriC at t={t[i_pre]:.0f} min (pre-initiation)\n{n_atp}/{n_tot} boxes DnaA-ATP-loaded", fontsize=10)
axR.set_xlim(-3.2, 3.2)
axR.set_ylim(-0.7, 1.7)
axR.axis("off")

fig.suptitle(title, fontsize=13, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out, "| pre-init tick", i_pre, "| initiations:", list(init_idx))
