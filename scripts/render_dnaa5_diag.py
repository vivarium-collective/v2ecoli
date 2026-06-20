"""dnaa-5 diagnostic figure: the first-pass mechanistic initiation trigger collapses the
cell cycle (runaway growth, no division in gen 2) because oriC-high never reaches saturation."""
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "listeners__replication_data__"
def load(run_dir, eid):
    fs = sorted(glob.glob(f"{run_dir}/{eid}/history/**/*.pq", recursive=True))
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        ["generation", "global_time", pl.col(L+"number_of_oric"),
         pl.col(L+"oriC_high_bound_atp"), pl.col("listeners__mass__cell_mass")]
    ).collect().sort(["generation", "global_time"])
    return df

mech = load("out/dnaa5_mech_diag_seed1_8gen", "dnaa5_mech_diag_seed1_8gen")
mass = load("out/dnaa4_s06_F05_seed1_12gen", "dnaa4_s06_F05_seed1_12gen")
a = lambda d, c: np.asarray(d[c].to_list(), float)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
# Panel A: cell mass
tm = a(mass, "global_time")/60; cm_mass = a(mass, "listeners__mass__cell_mass")
te = a(mech, "global_time")/60; cm_mech = a(mech, "listeners__mass__cell_mass")
# clip mass control to the mech time window for visual comparison
ax1.plot(tm[tm <= te.max()+10], cm_mass[tm <= te.max()+10], color="#1f77b4", lw=1.3,
         label="MASS trigger (control) — clean sawtooth, divides ~735 fg")
ax1.plot(te, cm_mech, color="#d62728", lw=1.8,
         label="MECHANISTIC trigger — runs away in gen 2 (no division)")
ax1.axhline(735, color="#1f77b4", ls=":", lw=1, alpha=0.7)
ax1.annotate("gen 2: no initiation ->\nruns away to ~6159 fg (~8x)", xy=(te[-1], cm_mech[-1]),
             xytext=(te[-1]*0.45, cm_mech.max()*0.8), fontsize=9, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728"))
ax1.set_ylabel("cell mass (fg)")
ax1.set_title("dnaa-5 — first-pass mechanistic initiation trigger COLLAPSES the cell cycle\n"
              "(DnaA-ATP/oriC saturation trigger, threshold = 3/3 high-affinity; seed 1)", fontsize=11)
ax1.legend(fontsize=8.5, loc="upper left"); ax1.grid(alpha=0.2)

# Panel B: oriC_high_bound_atp (the trigger signal) with threshold
hi = a(mech, L+"oriC_high_bound_atp"); noric = a(mech, L+"number_of_oric")
ax2.plot(te, hi, color="#16a34a", lw=1.4, label="oriC high-aff DnaA-ATP bound (trigger signal)")
ax2.plot(te, 3*np.maximum(noric, 1), color="#9333ea", ls="--", lw=1.3, label="threshold = 3 x n_oriC (all high-aff bound)")
# shade gen 2
g2 = mech.filter(pl.col("generation") == 2)
if len(g2):
    t2 = a(g2, "global_time")/60
    ax2.axvspan(t2.min(), t2.max(), color="#fee2e2", zorder=0)
    ax2.annotate("gen 2: signal stuck at 2/3 —\nthreshold never reached, no initiation",
                 xy=(t2.mean(), 2), xytext=(t2.min(), 4.2), fontsize=9, color="#b91c1c")
ax2.set_xlabel("time (min)"); ax2.set_ylabel("oriC high-aff\nDnaA-ATP (count)")
ax2.set_ylim(0, 6.5); ax2.legend(fontsize=8.5, loc="upper right"); ax2.grid(alpha=0.2)
for ax in (ax1, ax2): ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "out/dnaa5_mechanistic_collapse.png"
fig.savefig(out, dpi=140); fig.savefig(out.replace(".png", ".svg"))
print("wrote", out, "and .svg")
