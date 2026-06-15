"""Chromosome-state figure: the E. coli chromosome drawn as a ring, with every
DnaA box at its genomic position colored by what it is bound to (free /
DnaA-ATP / DnaA-ADP), the replication forks, the replicated region, and oriC.

Renders a row of snapshots across one cell cycle so you can SEE the boxes fill
with DnaA-ATP, oriC fire, and the replication bubble grow.

Usage: render_chromosome_state.py <run_dir> <generation> <out.png> [title]
"""
import sys
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

L = "listeners__replication_data__"
M = "listeners__mass__"
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
# pool labels
P_CHROM, P_ORIC_HI, P_ORIC_LO, P_PROM = 0, 1, 2, 3
# bound form
F_FREE, F_ATP, F_ADP = 0, 1, 2
C_FREE, C_ATP, C_ADP = "#cbd5e1", "#16a34a", "#f59e0b"

run_dir = sys.argv[1]
gen = int(sys.argv[2])
out = sys.argv[3]
title = sys.argv[4] if len(sys.argv) > 4 else f"Chromosome state across the cell cycle (generation {gen})"

fs = sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))
ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
idx = {k: ids.index(v) for k, v in BULK.items()}
cols = ["generation", "global_time", L + "number_of_oric", M + "cell_mass",
        L + "dnaa_box_coordinates", L + "dnaa_box_pool_label", L + "dnaa_box_bound_form",
        L + "fork_coordinates"] + BOUND_ATP + BOUND_ADP
df = pl.scan_parquet(fs, hive_partitioning=True).select(
    [pl.col(c) for c in cols]
    + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
).collect().sort(["generation", "global_time"]).filter(pl.col("generation") == gen)

t = df["global_time"].to_numpy() / 60.0
n_oric = df[L + "number_of_oric"].to_numpy()
nforks = np.array([len(x) if x is not None else 0 for x in df[L + "fork_coordinates"].to_list()])
a = lambda c: np.asarray(df[c].to_list(), float)
bound_atp = sum(a(c) for c in BOUND_ATP)
bound_adp = sum(a(c) for c in BOUND_ADP)
total = a("apo") + a("atp") + a("adp") + bound_atp + bound_adp
atp_frac = (a("atp") + bound_atp) / np.maximum(total, 1.0)

# scale for coordinate -> angle (oriC at top); use max |coordinate| seen
all_coords = np.concatenate([np.asarray(x, float) for x in df[L + "dnaa_box_coordinates"].to_list() if x is not None and len(x)])
scale = np.abs(all_coords).max() * 1.04

# choose 4 snapshots in time order: birth, pre-initiation (just before forks
# appear), post-initiation, and max-replication (largest theta bubble).
fork_extent = np.array([np.abs(np.asarray(x, float)).max() if (x is not None and len(x)) else 0.0
                        for x in df[L + "fork_coordinates"].to_list()])
fork_on = np.where(nforks > 0)[0]
roles = [(0, "birth")]
if len(fork_on):
    first = int(fork_on[0])
    roles.append((max(1, first - 1), "pre-initiation"))
    roles.append((min(len(t) - 1, first + max(60, (len(t) - first) // 6)), "post-initiation"))
    roles.append((int(np.argmax(fork_extent)), "max replication"))
else:
    roles += [(len(t) // 4, "early"), (len(t) // 2, "mid"), (len(t) - 2, "late")]
# dedup by index, keep first label, present left->right in time order
seen = {}
for i, lab in roles:
    if i not in seen:
        seen[i] = lab
snaps = sorted(seen)
labels = seen


def angle(coord):
    return np.pi / 2 - (np.asarray(coord, float) / scale) * np.pi


def draw(ax, i):
    bc = np.asarray(df[L + "dnaa_box_coordinates"][i], float)
    pl_ = np.asarray(df[L + "dnaa_box_pool_label"][i], int)
    bf = np.asarray(df[L + "dnaa_box_bound_form"][i], int)
    fk = df[L + "fork_coordinates"][i]
    fk = np.asarray(fk, float) if fk is not None else np.array([])
    R = 1.0
    # base ring
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), color="#94a3b8", lw=1.2, zorder=1)
    # replicated region: from oriC out to the forks (theta bubble), inner band
    if len(fk):
        cmax = np.abs(fk).max()
        a_hi, a_lo = angle(cmax), angle(-cmax)  # right(+) above-> a_hi< pi/2 ; left(-)-> a_lo> pi/2
        deg = np.degrees([a_hi, a_lo])
        w = Wedge((0, 0), 0.97, deg.min(), deg.max(), width=0.16,
                  facecolor="#bfdbfe", edgecolor="#3b82f6", lw=0.8, alpha=0.8, zorder=0)
        ax.add_patch(w)
    # DnaA boxes by pool, colored by bound form
    cmap = {F_FREE: C_FREE, F_ATP: C_ATP, F_ADP: C_ADP}
    colors = np.array([cmap[f] for f in bf])
    ang = angle(bc)
    x, y = R * np.cos(ang), R * np.sin(ang)
    # chromosomal high (small dots)
    m = pl_ == P_CHROM
    ax.scatter(x[m], y[m], c=colors[m], s=14, zorder=3, edgecolors="none")
    # oriC high + low (larger, ringed)
    m = (pl_ == P_ORIC_HI) | (pl_ == P_ORIC_LO)
    ax.scatter(x[m] * 1.0, y[m] * 1.0, c=colors[m], s=70, zorder=5,
               edgecolors="#1e293b", linewidths=0.7, marker="o")
    # promoter (the autoregulation sensor) — purple-edged squares
    m = pl_ == P_PROM
    ax.scatter(x[m], y[m], c=colors[m], s=80, zorder=6, marker="s",
               edgecolors="#7c3aed", linewidths=1.4)
    # forks
    if len(fk):
        fa = angle(fk)
        ax.scatter(R * np.cos(fa), R * np.sin(fa), marker="D", s=80, c="#dc2626",
                   edgecolors="white", linewidths=0.8, zorder=7)
    # oriC star at top, terC tick at bottom
    ax.scatter([0], [R], marker="*", s=240, c="#f59e0b", edgecolors="#1e293b",
               linewidths=0.8, zorder=8)
    ax.text(0, R + 0.13, "oriC", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0, -R - 0.13, "terC", ha="center", va="top", fontsize=8, color="#64748b")
    # DnaA-cycle regulatory loci (oriC-relative coords). DARS1/DARS2 reactivate
    # DnaA-ADP -> DnaA-ATP; datA titrates/hydrolyses (DDAH). Markers now; their
    # effect on box color (ATP recovery) appears once dnaa-5 DARS/DDAH are wired.
    LOCI = {"DARS1": (1529044, "#0891b2", "v"), "DARS2": (-956504, "#0891b2", "v"),
            "datA": (467079, "#b45309", "P")}
    for nm, (c, col, mk) in LOCI.items():
        aa = angle(c)
        x0, y0 = R * np.cos(aa), R * np.sin(aa)
        x1, y1 = 1.17 * np.cos(aa), 1.17 * np.sin(aa)
        ax.plot([x0, x1], [y0, y1], color=col, lw=1.0, zorder=4)
        ax.scatter([x0], [y0], marker=mk, s=60, c=col, edgecolors="white",
                   linewidths=0.6, zorder=9)
        ax.text(x1 * 1.03, y1 * 1.03, nm, fontsize=7.5, color=col, fontweight="bold",
                ha="left" if x1 >= 0 else "right", va="center")
    # center stats
    ax.text(0, 0.12, f"t = {t[i]:.0f} min", ha="center", fontsize=10, fontweight="bold")
    ax.text(0, -0.02, f"DnaA {total[i]:.0f}", ha="center", fontsize=8.5)
    ax.text(0, -0.14, f"ATP-form {atp_frac[i]*100:.0f}%", ha="center", fontsize=8.5, color=C_ATP)
    ax.text(0, -0.27, f"oriC×{int(n_oric[i])}  forks×{len(fk)}", ha="center", fontsize=8.5, color="#dc2626")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(labels.get(i, ""), fontsize=11, pad=2)


fig, axes = plt.subplots(1, len(snaps), figsize=(4.3 * len(snaps), 4.9))
if len(snaps) == 1:
    axes = [axes]
for ax, i in zip(axes, snaps):
    draw(ax, i)

# legend
from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FREE, markersize=9, label="box: free"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ATP, markersize=9, label="box: bound DnaA-ATP"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ADP, markersize=9, label="box: bound DnaA-ADP"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_ATP, markeredgecolor="#7c3aed",
           markersize=10, label="dnaA-promoter box (sensor)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#dc2626", markersize=9, label="replication fork"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#f59e0b", markersize=15, label="oriC"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#bfdbfe", markeredgecolor="#3b82f6",
           markersize=11, label="replicated (2 copies)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#0891b2", markersize=9, label="DARS1/2 (reactivation)"),
    Line2D([0], [0], marker="P", color="w", markerfacecolor="#b45309", markersize=9, label="datA (DDAH/titration)"),
]
fig.legend(handles=leg, loc="lower center", ncol=7, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(title, fontsize=13.5, y=1.02)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
