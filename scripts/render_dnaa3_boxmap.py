"""dnaa-3 — chromosome map of the 307 consensus DnaA boxes (positions only).

Marks oriC + the dnaA promoter ONLY (Rashmi 2026-06-05: datA / DARS1 / DARS2 are
out of scope at this stage and removed). Box coordinates are read from v2ecoli
sim_data exactly as Replication._build_motifs() computes them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

CHROM_LEN = 4_641_652

def main():
    from v2ecoli.processes.parca.data_loader import hydrate_sim_data_from_state, load_parca_state
    sim = hydrate_sim_data_from_state(load_parca_state("out/sim_data_dnaa2/parca_state.pkl.gz"))
    coords = np.asarray(sim.process.replication.motif_coordinates["DnaA_box"], dtype=float)
    ang = 2 * np.pi * ((coords % CHROM_LEN) / CHROM_LEN)  # 0 = oriC = top

    fig, ax = plt.subplots(figsize=(8, 8.6))
    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.sin(th), np.cos(th), color="#cbd5e1", lw=1.6, zorder=1)
    # each box = a short radial tick
    for a in ang:
        x, y = np.sin(a), np.cos(a)
        ax.plot([0.97 * x, 1.03 * x], [0.97 * y, 1.03 * y], color="#475569", lw=0.6, zorder=2)
    # region markers — oriC + dnaA promoter only. oriC (0) and the dnaA promoter
    # (-44 kb) sit ~adjacent on the circle, so place the labels apart with leader
    # lines (Rashmi 2026-06-05: the two labels were overlapping).
    def mark(signed, label, color, label_xy):
        a = 2 * np.pi * ((signed % CHROM_LEN) / CHROM_LEN)
        x, y = np.sin(a), np.cos(a)
        ax.plot([1.04 * x, 1.16 * x], [1.04 * y, 1.16 * y], color=color, lw=4, zorder=4)
        lx, ly = label_xy
        ax.plot([1.16 * x, lx], [1.16 * y, ly], color=color, lw=0.8, ls=":", zorder=3)
        ax.text(lx, ly, label, ha="center", va="center", color=color,
                fontsize=13, fontweight="bold")
    mark(0, "oriC", "#16a34a", (0.42, 1.46))            # up-right
    mark(-44_000, "dnaA promoter", "#0891b2", (-0.92, 1.34))  # up-left
    # ter
    ax.plot(0, -1.0, "s", ms=12, mfc="#dc2626", mec="#dc2626", zorder=5)
    ax.text(0, -1.16, "ter", ha="center", color="#dc2626", fontsize=12, fontweight="bold")

    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.65); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("E. coli K-12 chromosome — 307 consensus DnaA boxes (TTWTNCACA)\n"
                 "from the v2ecoli whole-cell-model genome; oriC + dnaA promoter marked",
                 fontsize=13, y=0.97)
    fig.text(0.5, 0.04,
             "Each tick = one consensus DnaA box (Schaper-Messer 1995, exactly as "
             "Replication._build_motifs() computes).\ndnaa-3 partitions these into "
             "chromosomal-high (~302), oriC-high (R1/R2/R4), + adds 8 oriC-low & 2 "
             "promoter-high sites.\ndatA / DARS1 / DARS2 are out of scope at this stage "
             "(Rashmi 2026-06-05).", ha="center", fontsize=9, color="#475569")

    out = Path("studies/dnaa-3-box-binding/charts/chromosome_dnaa_boxes")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote {out}.png/.svg — {len(coords)} boxes, oriC + dnaA promoter only")

if __name__ == "__main__":
    main()
