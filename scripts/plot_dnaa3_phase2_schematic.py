"""dnaa-3 Phase 2 mechanism schematic.

Renders a single-page diagram of what Phase 2 wires into the simulation:

  Translation pool  →  apo DnaA  →  charging (bf8b82e)  →  bulk DnaA-ATP / DnaA-ADP
                                                              │
                                                              ├──→ 4 affinity pools
                                                              │      • chromosomal_high (302 sites, K_d=1 nM, ATP or ADP)
                                                              │      • oriC_high      (3 sites:  R1/R2/R4, K_d=1 nM, ATP or ADP)
                                                              │      • oriC_low       (8 sites:  R5M/τ2/I1/I2/C3/C2/I3/C1, K_d=100 nM, ATP only)
                                                              │      • promoter_high  (2 sites:  box1/box2, K_d=1 nM, ATP or ADP)
                                                              │
                                                              ├──→ bf8b82e hydrolysis  (free + bound pools, k=0.046/min, Pi+PROTON+WATER tracked via stoich)
                                                              │
                                                              └──→ fork-passage release (DnaA returns to bulk when replisome crosses a bound box)

Box-pool occupancy listener emits 11 per-pool / per-form counts.

Initiation gate is unchanged from Phase 1 — still mass-only.

Usage:
    python scripts/plot_dnaa3_phase2_schematic.py \\
        --out out/figures/dnaa3_phase2_schematic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _box(ax, xy, w, h, label, fc="#f1f5f9", ec="#334155", lw=1.4,
         fontsize=9, color="#0f172a"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=color, zorder=3)


def _arrow(ax, xy_from, xy_to, color="#0f172a", lw=1.3, ls="-",
           connectionstyle="arc3,rad=0.0"):
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(arrow)


def _label(ax, xy, text, fontsize=8, color="#475569",
           ha="left", va="center", bbox=None):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color,
            ha=ha, va=va, bbox=bbox, zorder=4)


def draw(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "dnaa-3 Phase 2 — DnaA-box binding mechanism (what we wired into v2ecoli)",
        fontsize=12, y=0.96,
    )

    # Translation source -------------------------------------------------
    _box(ax, (0.3, 7.5), 1.8, 0.7,
         "Translation\n(dnaA mRNA → apo DnaA)",
         fc="#fef3c7", ec="#b45309", fontsize=8)

    # Bulk apo + charging ------------------------------------------------
    _box(ax, (2.8, 7.5), 1.6, 0.7,
         "apo DnaA\n[PD03831]",
         fc="#fef9c3", ec="#a16207", fontsize=8)

    _box(ax, (4.9, 7.55), 0.9, 0.6,
         "DnaA\ncharging",
         fc="#dbeafe", ec="#1d4ed8", fontsize=8)

    _box(ax, (6.3, 7.5), 2.6, 0.7,
         "bulk DnaA-ATP    bulk DnaA-ADP\n[MONOMER0-160] [MONOMER0-4565]",
         fc="#dcfce7", ec="#166534", fontsize=8)

    _arrow(ax, (2.1, 7.85), (2.8, 7.85))
    _arrow(ax, (4.4, 7.85), (4.9, 7.85))
    _arrow(ax, (5.8, 7.85), (6.3, 7.85))
    _label(ax, (5.35, 8.30), "apo + ATP ⇌ DnaA-ATP", fontsize=7,
           color="#1e3a8a", ha="center")
    _label(ax, (5.35, 8.55), "apo + ADP ⇌ DnaA-ADP", fontsize=7,
           color="#1e3a8a", ha="center")

    # Intrinsic hydrolysis (with the Phase 2 extension to bound pool) -----
    _box(ax, (9.3, 7.4), 4.4, 0.95,
         "DnaA-ATP intrinsic hydrolysis (Sekimizu 1987)\n"
         "k = 0.046/min on (free + bound) DnaA-ATP\n"
         "→ DnaA-ADP + Pi + PROTON − WATER (stoich-tracked)\n"
         "Bound-pool: in-place ATP→ADP form swap; molecule\n"
         "stays attached, then re-equilibrates next tick",
         fc="#fce7f3", ec="#9d174d", fontsize=7)
    _arrow(ax, (8.9, 7.85), (9.3, 7.85))

    # The 4 pools (binding-step targets) ---------------------------------
    pool_y = 4.3
    pool_h = 1.0
    pool_w = 3.0

    _box(ax, (0.4, pool_y), pool_w, pool_h,
         "chromosomal_high\n302 sites\nK_d = 1 nM\nbinds ATP or ADP",
         fc="#e0e7ff", ec="#3730a3", fontsize=8)

    _box(ax, (3.7, pool_y), pool_w, pool_h,
         "oriC_high\n3 sites (R1, R2, R4)\nK_d = 1 nM\nbinds ATP or ADP",
         fc="#fae8ff", ec="#6b21a8", fontsize=8)

    _box(ax, (7.0, pool_y), pool_w, pool_h,
         "oriC_low\n8 sites (R5M, τ2, I1, I2,\nC3, C2, I3, C1)\n"
         "K_d = 100 nM   ATP only",
         fc="#cffafe", ec="#0e7490", fontsize=8)

    _box(ax, (10.3, pool_y), pool_w, pool_h,
         "promoter_high\n2 sites (box1, box2)\nK_d = 1 nM\nbinds ATP or ADP",
         fc="#fff7ed", ec="#9a3412", fontsize=8)

    # Binding step label between bulk and pools --------------------------
    _box(ax, (5.4, 5.85), 4.4, 0.85,
         "dnaa_box_binding step — fast-equilibrium competitive Langmuir\n"
         "Mass-balance system in (A_free, D_free) solved by scipy.optimize.root\n"
         "(MINPACK hybr, Newton-like; converges ~10 iter, mass-balance exact)",
         fc="#ecfeff", ec="#155e75", fontsize=7)

    # Arrows from bulk → binding step → pools ----------------------------
    _arrow(ax, (7.6, 7.5), (7.6, 6.6), color="#155e75")
    # binding step → each pool
    for px in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, (7.6, 6.05), (px, pool_y + pool_h),
               color="#155e75", lw=1.0,
               connectionstyle="arc3,rad=0.0")

    # Fork-passage release -----------------------------------------------
    # Place it directly under the bulk DnaA pool so the arrow can be a short
    # vertical line that doesn't cross any other box.
    _box(ax, (5.7, 1.8), 5.6, 1.4,
         "chromosome_structure.py — fork-passage DnaA release (Phase 2 fix)\n\n"
         "When a replication fork crosses a DnaA box:\n"
         " • parent box is deleted, 2 fresh child boxes added (bound_form=0)\n"
         " • bound DnaA-ATP / DnaA-ADP on the parent → released back to bulk\n"
         " • mass conserved; matches Katayama 2017 fork-passage dissociation",
         fc="#fee2e2", ec="#991b1b", fontsize=7)
    # Per-pool occupancy listener moved to make room (see below).

    # Per-pool occupancy listener (moved to left side) -------------------
    _box(ax, (0.3, 1.8), 5.2, 1.4,
         "replication_data listener — 11 per-pool occupancy counts\n\n"
         " chromosomal_high_{free,bound_atp,bound_adp}\n"
         " oriC_high_{free,bound_atp,bound_adp}\n"
         " oriC_low_{free,bound_atp}\n"
         " promoter_high_{free,bound_atp,bound_adp}",
         fc="#f3f4f6", ec="#374151", fontsize=7)
    for px in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, (px, pool_y), (2.9, 3.2),
               color="#6b7280", lw=0.8,
               connectionstyle="arc3,rad=-0.05")

    # Fork-release → bulk DnaA: straight vertical arrow that threads the
    # 0.3-unit gap between oriC_high (ends at x=6.7) and oriC_low (x=7.0).
    _arrow(ax, (6.85, 3.2), (6.85, 7.5),
           color="#991b1b", lw=1.4,
           connectionstyle="arc3,rad=0.0")
    _label(ax, (6.95, 5.5), "fork-released DnaA-ATP/ADP\n→ back to bulk",
           fontsize=7, color="#991b1b", ha="left")

    # Initiation gate note (out of scope) -------------------------------
    _label(
        ax, (7.0, 0.85),
        "Initiation gate — unchanged from Phase 1 (mass-only). "
        "Phase 2 is bookkeeping; Phase 3 will gate on oriC_low occupancy.",
        fontsize=8, color="#1f2937", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7ed",
                  edgecolor="#9a3412"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/figures/dnaa3_phase2_schematic.png")
    args = ap.parse_args()
    draw(Path(args.out))


if __name__ == "__main__":
    main()
