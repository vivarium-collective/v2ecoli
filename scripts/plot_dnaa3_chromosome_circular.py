"""Circular-chromosome DnaA-box occupancy snapshot + oriC trajectory.

Top: circular chromosome with every active DnaA box plotted at its angular
position (= signed genome coordinate / replichore length). Each pool occupies
its own concentric ring; bound state (free / ATP / ADP) is colour-coded.
Drawn in Cartesian coords so layout is predictable.

Bottom: oriC count over the same generation (from parquet listener).

Usage:
    python scripts/plot_dnaa3_chromosome_circular.py \\
        --dill out/dnaa3_phase2_v1.2e3_forkrelease_seed1/gen_dills/gen4.dill \\
        --exp-root out/dnaa3_phase2_v1.2e3_forkrelease_seed1_parquet/dnaa3_phase2_v1.2e3_forkrelease_seed1 \\
        --exp-id dnaa3_phase2_v1.2e3_forkrelease_seed1 \\
        --lineage-seed 1 --gen 4 \\
        --out out/figures/dnaa3_chromosome_gen4.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import dill
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyarrow.parquet as pq


# (label, ring_radius, marker_size)
POOL_INFO = {
    0: ("chromosomal_high (302×)", 1.00, 14),
    1: ("oriC_high (3×)",          1.20, 90),
    2: ("oriC_low (8×)",           1.36, 90),
    3: ("promoter_high (2×)",      1.52, 90),
}
FORM_COLOR = {0: "#cbd5e1", 1: "#16a34a", 2: "#dc2626"}
FORM_LABEL = {0: "free", 1: "bound ATP", 2: "bound ADP"}


def load_boxes(dill_path: Path):
    state = dill.load(open(dill_path, "rb"))
    boxes = state["unique"]["DnaA_box"]
    return boxes[boxes["_entryState"] == 1]


def load_gen_oric(exp_root: str, exp_id: str, lineage_seed: int,
                  gen: int):
    agent = "0" * gen
    pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
           f"lineage_seed={lineage_seed}/generation={gen}/agent_id={agent}/*.pq")
    files = sorted(glob.glob(pat),
                   key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
    import pandas as pd
    cols = ["global_time", "listeners__replication_data__number_of_oric"]
    rows = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    return pd.concat(rows).sort_values("global_time").reset_index(drop=True)


def _coord_to_theta(coords: np.ndarray, L: int) -> np.ndarray:
    """Map signed coordinate ∈ [-L, +L] → angle in radians, oriC at top
    (π/2), terC at bottom (-π/2). Right replichore (coord > 0) sweeps clockwise
    from oriC through east (3 o'clock) down to terC; left replichore (coord < 0)
    sweeps counter-clockwise from oriC through west (9 o'clock) down to terC.
    """
    return np.pi / 2 - (coords / L) * np.pi


def plot(active, oric_df, gen: int, replichore_len: int,
         out_path: Path, title_extra: str) -> None:
    coords = active["coordinates"].astype(np.int64)
    pool_lbl = active["pool_label"].astype(np.int8)
    bound_form = active["DnaA_bound_form"].astype(np.int8)
    domain = active["domain_index"].astype(np.int64)

    L = replichore_len
    theta = _coord_to_theta(coords, L)

    fig = plt.figure(figsize=(11.5, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.18)

    # ----- Top: Cartesian "polar-style" chromosome -----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal")
    ax.axis("off")
    R_OUT = 1.75
    ax.set_xlim(-R_OUT, R_OUT)
    ax.set_ylim(-R_OUT, R_OUT)

    # Faint backbone rings (one per pool radius). Pool labels go in a small
    # legend block to the upper-left of the disc so they don't overlap markers.
    th = np.linspace(0, 2 * np.pi, 400)
    for pool, (label, r, _) in POOL_INFO.items():
        ax.plot(r * np.cos(th), r * np.sin(th),
                color="#e2e8f0", lw=0.8, zorder=1)
    label_lines = []
    for pool, (label, r, _) in POOL_INFO.items():
        label_lines.append(f"r = {r:.2f}   →   {label}")
    ax.text(-R_OUT + 0.04, R_OUT - 0.04, "\n".join(label_lines),
            ha="left", va="top", fontsize=8, color="#475569",
            family="monospace")

    # Cardinal markers.
    ax.scatter([0], [R_OUT - 0.08], s=160, marker="*",
               color="#7c3aed", zorder=5)
    ax.text(0, R_OUT - 0.22, "oriC", ha="center", va="top",
            fontsize=10, color="#7c3aed", weight="bold")
    ax.plot([0, 0], [-R_OUT + 0.20, -R_OUT + 0.05],
            color="#475569", lw=1.0)
    ax.text(0, -R_OUT + 0.02, "terC", ha="center", va="top",
            fontsize=10, color="#475569")
    # Right / left mid-chromosome ticks (±L/2 → 3 o'clock / 9 o'clock).
    for sign, txt in ((1, f"+{L / 2 / 1e6:.2f} Mb"),
                      (-1, f"−{L / 2 / 1e6:.2f} Mb")):
        ax.plot([sign * (R_OUT - 0.05), sign * (R_OUT - 0.18)], [0, 0],
                color="#94a3b8", lw=0.8)
        ax.text(sign * (R_OUT + 0.02), 0,
                txt, ha="left" if sign > 0 else "right", va="center",
                fontsize=8, color="#475569")

    # Plot boxes, ring by ring. Free first (background), then ATP/ADP on top.
    for pool, (label, r, msize) in POOL_INFO.items():
        msk = pool_lbl == pool
        if not msk.any():
            continue
        dom = domain[msk]
        uniq_dom = np.unique(dom)
        # Small radial offset per chromosome copy (so duplicated boxes are
        # visible as two concentric arcs instead of stacking).
        dom_off = {d: (i - (len(uniq_dom) - 1) / 2) * 0.045
                   for i, d in enumerate(uniq_dom)}
        r_pt = np.array([r + dom_off[d] for d in dom])
        t_pt = theta[msk]
        bf_pt = bound_form[msk]
        x = r_pt * np.cos(t_pt)
        y = r_pt * np.sin(t_pt)
        for form in (0, 1, 2):
            sub = bf_pt == form
            if not sub.any():
                continue
            ax.scatter(x[sub], y[sub], s=msize,
                       facecolor=FORM_COLOR[form],
                       edgecolor="#0f172a",
                       linewidth=0.25 if pool == 0 else 0.5,
                       alpha=0.55 if form == 0 else 0.95,
                       zorder=3 if form > 0 else 2)

    # Legend in the bottom-right of the plot area.
    handles = [
        mpatches.Patch(facecolor=FORM_COLOR[1], edgecolor="#0f172a",
                       label="bound DnaA-ATP"),
        mpatches.Patch(facecolor=FORM_COLOR[2], edgecolor="#0f172a",
                       label="bound DnaA-ADP"),
        mpatches.Patch(facecolor=FORM_COLOR[0], edgecolor="#0f172a",
                       label="free box"),
        mpatches.Patch(facecolor="#7c3aed", edgecolor="#0f172a",
                       label="oriC"),
    ]
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.02, -0.02),
              frameon=False, fontsize=9)

    # Per-pool occupancy summary in the centre of the disc.
    summary = []
    for pool, (label, _, _) in POOL_INFO.items():
        msk = pool_lbl == pool
        n = int(msk.sum())
        if n == 0:
            continue
        n_atp = int(((bound_form == 1) & msk).sum())
        n_adp = int(((bound_form == 2) & msk).sum())
        n_free = n - n_atp - n_adp
        short = label.split(" (")[0]
        summary.append(
            f"{short:<17s}  ATP {n_atp:3d}   ADP {n_adp:3d}   "
            f"free {n_free:3d}   (of {n})"
        )
    ax.text(0, 0, "\n".join(summary), ha="center", va="center",
            fontsize=8.5, family="monospace", color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                      edgecolor="#cbd5e1", linewidth=0.8))

    ax.set_title(
        f"{title_extra}\nDnaA-box occupancy on the chromosome — "
        f"gen {gen} end-of-cycle snapshot",
        fontsize=11.5, pad=8,
    )

    # ----- Bottom: oriC count over gen 4 -----
    ax2 = fig.add_subplot(gs[1, 0])
    t = oric_df["global_time"].to_numpy()
    t_min = (t - t.min()) / 60.0
    n_oric = oric_df["listeners__replication_data__number_of_oric"].to_numpy()
    ax2.step(t_min, n_oric, color="#7c3aed", lw=1.7, where="post")
    ax2.set_ylim(-0.3, max(int(n_oric.max()) + 1, 4))
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_ylabel("oriC count")
    ax2.set_xlabel(f"time within gen {gen} (min)")
    ax2.set_xlim(t_min.min(), t_min.max())
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"oriC trajectory over gen {gen}",
                  fontsize=10, loc="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dill", required=True)
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=1)
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--replichore-len", type=int, default=2_320_000)
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    active = load_boxes(Path(args.dill))
    oric_df = load_gen_oric(args.exp_root, args.exp_id, args.lineage_seed,
                            args.gen)
    plot(active, oric_df, args.gen, args.replichore_len, Path(args.out),
         args.title_extra)


if __name__ == "__main__":
    main()
