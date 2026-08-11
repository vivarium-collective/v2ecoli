"""Diagnostic plot: total bound / total empty / total DnaA boxes over time.

Haochen's diagnostic to see how saturated the K_d=1 nM high-affinity sites
actually are. If our model is over-binding, the total-bound curve will sit
essentially on top of the total-boxes curve (with the empty curve flat near
zero). Healthy behaviour: a meaningful gap between bound and total, with
empty rising and falling each cell cycle as bulk DnaA-ATP fluctuates.

Per-pool overlay (chromosomal_high, oriC_high, oriC_low, promoter_high) gives
the breakdown — the K_d=1 nM pools (chromosomal_high + oriC_high +
promoter_high) are the ones the high-affinity diagnosis is about.

Usage:
    python scripts/plot_dnaa3_box_occupancy.py \\
        --exp-root out/dnaa3_phase2_v1e3_steadystart_parquet/dnaa3_phase2_v1e3_steadystart \\
        --exp-id dnaa3_phase2_v1e3_steadystart \\
        --lineage-seed 7 --gens 8 \\
        --title-extra "V=1.0e-3 + steady-state start + seed=7" \\
        --out out/figures/dnaa3_box_occupancy.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

POOL_FIELDS = {
    "chromosomal_high (Kd=1 nM)": (
        "listeners__replication_data__chromosomal_high_bound_atp",
        "listeners__replication_data__chromosomal_high_bound_adp",
    ),
    "oriC_high (Kd=1 nM)": (
        "listeners__replication_data__oriC_high_bound_atp",
        "listeners__replication_data__oriC_high_bound_adp",
    ),
    "oriC_low (Kd=100 nM)": (
        "listeners__replication_data__oriC_low_bound_atp",
        None,  # ATP-only pool
    ),
    "promoter_high (Kd=1 nM)": (
        "listeners__replication_data__promoter_high_bound_atp",
        "listeners__replication_data__promoter_high_bound_adp",
    ),
}

POOL_COLORS = {
    "chromosomal_high (Kd=1 nM)": "#1d4ed8",
    "oriC_high (Kd=1 nM)": "#7c3aed",
    "oriC_low (Kd=100 nM)": "#0e7490",
    "promoter_high (Kd=1 nM)": "#9a3412",
}


def load(exp_root: str, exp_id: str, seed: int, n_gens: int):
    """Return dict with time series stitched across n_gens."""
    t = []
    total = []
    free = []
    boundaries = []
    per_pool = {p: [] for p in POOL_FIELDS}

    cols = [
        "global_time",
        "listeners__replication_data__total_DnaA_boxes",
        "listeners__replication_data__free_DnaA_boxes",
    ]
    for atp_col, adp_col in POOL_FIELDS.values():
        cols.append(atp_col)
        if adp_col is not None:
            cols.append(adp_col)

    for gen in range(1, n_gens + 1):
        agent = "0" * gen
        pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
               f"lineage_seed={seed}/generation={gen}/agent_id={agent}/*.pq")
        fs = sorted(glob.glob(pat),
                    key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
        if not fs:
            continue
        gen_t, gen_total, gen_free = [], [], []
        gen_pool = {p: [] for p in POOL_FIELDS}
        for f in fs:
            tbl = pq.read_table(f, columns=cols).to_pandas()
            gen_t.append(tbl["global_time"].to_numpy())
            gen_total.append(
                tbl["listeners__replication_data__total_DnaA_boxes"]
                .to_numpy())
            gen_free.append(
                tbl["listeners__replication_data__free_DnaA_boxes"]
                .to_numpy())
            for p, (atp_col, adp_col) in POOL_FIELDS.items():
                vals = tbl[atp_col].to_numpy()
                if adp_col is not None:
                    vals = vals + tbl[adp_col].to_numpy()
                gen_pool[p].append(vals)

        # Each generation's parquet resets global_time to 0 — offset so
        # the lineage timeline is monotonic.
        offset = (boundaries[-1] if boundaries
                  else 0)
        gen_t_arr = np.concatenate(gen_t) + offset
        t.append(gen_t_arr)
        total.append(np.concatenate(gen_total))
        free.append(np.concatenate(gen_free))
        for p in POOL_FIELDS:
            per_pool[p].append(np.concatenate(gen_pool[p]))
        boundaries.append(gen_t_arr[-1])

    return {
        "t": np.concatenate(t) if t else np.array([]),
        "total": np.concatenate(total) if total else np.array([]),
        "free": np.concatenate(free) if free else np.array([]),
        "per_pool": {p: np.concatenate(v) if v else np.array([])
                     for p, v in per_pool.items()},
        "gen_boundaries": boundaries[:-1] if boundaries else [],
    }


def plot(d: dict, out_path: Path, title_extra: str) -> None:
    t_min = d["t"] / 60
    boundaries_min = [b / 60 for b in d["gen_boundaries"]]
    total = d["total"]
    free = d["free"]
    bound = total - free

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"{title_extra}\n"
        "DnaA-box occupancy diagnostic\n"
        "(Healthy: total empty visibly nonzero, with cycle-to-cycle "
        "fluctuation)",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for b in boundaries_min:
            ax.axvline(b, color="#94a3b8", lw=1.0, ls="--",
                       alpha=0.6, zorder=0)

    # Panel 1 — total / bound (overlapping curves at large scale).
    ax = axes[0]
    ax.plot(t_min, total, color="#0f172a", lw=1.6,
            label=f"total DnaA boxes (max={int(total.max())})")
    ax.plot(t_min, bound, color="#dc2626", lw=1.4,
            label=f"total bound (mean frac={bound.mean()/total.mean():.2f})")
    vlines(ax)
    ax.set_ylabel("DnaA box count")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel 2 — total empty on its own axis (zoomed in).
    ax = axes[1]
    ax.plot(t_min, free, color="#16a34a", lw=1.4,
            label=f"total empty (mean={free.mean():.1f},  max={int(free.max())})")
    vlines(ax)
    ax.set_ylabel("Empty DnaA boxes")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel 3 — chromosomal_high bound (large scale, the 307-site pool).
    ax = axes[2]
    large_pool = "chromosomal_high (Kd=1 nM)"
    vals = d["per_pool"][large_pool]
    if vals.size:
        ax.plot(t_min, vals, color=POOL_COLORS[large_pool], lw=1.3,
                label=f"{large_pool} (mean bound={vals.mean():.1f})")
    vlines(ax)
    ax.set_ylabel("Chromosomal_high bound")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel 4 — small pools zoomed (oriC_high, oriC_low, promoter_high).
    ax = axes[3]
    for p, color in POOL_COLORS.items():
        if p == large_pool:
            continue
        vals = d["per_pool"][p]
        if vals.size == 0:
            continue
        ax.plot(t_min, vals, color=color, lw=1.3,
                label=f"{p} (mean bound={vals.mean():.1f})")
    vlines(ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bound count (small pools)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load(args.exp_root, args.exp_id, args.lineage_seed, args.gens)
    if d["t"].size == 0:
        raise SystemExit(f"no parquet found for {args.exp_id}")
    plot(d, Path(args.out), args.title_extra)


if __name__ == "__main__":
    main()
