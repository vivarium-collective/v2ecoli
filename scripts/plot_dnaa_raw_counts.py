"""DnaA raw counts across a multigen lineage.

Same panel layout as plot_dnaa_concentrations.py but in raw molecule counts
(no normalization). Useful when you want to read off molecules-per-cell
directly without thinking about cell volume.

Usage:
    python scripts/plot_dnaa_raw_counts.py \\
        --exp-root out/<exp>_parquet/<exp> \\
        --exp-id <exp> \\
        --lineage-seed 1 --gens 8 \\
        --out out/figures/<exp>_8gen_raw_counts.png
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

DNAA_ATP = "MONOMER0-160[c]"
DNAA_ADP = "MONOMER0-4565[c]"
DNAA_APO = "PD03831[c]"


def _agent(gen: int) -> str:
    return "0" * gen


def load_lineage(exp_root: str, exp_id: str, seed: int, n_gens: int) -> dict:
    files0 = sorted(glob.glob(
        f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
        f"lineage_seed={seed}/generation=1/agent_id=0/*.pq"),
        key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
    bulk_ids = pq.read_table(files0[0], columns=["bulk__id"]).column(
        "bulk__id")[0].as_py()
    atp_idx = bulk_ids.index(DNAA_ATP)
    adp_idx = bulk_ids.index(DNAA_ADP)
    apo_idx = bulk_ids.index(DNAA_APO) if DNAA_APO in bulk_ids else None

    cols = ["global_time",
            "listeners__mass__cell_mass",
            "bulk__count",
            "listeners__replication_data__number_of_oric",
            "listeners__replication_data__chromosomal_high_bound_atp",
            "listeners__replication_data__chromosomal_high_bound_adp",
            "listeners__replication_data__oriC_high_bound_atp",
            "listeners__replication_data__oriC_high_bound_adp",
            "listeners__replication_data__oriC_low_bound_atp",
            "listeners__replication_data__promoter_high_bound_atp",
            "listeners__replication_data__promoter_high_bound_adp"]
    import pandas as pd
    rows_all = []
    gen_starts = []
    cum_t = 0.0
    for gen in range(1, n_gens + 1):
        agent = _agent(gen)
        pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
               f"lineage_seed={seed}/generation={gen}/agent_id={agent}/*.pq")
        files = sorted(glob.glob(pat),
                       key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
        if not files:
            continue
        rs = [pq.read_table(f, columns=cols).to_pandas() for f in files]
        df = pd.concat(rs).sort_values("global_time").reset_index(drop=True)
        gen_starts.append(cum_t)
        df["t_rel"] = (df["global_time"] - df["global_time"].iloc[0]) / 60.0
        df["t_cum_min"] = cum_t + df["t_rel"]
        cum_t = float(df["t_cum_min"].iloc[-1])
        rows_all.append(df)
    df_all = pd.concat(rows_all).reset_index(drop=True)
    bulk = np.stack(df_all["bulk__count"].to_numpy())
    return {
        "t_min": df_all["t_cum_min"].to_numpy(),
        "cell_mass_fg": df_all["listeners__mass__cell_mass"].to_numpy(),
        "atp_bulk": bulk[:, atp_idx],
        "adp_bulk": bulk[:, adp_idx],
        "apo_bulk": (bulk[:, apo_idx] if apo_idx is not None
                     else np.zeros_like(bulk[:, atp_idx])),
        "ch_atp": df_all["listeners__replication_data__chromosomal_high_bound_atp"
                         ].to_numpy(),
        "ch_adp": df_all["listeners__replication_data__chromosomal_high_bound_adp"
                         ].to_numpy(),
        "oh_atp": df_all["listeners__replication_data__oriC_high_bound_atp"
                         ].to_numpy(),
        "oh_adp": df_all["listeners__replication_data__oriC_high_bound_adp"
                         ].to_numpy(),
        "ol_atp": df_all["listeners__replication_data__oriC_low_bound_atp"
                         ].to_numpy(),
        "pr_atp": df_all["listeners__replication_data__promoter_high_bound_atp"
                         ].to_numpy(),
        "pr_adp": df_all["listeners__replication_data__promoter_high_bound_adp"
                         ].to_numpy(),
        "gen_starts": gen_starts,
    }


def _smooth(x, w):
    if x.size <= w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def plot(d: dict, out_path: Path, title_extra: str) -> None:
    t = d["t_min"]
    total_atp = d["atp_bulk"] + d["ch_atp"] + d["oh_atp"] + d["ol_atp"] + d["pr_atp"]
    total_adp = d["adp_bulk"] + d["ch_adp"] + d["oh_adp"] + d["pr_adp"]
    total = total_atp + total_adp + d["apo_bulk"]

    ch_bound = d["ch_atp"] + d["ch_adp"]
    oh_bound = d["oh_atp"] + d["oh_adp"]
    pr_bound = d["pr_atp"] + d["pr_adp"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"{title_extra}\nDnaA species as raw molecule counts",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for g in d["gen_starts"]:
            ax.axvline(g, color="#cbd5e1", lw=0.6, zorder=0)

    # Panel 1: cell mass (fg)
    ax = axes[0]
    ax.plot(t, d["cell_mass_fg"], color="#0ea5e9", lw=1.0)
    ax.set_ylabel("Cell mass (fg)")
    vlines(ax)
    ax.set_title("Cell mass", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 2: bulk DnaA forms (raw counts, log scale). Tick-level stochastic
    # noise is visible on integer counts, so smooth all bulk traces.
    ax = axes[1]
    w = max(60, t.size // 200)
    ax.plot(t, _smooth(d["atp_bulk"], w), color="#16a34a", lw=1.0,
            label=f"bulk DnaA-ATP (smoothed, w={w})")
    ax.plot(t, _smooth(d["adp_bulk"], w), color="#dc2626", lw=1.0,
            label="bulk DnaA-ADP")
    ax.plot(t, _smooth(d["apo_bulk"], w), color="#475569", lw=0.8,
            label="bulk apo DnaA")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Bulk DnaA (molecules)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    vlines(ax)
    ax.set_title("Bulk (free) DnaA molecule counts",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 3: total cellular DnaA counts (bulk + bound)
    ax = axes[2]
    ax.plot(t, total, color="#0f172a", lw=1.2, label="total DnaA")
    ax.plot(t, total_atp, color="#16a34a", lw=1.0, label="total DnaA-ATP")
    ax.plot(t, total_adp, color="#dc2626", lw=1.0, label="total DnaA-ADP")
    ax.axhspan(300, 800, color="#bbf7d0", alpha=0.3, zorder=0,
               label="PDF target band [300, 800]")
    ax.set_ylabel("Total DnaA (molecules)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    vlines(ax)
    ax.set_title("Total cellular DnaA (bulk + all bound pools) — molecule counts",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 4: bound counts per pool. Small pools (oriC_high=3, promoter=2,
    # oriC_low=8) flip 1-2 boxes per tick from stochastic rounding — smooth
    # with the same window as the bulk panel.
    ax = axes[3]
    ax.plot(t, _smooth(ch_bound, w), color="#3730a3", lw=1.0,
            label=f"chromosomal_high bound (302+ sites, smoothed w={w})")
    ax.plot(t, _smooth(oh_bound, w), color="#6b21a8", lw=1.0,
            label="oriC_high bound (3 sites)")
    ax.plot(t, _smooth(d["ol_atp"], w), color="#0e7490", lw=1.2,
            label="oriC_low bound ATP (8 sites)")
    ax.plot(t, _smooth(pr_bound, w), color="#9a3412", lw=1.0,
            label="promoter_high bound (2 sites)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Bound DnaA (molecules)")
    ax.set_xlabel("Cumulative time (min)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    vlines(ax)
    ax.set_title("Bound DnaA counts per pool",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=1)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load_lineage(args.exp_root, args.exp_id, args.lineage_seed, args.gens)
    plot(d, Path(args.out), args.title_extra)


if __name__ == "__main__":
    main()
