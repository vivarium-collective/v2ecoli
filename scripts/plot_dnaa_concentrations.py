"""DnaA concentrations (counts / cell-volume) across a multigen lineage.

Renders bulk and total DnaA species in molar units with K_d reference lines,
so the reader can see whether free DnaA-ATP is above or below the K_d for
the high-aff / low-aff oriC sites.

Usage:
    python scripts/plot_dnaa_concentrations.py \\
        --exp-root out/<exp>_parquet/<exp> \\
        --exp-id <exp> \\
        --lineage-seed 1 --gens 8 \\
        --out out/figures/<exp>_8gen_concentrations.png
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

N_A = 6.022e23
CELL_DENSITY_G_PER_L = 1100.0

DNAA_ATP = "MONOMER0-160[c]"
DNAA_ADP = "MONOMER0-4565[c]"
DNAA_APO = "PD03831[c]"

K_D_HIGH_NM = 1.0
K_D_LOW_NM = 100.0


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
        df = (rs[0].__class__.__mro__[1] if False else __import__("pandas")
              ).concat(rs).sort_values("global_time").reset_index(drop=True)
        gen_starts.append(cum_t)
        df["t_rel"] = (df["global_time"] - df["global_time"].iloc[0]) / 60.0
        df["t_cum_min"] = cum_t + df["t_rel"]
        cum_t = float(df["t_cum_min"].iloc[-1])
        rows_all.append(df)
    df_all = (__import__("pandas")).concat(rows_all).reset_index(drop=True)

    bulk = np.stack(df_all["bulk__count"].to_numpy())
    t_min = df_all["t_cum_min"].to_numpy()
    cell_mass_fg = df_all["listeners__mass__cell_mass"].to_numpy()
    V_L = cell_mass_fg * 1e-15 / CELL_DENSITY_G_PER_L
    return {
        "t_min": t_min,
        "V_L": V_L,
        "V_fL": V_L * 1e15,
        "atp_bulk": bulk[:, atp_idx],
        "adp_bulk": bulk[:, adp_idx],
        "apo_bulk": bulk[:, apo_idx] if apo_idx is not None else np.zeros_like(
            bulk[:, atp_idx]),
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
        "n_oric": df_all["listeners__replication_data__number_of_oric"].to_numpy(),
        "gen_starts": gen_starts,
    }


def count_to_nM(count: np.ndarray, V_L: np.ndarray) -> np.ndarray:
    """molecules / volume[L] / N_A → molar → multiply by 1e9 → nM."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return count / V_L / N_A * 1e9


def plot(d: dict, out_path: Path, title_extra: str) -> None:
    t = d["t_min"]
    V_L = d["V_L"]
    bulk_atp_nm = count_to_nM(d["atp_bulk"], V_L)
    bulk_adp_nm = count_to_nM(d["adp_bulk"], V_L)
    bulk_apo_nm = count_to_nM(d["apo_bulk"], V_L)
    total_atp = d["atp_bulk"] + d["ch_atp"] + d["oh_atp"] + d["ol_atp"] + d["pr_atp"]
    total_adp = d["adp_bulk"] + d["ch_adp"] + d["oh_adp"] + d["pr_adp"]
    total_dnaa_count = total_atp + total_adp + d["apo_bulk"]
    total_dnaa_nm = count_to_nM(total_dnaa_count, V_L)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"{title_extra}\nDnaA species expressed as concentrations "
        "(counts / cell volume; cell_density = 1.1 g/mL)",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for g in d["gen_starts"]:
            ax.axvline(g, color="#cbd5e1", lw=0.6, zorder=0)

    # Panel 1: cell volume (fL)
    ax = axes[0]
    ax.plot(t, d["V_fL"], color="#0ea5e9", lw=1.0)
    ax.set_ylabel("Cell volume (fL)")
    vlines(ax)
    ax.set_title("Cell volume", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 2: bulk DnaA forms in nM with K_d reference lines
    ax = axes[1]
    ax.axhline(K_D_HIGH_NM, color="#9333ea", ls=":", lw=1.0,
               label=f"K_d high-aff = {K_D_HIGH_NM:.0f} nM")
    ax.axhline(K_D_LOW_NM, color="#a16207", ls=":", lw=1.0,
               label=f"K_d low-aff  = {K_D_LOW_NM:.0f} nM")
    # Tick-level stochastic-rounding noise on integer molecule counts is ±1-2
    # molecules per tick — visible at this concentration scale. Smooth all
    # bulk traces with the same rolling-mean window to suppress per-tick
    # noise while preserving cycle-scale dynamics.
    w = max(60, t.size // 200) if t.size > 60 else 1
    kernel = np.ones(w) / w if w > 1 else None
    smooth = (lambda x: np.convolve(x, kernel, mode="same")) if kernel is not None else (lambda x: x)
    ax.plot(t, smooth(bulk_atp_nm), color="#16a34a", lw=1.0,
            label=f"bulk DnaA-ATP [nM] (smoothed, w={w})")
    ax.plot(t, smooth(bulk_adp_nm), color="#dc2626", lw=1.0,
            label=f"bulk DnaA-ADP [nM]")
    ax.plot(t, smooth(bulk_apo_nm), color="#475569", lw=0.8,
            label=f"bulk apo DnaA [nM]")
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(1e4, np.nanmax(bulk_adp_nm) * 2))
    ax.set_ylabel("Bulk concentration (nM)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    vlines(ax)
    ax.set_title("Bulk (free) DnaA concentrations + K_d reference",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 3: total cellular DnaA concentration (bulk + bound)
    ax = axes[2]
    ax.plot(t, total_dnaa_nm, color="#0f172a", lw=1.2,
            label="total DnaA [nM]")
    ax.plot(t, count_to_nM(total_atp, V_L), color="#16a34a", lw=1.0,
            label="total DnaA-ATP [nM]")
    ax.plot(t, count_to_nM(total_adp, V_L), color="#dc2626", lw=1.0,
            label="total DnaA-ADP [nM]")
    ax.set_ylabel("Total concentration (nM)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    vlines(ax)
    ax.set_title("Total cellular DnaA (bulk + all bound pools)",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 4: free-vs-K_d ratio (the gate signal); same smoothing window
    ax = axes[3]
    ratio_high = smooth(bulk_atp_nm) / K_D_HIGH_NM
    ratio_low = smooth(bulk_atp_nm) / K_D_LOW_NM
    ax.axhline(1.0, color="#94a3b8", ls="--", lw=0.8,
               label="ratio = 1 (50% saturation)")
    ax.plot(t, ratio_high, color="#9333ea", lw=1.0,
            label="[ATP_free] / K_d_high (smoothed)")
    ax.plot(t, ratio_low, color="#a16207", lw=1.0,
            label="[ATP_free] / K_d_low (smoothed)")
    ax.set_yscale("log")
    ax.set_ylabel("Free / K_d")
    ax.set_xlabel("Cumulative time (min)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    vlines(ax)
    ax.set_title("Free DnaA-ATP relative to K_d "
                 "(>1 → site saturating; <1 → undersaturated)",
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
