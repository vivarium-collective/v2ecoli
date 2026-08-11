"""In-depth analysis of the dnaA-promoter DnaA-box pool (4 sites, K_d=1 nM).

Three panels:
  1. Time series — bound ATP / ADP stacked, with total-sites reference line.
  2. ATP-fraction-of-bound — instantaneous A/(A+D) at the bound pool, tracking
     how much of the bound DnaA is still in the active form vs hydrolyzed.
  3. Occupancy histogram — distribution of total-bound (0..4 sites) across the
     lineage, with per-form means in the panel title.

Usage:
    python scripts/plot_dnaa3_promoter_analysis.py \\
        --exp-root out/dnaa3_phase2_v1e3_steadystart_parquet/dnaa3_phase2_v1e3_steadystart \\
        --exp-id dnaa3_phase2_v1e3_steadystart \\
        --lineage-seed 7 --gens 8 \\
        --title-extra "V=1.0e-3 + steady-state start + seed=7" \\
        --out out/figures/dnaa3_promoter_analysis.png
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

ATP_COL = "listeners__replication_data__promoter_high_bound_atp"
ADP_COL = "listeners__replication_data__promoter_high_bound_adp"


def load(exp_root: str, exp_id: str, seed: int, n_gens: int):
    t, a, d, boundaries = [], [], [], []
    cols = ["global_time", ATP_COL, ADP_COL]
    for gen in range(1, n_gens + 1):
        agent = "0" * gen
        pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
               f"lineage_seed={seed}/generation={gen}/agent_id={agent}/*.pq")
        fs = sorted(glob.glob(pat),
                    key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
        if not fs:
            continue
        gen_t, gen_a, gen_d = [], [], []
        for f in fs:
            tbl = pq.read_table(f, columns=cols).to_pandas()
            gen_t.append(tbl["global_time"].to_numpy())
            gen_a.append(tbl[ATP_COL].to_numpy())
            gen_d.append(tbl[ADP_COL].to_numpy())
        offset = boundaries[-1] if boundaries else 0
        gen_t_arr = np.concatenate(gen_t) + offset
        t.append(gen_t_arr)
        a.append(np.concatenate(gen_a))
        d.append(np.concatenate(gen_d))
        boundaries.append(gen_t_arr[-1])
    return {
        "t": np.concatenate(t) if t else np.array([]),
        "atp": np.concatenate(a) if a else np.array([]),
        "adp": np.concatenate(d) if d else np.array([]),
        "gen_boundaries": boundaries[:-1] if boundaries else [],
    }


def plot(d: dict, out_path: Path, title_extra: str, n_sites: int = 4) -> None:
    t_min = d["t"] / 60
    boundaries_min = [b / 60 for b in d["gen_boundaries"]]
    a = d["atp"].astype(float)
    adp = d["adp"].astype(float)
    bound = a + adp
    free = n_sites - bound
    atp_frac_of_bound = np.where(bound > 0, a / bound, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [1.3, 1.0, 1.0]})
    fig.suptitle(
        f"{title_extra}\n"
        f"dnaA-promoter pool (K_d=1 nM, {n_sites} sites) — in-depth analysis\n"
        f"Overall: mean bound {bound.mean():.2f}/{n_sites}  "
        f"(ATP {a.mean():.2f} | ADP {adp.mean():.2f}),  "
        f"never-empty {100 * (bound > 0).mean():.1f}% of ticks",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for b in boundaries_min:
            ax.axvline(b, color="#94a3b8", lw=1.0, ls="--",
                       alpha=0.6, zorder=0)

    # Panel 1 — stacked ATP / ADP / empty with total reference line.
    ax = axes[0]
    ax.fill_between(t_min, 0, a, color="#dc2626", alpha=0.85,
                    label=f"bound DnaA-ATP (mean {a.mean():.2f})")
    ax.fill_between(t_min, a, a + adp, color="#7c3aed", alpha=0.7,
                    label=f"bound DnaA-ADP (mean {adp.mean():.2f})")
    ax.fill_between(t_min, a + adp, n_sites, color="#cbd5e1", alpha=0.5,
                    label=f"empty (mean {free.mean():.2f})")
    ax.axhline(n_sites, color="#0f172a", lw=1.0, ls=":",
               label=f"{n_sites} total sites")
    vlines(ax)
    ax.set_ylabel("Promoter-box state\n(stacked counts)")
    ax.set_ylim(0, n_sites + 0.2)
    ax.legend(loc="lower right", fontsize=9, frameon=False, ncol=4)

    # Panel 2 — ATP-fraction-of-bound (active-form fraction at the promoter).
    ax = axes[1]
    ax.plot(t_min, atp_frac_of_bound, color="#0f766e", lw=1.1,
            label=f"DnaA-ATP / (ATP+ADP) bound  (mean "
                  f"{np.nanmean(atp_frac_of_bound):.3f})")
    vlines(ax)
    ax.set_ylabel("ATP-form fraction\nof bound DnaA")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (min)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # Panel 3 — occupancy histogram (counts of n-sites-bound).
    ax = axes[2]
    edges = np.arange(-0.5, n_sites + 1.5, 1.0)
    centers = np.arange(0, n_sites + 1)
    counts = np.array([(bound == k).sum() for k in centers])
    fracs = counts / counts.sum()
    bars = ax.bar(centers, fracs, color="#1d4ed8", alpha=0.85, edgecolor="#0f172a")
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{100 * frac:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(centers)
    ax.set_xlabel("Number of promoter sites bound (out of 4)")
    ax.set_ylabel("Fraction of ticks")
    ax.set_ylim(0, max(0.05, fracs.max() * 1.18))
    ax.set_title(
        f"Distribution across {bound.size:,} ticks (8 gens)",
        fontsize=10, loc="left", pad=4,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--n-sites", type=int, default=4,
                    help="Total promoter sites in the pool (default 4).")
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load(args.exp_root, args.exp_id, args.lineage_seed, args.gens)
    if d["t"].size == 0:
        raise SystemExit(f"no parquet found for {args.exp_id}")
    plot(d, Path(args.out), args.title_extra, n_sites=args.n_sites)


if __name__ == "__main__":
    main()
