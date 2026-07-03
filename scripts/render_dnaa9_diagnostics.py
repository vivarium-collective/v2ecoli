#!/usr/bin/env python
"""Diagnostic plots for dnaa-9 requested by Rashmi (2026-07-02):
actual oriC-low box COUNTS (not the fraction), cell mass, and bulk DnaA-ATP level,
with the cooperativity n / K labeled. Also plots oriC at full resolution so the
per-event +1 steps are visible (the apparent 1->3 jump is a downsampling artifact
of the over-replicating regime; each initiation is +1).

  python scripts/render_dnaa9_diagnostics.py --run out/opclean_s1 \
      --study-dir workspace/studies/dnaa-9-async-initiation --n 4 --K 30
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.pbg_plot_style import PALETTE, style_axes, mark_lineages

RD = "listeners__replication_data__"
ATP = "MONOMER0-160[c]"   # free DnaA-ATP bulk species
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "svg.fonttype": "none"})


def load(run_dir, downsample=0):
    files = [p for p in glob.glob(os.path.join(run_dir, "**", "history", "**", "*.pq"), recursive=True)
             if re.search(r"agent_id=0+/", p + "/")]
    ids = pl.scan_parquet(files[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    i_atp = ids.index(ATP)
    bc = pl.col("bulk__count")
    need = [RD + "number_of_oric", RD + "oriC_low_bound_atp", RD + "oriC_low_free",
            "listeners__mass__cell_mass"]
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains(r"^0+$"))
          .select(["generation", "global_time"] + need + [bc.list.get(i_atp).alias("bulk_dnaa_atp")])
          .sort(["generation", "global_time"]).collect())
    dur = (df.group_by("generation").agg(((pl.col("global_time").max() - pl.col("global_time").min()) / 60.0).alias("_d")))
    df = df.filter(pl.col("generation").is_in(dur.filter(pl.col("_d") >= 5.0)["generation"].to_list()))
    offset, cum, bounds = 0.0, [], []
    for g in sorted(df["generation"].unique().to_list()):
        t = df.filter(pl.col("generation") == g)["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0); offset += float(t.max()); bounds.append(offset / 60.0)
    df = df.with_columns(pl.Series("t_min", cum))
    if downsample and df.height > downsample:
        df = df.gather_every(max(1, df.height // downsample))
    return df, bounds[:-1]


def render(run_dir, study_dir, n, K):
    # oriC at full resolution (no downsample) so +1 steps are visible; others lightly sampled
    dfull, bounds = load(run_dir, downsample=0)
    df, _ = load(run_dir, downsample=6000)
    x = df["t_min"].to_numpy(); xf = dfull["t_min"].to_numpy()
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    # 1 cell mass
    ax[0].plot(x, df["listeners__mass__cell_mass"].to_numpy(), color=PALETTE["axis"], lw=1.3)
    style_axes(ax[0], "", "cell mass (fg)"); mark_lineages(ax[0], bounds)
    ax[0].set_title("Cell mass")
    # 2 oriC count (full resolution — every initiation is +1, 1->2->3->4)
    ax[1].step(xf, dfull[RD + "number_of_oric"].to_numpy(), color=PALETTE["purple"], lw=1.0, where="post")
    style_axes(ax[1], "", "oriC count"); mark_lineages(ax[1], bounds)
    ax[1].set_ylim(0.5, 5.5); ax[1].set_yticks([1, 2, 3, 4, 5])
    ax[1].set_title("oriC count (full resolution — each initiation is +1; the over-replication makes 1->2->3 fast)")
    # 3 oriC-low bound COUNT (boxes) with the per-origin fire threshold
    bound = df[RD + "oriC_low_bound_atp"].to_numpy().astype(float)
    noric = df[RD + "number_of_oric"].to_numpy().astype(float)
    ax[2].plot(x, bound, color=PALETTE["green"], lw=1.1, label="oriC-low DnaA-ATP bound (boxes)")
    ax[2].plot(x, 8 * noric, color=PALETTE["muted"], lw=1.0, ls="--", label="8 boxes x n_oriC (all sites)")
    style_axes(ax[2], "", "bound boxes (count)"); mark_lineages(ax[2], bounds)
    ax[2].legend(fontsize=8, loc="upper right", frameon=False)
    ax[2].set_title("oriC-low bound boxes (COUNT) — each origin fires when ITS 8 sites fill (per-origin 8/8), "
                    "so the global count tracks 8xn_oriC")
    # 4 bulk DnaA-ATP level
    ax[3].plot(x, df["bulk_dnaa_atp"].to_numpy(), color=PALETTE["blue"], lw=1.1)
    style_axes(ax[3], "", "bulk DnaA-ATP\n(free, count)"); mark_lineages(ax[3], bounds)
    ax[3].set_title("Bulk (free) DnaA-ATP level — MONOMER0-160[c]")
    # 5 oriC-low occupancy fraction (for reference; note it is the GLOBAL average)
    occ = bound / np.maximum(bound + df[RD + "oriC_low_free"].to_numpy().astype(float), 1e-9)
    ax[4].plot(x, occ, color=PALETTE["amber"], lw=1.0)
    style_axes(ax[4], "lineage time (min)", "oriC-low occupancy\n(GLOBAL fraction)"); mark_lineages(ax[4], bounds)
    ax[4].set_ylim(0, 1.05)
    ax[4].set_title("oriC-low occupancy FRACTION (global average — dips to ~0.6 when one origin is full and another "
                    "is refilling; NOT partial firing)")
    fig.suptitle(f"dnaa-9 diagnostics — cooperativity n={n}, K={K} nM · mechanistic oriC-low trigger, threshold 8 "
                 f"(per-origin 8/8), async, hydrolysis 0.025\n(RIDA/DDAH/DARS off; seed from run)",
                 fontsize=10.5, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    base = os.path.join(study_dir, "charts", "dnaa9_diagnostics")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".svg"); plt.close(fig)
    json.dump({"title": f"dnaa-9 diagnostics — box counts, cell mass, bulk DnaA-ATP (n={n}, K={K})",
               "caption": f"Cooperativity n={n}, K={K} nM. Panels: cell mass; oriC count at full resolution "
               f"(every initiation is +1 — the apparent 1->3 is downsampling of the fast 1->2->3 re-firing); "
               f"oriC-low bound-box COUNT with the 8xn_oriC line (each origin fires at its own 8/8); bulk free "
               f"DnaA-ATP; and the GLOBAL occupancy fraction (which dips to ~0.6 when one origin is full and "
               f"another refilling — a global-average artifact, not partial firing).",
               "interpretation": "Clarifies Rashmi's two questions: the trigger fires per-origin at full 8/8 "
               "occupancy (the ~0.6 is the global average across a full + a refilling origin), and every oriC "
               "step is +1 (the 1->3 is a plot-resolution artifact of the over-replicating re-firing).",
               "source_runs": [os.path.basename(run_dir)], "script": "scripts/render_dnaa9_diagnostics.py"},
              open(base + ".png.meta.json", "w"), indent=2)
    print(f"  wrote {base}.png/.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/opclean_s1")
    ap.add_argument("--study-dir", default="workspace/studies/dnaa-9-async-initiation")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--K", type=int, default=30)
    a = ap.parse_args()
    render(a.run, a.study_dir, a.n, a.K)


if __name__ == "__main__":
    main()
