#!/usr/bin/env python
"""Figures for dnaa-9 (async over-replication problem) and dnaa-10 (SeqA fix).

Rashmi's arc (2026-07-02): the DnaA-ATP fraction is tuned into [0.2,0.5] by the
HYDROLYSIS rate (DNAA_HYDROLYSIS_RATE_PER_MIN=0.025) with the extrinsic mechanisms
(RIDA/DDAH/DARS) OFF. Each oriC binds DnaA independently, so replication is
ASYNCHRONOUS (DNAA_ASYNC_INITIATION) — but async ALONE over-replicates (oriC 3/4/5,
DnaA climbs) because nothing throttles re-initiation frequency. Adding SeqA (the
per-origin eclipse) FIXES it: controlled asynchronous once-per-cycle, DnaA back in
band. RIDA stays off throughout.

  A. dnaa9_async_overreplication — the problem: 4-panel trajectory of the async
     config with no re-init control (ATP fraction in band, but oriC runs to 3/4/5
     and total DnaA climbs above the [300,800] band).
  B. dnaa10_seqa_fix — before/after: oriC count and total DnaA, async-no-SeqA
     (over-replicates) vs async+SeqA (controlled once-per-cycle, DnaA in band).

  python scripts/render_dnaa910.py
"""
from __future__ import annotations
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from scripts.render_dnaa9_operating_point import _load, RD
from scripts.pbg_plot_style import PALETTE, style_axes, mark_lineages
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "svg.fonttype": "none"})


def _meta(path, title, caption, interp, runs):
    json.dump({"title": title, "caption": caption, "interpretation": interp,
               "source_runs": runs, "script": "scripts/render_dnaa910.py"},
              open(path + ".meta.json", "w"), indent=2)


def problem_fig(prob_dir, study_dir):
    df, bounds = _load(prob_dir)
    if df is None:
        raise SystemExit(f"no data {prob_dir}")
    x = df["t_min"].to_numpy()
    afr = float(np.mean(df.filter(pl.col("generation") >= 4)["atp_frac_total"].to_numpy()))
    fig, ax = plt.subplots(4, 1, figsize=(8.8, 9.6), sharex=True)
    ax[0].axhspan(0.2, 0.5, color=PALETTE["blue"], alpha=0.10, lw=0)
    ax[0].plot(x, df["atp_frac_total"].to_numpy(), color=PALETTE["blue"], lw=1.2)
    style_axes(ax[0], "", "DnaA-ATP fraction"); mark_lineages(ax[0], bounds); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"DnaA-ATP fraction IN BAND via hydrolysis (k_h=0.025), mean {afr:.2f} — the fraction is fixed")
    ax[1].plot(x, df["oric_low_occ"].to_numpy(), color=PALETTE["green"], lw=1.1)
    style_axes(ax[1], "", "oriC-low occupancy"); mark_lineages(ax[1], bounds); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("oriC-low occupancy fires at full 1.0 (threshold 8)")
    ax[2].step(x, df[RD + "number_of_oric"].to_numpy(), color=PALETTE["red"], lw=1.4, where="post")
    style_axes(ax[2], "", "oriC count"); mark_lineages(ax[2], bounds)
    ax[2].set_ylim(0.5, 5.5); ax[2].set_yticks([1, 2, 3, 4, 5])
    ax[2].set_title("PROBLEM — oriC OVER-REPLICATES (3/4/5): async with no re-initiation control")
    ax[3].axhspan(300, 800, color="0.5", alpha=0.10, lw=0)
    ax[3].plot(x, df["total_dnaa"].to_numpy(), color=PALETTE["axis"], lw=1.2)
    style_axes(ax[3], "lineage time (min)", "total DnaA"); mark_lineages(ax[3], bounds)
    ax[3].set_title("total DnaA climbs ABOVE the [300,800] band (extra origins → extra gene dosage)")
    fig.suptitle("dnaa-9 — asynchronous initiation (each oriC binds DnaA independently) OVER-REPLICATES\n"
                 "hydrolysis fixes the DnaA-ATP fraction, threshold 8 fires at full occupancy, but with no "
                 "re-initiation control the origins keep re-firing (RIDA/DDAH/DARS off)", fontsize=10.5, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    base = os.path.join(study_dir, "charts", "dnaa9_async_overreplication")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-9 — asynchronous initiation over-replicates without a re-initiation control",
          f"Config: DNAA_HYDROLYSIS_RATE_PER_MIN=0.025 + threshold 8 + async, RIDA/DDAH/DARS off "
          f"(seed 1). The hydrolysis rate puts the total DnaA-ATP fraction in the [0.2,0.5] band "
          f"(mean {afr:.2f}) and the trigger fires at full oriC-low occupancy (1.0). But because each "
          f"oriC binds DnaA and initiates INDEPENDENTLY with nothing throttling re-initiation, the "
          f"origins re-fire continuously — oriC runs to 3/4/5 and total DnaA climbs above [300,800].",
          "Two of the targets (DnaA-ATP fraction in band via hydrolysis; full-occupancy firing) are met, "
          "and per-origin asynchronous initiation is in place — but async ALONE over-replicates. A "
          "re-initiation frequency control (SeqA) is required; that is dnaa-10.",
          [os.path.basename(prob_dir)])
    print(f"  wrote {base}.png/.svg (ATP fraction {afr:.2f})")


def fix_fig(prob_dir, fix_dir, study_dir):
    dp, bp = _load(prob_dir); dfx, bf = _load(fix_dir)
    if dp is None or dfx is None:
        raise SystemExit("missing problem/fix data")
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.6), sharex="col")
    for c, (d, b, lab, col) in enumerate([
            (dp, bp, "async, NO SeqA — over-replicates", PALETTE["red"]),
            (dfx, bf, "async + SeqA eclipse — controlled", PALETTE["green"])]):
        x = d["t_min"].to_numpy()
        ax[0][c].step(x, d[RD + "number_of_oric"].to_numpy(), color=col, lw=1.4, where="post")
        style_axes(ax[0][c], "", "oriC count" if c == 0 else ""); mark_lineages(ax[0][c], b)
        ax[0][c].set_ylim(0.5, 5.5); ax[0][c].set_yticks([1, 2, 3, 4, 5]); ax[0][c].set_title(lab, fontsize=10.5)
        ax[1][c].axhspan(300, 800, color="0.5", alpha=0.10, lw=0)
        ax[1][c].plot(x, d["total_dnaa"].to_numpy(), color=PALETTE["axis"], lw=1.2)
        style_axes(ax[1][c], "lineage time (min)", "total DnaA" if c == 0 else ""); mark_lineages(ax[1][c], b)
        ax[1][c].set_ylim(0, 950)
    fig.suptitle("dnaa-10 — SeqA (per-origin eclipse) fixes the asynchronous over-replication\n"
                 "adding SeqA throttles re-initiation → oriC returns to controlled 1↔2 (+async 2↔3)\n"
                 "and total DnaA back in the [300,800] band (hydrolysis 0.025 + threshold 8 + async, RIDA off)",
                 fontsize=10, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    base = os.path.join(study_dir, "charts", "dnaa10_seqa_fix")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-10 — SeqA eclipse fixes the asynchronous over-replication (RIDA stays off)",
          "Same config as dnaa-9 (hydrolysis 0.025 + threshold 8 + async, RIDA/DDAH/DARS off), seed 1. "
          "LEFT (no SeqA): oriC over-replicates to 3/4/5 and total DnaA climbs above [300,800]. RIGHT "
          "(+ SeqA per-origin eclipse): re-initiation is throttled to once per origin per cycle — oriC "
          "returns to a controlled 1↔2 with decoupled 2↔3 async episodes, and total DnaA is back in the "
          "[300,800] band. The DnaA-ATP fraction stays in [0.2,0.5] (still set by hydrolysis, not RIDA).",
          "SeqA (the post-initiation sequestration that blocks a just-fired origin from re-firing) is the "
          "re-initiation FREQUENCY control that asynchronous initiation needs. With it, all four targets "
          "hold together: DnaA-ATP fraction in band (hydrolysis), full-occupancy firing (threshold 8), "
          "per-origin asynchronous initiation, and once-per-cycle DnaA homeostasis — RIDA not required.",
          [os.path.basename(prob_dir), os.path.basename(fix_dir)])
    print(f"  wrote {base}.png/.svg")


if __name__ == "__main__":
    d9 = "workspace/studies/dnaa-9-async-initiation"
    d10 = "workspace/studies/dnaa-10-seqa-reinit-control"
    for d in (d9, d10):
        os.makedirs(os.path.join(d, "charts"), exist_ok=True)
        os.makedirs(os.path.join(d, "analyses"), exist_ok=True)
    problem_fig("out/opclean_s1", d9)
    fix_fig("out/opclean_s1", "out/fixSeqA_s1", d10)
