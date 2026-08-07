#!/usr/bin/env python
"""Aesthetic rebuild of the dnaa-6 figures, consistent with finding F-04.

The old dnaa-6 charts told the superseded over-initiation story (oriC→4). F-04
(2026-06-27) established that on current code the cooperative oriC-low +
mechanistic-low trigger gives CLEAN once-per-cell-cycle initiation (oriC 1↔2).
This rebuilds the figures from fresh in-sim data (the dnaa-8 n×K sweep, seed 1)
so they are both prettier and TRUE to the current finding — no extra runs:

  A. dnaa6_once_per_cycle  — the payoff at the reference operating point (n=4,
     K=30): cell-mass sawtooth, oriC stepping 1↔2, and the oriC-low occupancy
     fill→fire→reset, one clean cycle per generation.
  B. dnaa6_cooperativity_contrast — WHY cooperativity is needed: n=1 (gradual
     Langmuir, dnaa-3 regime) vs n=4 (sharp switch). Same K=30. Shows the
     oriC-low occupancy only becomes a clean all-or-none switch with cooperativity.

Reads the sweep run dirs via scripts.dnaa_sweep_analysis.load_trajectory.

  python scripts/render_dnaa6_figures.py \
      --ref out/nk_n4_K30 --weak out/nk_n1_K30 \
      --study-dir workspace/studies/dnaa-6-mechanistic-initiation
"""
from __future__ import annotations
import argparse, json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dnaa_sweep_analysis import load_trajectory, compute_metric, SAT
from scripts.pbg_plot_style import PALETTE, style_axes, mark_lineages

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "figure.dpi": 150, "svg.fonttype": "none",
})


def _meta(path, title, caption, interp, runs, script):
    json.dump({"title": title, "caption": caption, "interpretation": interp,
               "source_runs": runs, "script": script},
              open(path + ".meta.json", "w"), indent=2)


def once_per_cycle(ref_dir, study_dir):
    df, bounds = load_trajectory(ref_dir)
    if df is None:
        raise SystemExit(f"no data in {ref_dir}")
    m = compute_metric(ref_dir)
    x = df["t_min"].to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(8.6, 8.0), sharex=True)
    # 1 — cell mass sawtooth
    ax[0].plot(x, df["cell_mass"].to_numpy(), color=PALETTE["axis"], lw=1.5)
    style_axes(ax[0], "", "cell mass (fg)"); mark_lineages(ax[0], bounds)
    ax[0].set_title("Cell mass — a clean division sawtooth every generation")
    # 2 — oriC count 1<->2
    ax[1].step(x, df["number_of_oric"].to_numpy(), color=PALETTE["purple"],
               lw=1.7, where="post")
    style_axes(ax[1], "", "oriC count"); mark_lineages(ax[1], bounds)
    ax[1].set_ylim(0.5, 2.5); ax[1].set_yticks([1, 2])
    ax[1].set_title("Number of oriC — strictly 1↔2 (one initiation per cycle, no re-init)")
    # 3 — oriC-low occupancy fill->fire->reset
    ax[2].plot(x, df["oric_low_occ"].to_numpy(), color=PALETTE["green"], lw=1.4)
    style_axes(ax[2], "lineage time (min)", "oriC-low occupancy"); mark_lineages(ax[2], bounds)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("oriC-low DnaA-ATP occupancy — fill → fire → reset, once per generation")
    fig.suptitle("dnaa-6 — mechanistic DnaA-ATP/oriC initiation: clean once-per-cell-cycle "
                 f"(n=4, K=30, seed 1)\nmean total DnaA {m['dnaa_mean']:.0f} counts "
                 f"(band [300,800]) · {m['n_gens']} generations",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = os.path.join(study_dir, "charts", "dnaa6_once_per_cycle")
    fig.savefig(base + ".png"); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-6 payoff — mechanistic DnaA-ATP/oriC initiation gives clean once-per-cell-cycle replication",
          f"Reference operating point (n=4, K=30 nM), mechanistic oriC-low trigger, seed 1, "
          f"{m['n_gens']} generations. TOP: cell mass sawtooths and divides every generation. "
          f"MIDDLE: oriC count steps strictly 1↔2 — exactly one initiation per cycle, no "
          f"re-initiation (the over-initiation to oriC→4 in the earlier build does not reproduce; "
          f"F-04). BOTTOM: oriC-low DnaA-ATP occupancy fills, fires, and resets once per "
          f"generation. Total cellular DnaA holds the [300,800] homeostatic band (mean "
          f"{m['dnaa_mean']:.0f}).",
          "Biology-driven replication initiation (DnaA-ATP filament at oriC) now drives a clean, "
          "periodic cell cycle in place of the cell-mass heuristic: one fill→fire→reset per "
          "generation, oriC 1↔2, DnaA homeostatic. This is finding F-04 shown directly.",
          [os.path.basename(ref_dir)], "scripts/render_dnaa6_figures.py")
    print(f"  wrote {base}.png/.svg")


def cooperativity_contrast(weak_dir, ref_dir, study_dir):
    dw, bw = load_trajectory(weak_dir)
    dr, br = load_trajectory(ref_dir)
    mw, mr = compute_metric(weak_dir), compute_metric(ref_dir)
    if dw is None or dr is None:
        raise SystemExit("missing weak/ref data")
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    for c, (d, b, m, label, color) in enumerate([
            (dw, bw, mw, "n = 1  (gradual Langmuir)", PALETTE["amber"]),
            (dr, br, mr, "n = 4  (sharp cooperative switch)", PALETTE["green"])]):
        x = d["t_min"].to_numpy()
        # occupancy row — sharpness of the switch
        ax[0][c].plot(x, d["oric_low_occ"].to_numpy(), color=color, lw=1.4)
        style_axes(ax[0][c], "", "oriC-low occupancy" if c == 0 else "")
        mark_lineages(ax[0][c], b); ax[0][c].set_ylim(0, 1.05)
        ax[0][c].set_title(f"{label}   ·   DnaA drift {m['dnaa_drift']:.2f}, CV {m['dnaa_cv']:.2f}",
                           fontsize=10.5)
        # total DnaA row — steadiness of the homeostatic band
        ax[1][c].axhspan(300, 800, color="0.5", alpha=0.10, lw=0)
        ax[1][c].plot(x, d["total_dnaa"].to_numpy(), color=PALETTE["blue"], lw=1.3)
        style_axes(ax[1][c], "lineage time (min)", "total DnaA (counts)" if c == 0 else "")
        mark_lineages(ax[1][c], b); ax[1][c].set_ylim(0, 900)
    fig.suptitle("dnaa-6 — cooperativity sharpens the oriC-low switch and steadies DnaA homeostasis\n"
                 "both cycle once-per-generation (oriC 1↔2), but the cooperative switch (n=4) fires "
                 "in a sharp spike and holds DnaA flatter in [300,800]\n(same K=30 nM, mechanistic "
                 "oriC-low trigger, seed 1)", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    base = os.path.join(study_dir, "charts", "dnaa6_cooperativity_contrast")
    fig.savefig(base + ".png"); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-6 — cooperativity sharpens the oriC-low switch and steadies DnaA homeostasis",
          "Same K=30 nM, mechanistic oriC-low trigger, seed 1. Both n=1 and n=4 give "
          "once-per-cycle replication (oriC 1↔2), so cooperativity is not strictly required for "
          "cycling here — but it improves the switch. TOP (oriC-low occupancy): n=1 (left) fills "
          "gradually in a staircase (independent-site Langmuir, the dnaa-3 regime); n=4 (right) "
          "switches in a sharp spike. BOTTOM (total DnaA): the cooperative switch holds DnaA "
          f"steadier in the [300,800] band (drift {mr['dnaa_drift']:.2f}, CV {mr['dnaa_cv']:.2f}) "
          f"than n=1 (drift {mw['dnaa_drift']:.2f}, CV {mw['dnaa_cv']:.2f}).",
          "Cooperativity is a sharpening/homeostasis knob, not an on/off switch for cycling: at "
          "K=30 both n=1 and n=4 cycle once-per-generation, but higher n gives a crisper oriC-low "
          "trigger and a flatter DnaA band. This refines the dnaa-5 rationale — the operating point "
          "n=4 is chosen for switch sharpness + DnaA steadiness, confirmed in-sim by dnaa-8.",
          [os.path.basename(weak_dir), os.path.basename(ref_dir)],
          "scripts/render_dnaa6_figures.py")
    print(f"  wrote {base}.png/.svg")


def _load_collapse(run_dir):
    """oriC-high trigger signal + threshold + stitched-minutes cell mass, followed lineage."""
    import glob, re
    import numpy as np
    import polars as pl
    RD = "listeners__replication_data__"
    files = [p for p in glob.glob(os.path.join(run_dir, "**", "history", "**", "*.pq"),
                                  recursive=True)
             if re.search(r"agent_id=0+/", p + "/")]
    if not files:
        return None
    d = (pl.scan_parquet(files, hive_partitioning=True)
         .filter(pl.col("agent_id").cast(pl.Utf8).str.contains(r"^0+$"))
         .select(["generation", "global_time", RD + "number_of_oric",
                  RD + "oriC_high_bound_atp", "listeners__mass__cell_mass"])
         .sort(["generation", "global_time"]).collect())
    offset, cum = 0.0, []
    for g in sorted(d["generation"].unique().to_list()):
        t = d.filter(pl.col("generation") == g)["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0); offset += float(t.max())
    d = d.with_columns(pl.Series("t_min", cum))
    return d


def collapse_figure(firstpass_dir, ref_dir, study_dir):
    import numpy as np
    RD = "listeners__replication_data__"
    fp = _load_collapse(firstpass_dir)
    rf = _load_collapse(ref_dir)
    if fp is None or rf is None:
        print("  (collapse figure skipped — missing first-pass data)")
        return
    fig, ax = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True)
    # A — cell mass: first-pass runs away vs cooperative oriC-low sustains
    ax[0].plot(rf["t_min"].to_numpy(), rf["listeners__mass__cell_mass"].to_numpy(),
               color=PALETTE["green"], lw=1.5, label="cooperative oriC-low trigger (n=4) — sustains")
    ax[0].plot(fp["t_min"].to_numpy(), fp["listeners__mass__cell_mass"].to_numpy(),
               color=PALETTE["red"], lw=1.6,
               label="first-pass oriC-high trigger (no cooperativity) — runs away")
    style_axes(ax[0], "", "cell mass (fg)")
    mmax = float(fp["listeners__mass__cell_mass"].max())
    ax[0].legend(fontsize=9, loc="upper left", frameon=False)
    ax[0].set_title("Cell mass — the naive oriC-high trigger fails to initiate and grows unbounded")
    # B — trigger signal vs threshold (3 x n_oriC)
    x = fp["t_min"].to_numpy()
    sig = fp[RD + "oriC_high_bound_atp"].to_numpy().astype(float)
    thr = 3.0 * fp[RD + "number_of_oric"].to_numpy().astype(float)
    ax[1].plot(x, sig, color=PALETTE["green"], lw=1.3, label="oriC-high DnaA-ATP bound (trigger signal)")
    ax[1].plot(x, thr, color=PALETTE["purple"], lw=1.3, ls="--", label="threshold = 3 × n_oriC")
    style_axes(ax[1], "lineage time (min)", "oriC-high\nDnaA-ATP (count)")
    ax[1].legend(fontsize=9, loc="upper right", frameon=False)
    ax[1].set_title("Trigger signal never reaches threshold — independent-site binding "
                    "under-saturates (no cooperativity)")
    fig.suptitle(f"dnaa-6 — first-pass mechanistic trigger (oriC-high, no cooperativity) collapses "
                 f"the cell cycle\ncell runs away to ~{mmax:.0f} fg without dividing (seed 1) — "
                 f"the diagnostic that motivates the cooperative switch", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    base = os.path.join(study_dir, "charts", "dnaa6_mechanistic_collapse")
    fig.savefig(base + ".png"); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-6 — the first-pass mechanistic trigger (oriC-high, no cooperativity) collapses the cell cycle",
          f"Seed 1, dnaa-4 reference config, oriC-high DnaA-ATP saturation trigger (threshold 3 per "
          f"origin), cooperativity OFF (n=1). TOP: cell mass — the cooperative oriC-low trigger (n=4, "
          f"green) sawtooths and divides, while the naive oriC-high trigger (red) never initiates and "
          f"runs away to ~{mmax:.0f} fg. BOTTOM: the trigger signal (oriC-high DnaA-ATP bound) stays "
          f"below the threshold (3 × n_oriC) — independent-site binding under-saturates without a "
          f"cooperative filament. Regenerated in-sim (stitched-minutes axis).",
          "Wiring initiation onto full oriC-high DnaA-ATP saturation fails when the binding is "
          "independent-site: the third high-affinity box stochastically never binds, the threshold is "
          "never met, and the cell grows without dividing. This is why cooperativity (dnaa-5) is a "
          "prerequisite for the mechanistic trigger — the diagnostic behind finding F-02.",
          [os.path.basename(firstpass_dir), os.path.basename(ref_dir)],
          "scripts/render_dnaa6_figures.py")
    print(f"  wrote {base}.png/.svg (mass runaway to {mmax:.0f} fg)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="out/nk_n4_K30")
    ap.add_argument("--weak", default="out/nk_n1_K30")
    ap.add_argument("--firstpass", default="out/dnaa6_firstpass_collapse")
    ap.add_argument("--study-dir", default="workspace/studies/dnaa-6-mechanistic-initiation")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.study_dir, "charts"), exist_ok=True)
    once_per_cycle(args.ref, args.study_dir)
    cooperativity_contrast(args.weak, args.ref, args.study_dir)
    # collapse_figure(...) retired: the first-pass oriC-high collapse was a
    # 59e108fb-build artifact and does not reproduce on current code (gen 1 fires
    # at highATP 4 and divides even without cooperativity). Kept as a callable for
    # historical reference but no longer part of the dnaa-6 figure set.


if __name__ == "__main__":
    main()
