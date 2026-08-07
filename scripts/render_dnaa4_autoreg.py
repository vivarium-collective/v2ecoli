#!/usr/bin/env python
"""dnaa-4 autoregulation: metric extractor + autoreg-vs-control charts.

Reads the box-binding total-DnaA pools, dnaA-promoter occupancy, oriC, cell mass,
and the ATP/ADP fractions from a run's parquet history. Reports the five acceptance
metrics and renders the study charts — with a MINUTES x-axis and lineage (generation)
boundaries, plus a lineage-stability panel (oriC, cell mass, DnaA concentration,
DnaA-ATP/ADP fractions). Per Rashmi's 2026-06-12 feedback.

Usage:
  render_dnaa4_autoreg.py --autoreg <run_dir> --control <run_dir> [--charts <out_dir>]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import polars as pl

# canonical DnaA pool decomposition + ATP/ADP fraction — the SINGLE source of truth.
# These constants and the (free+bound)/total fraction live in one module so dnaa-3/4/7
# render scripts can never diverge on how they define the fraction (see dnaa_observables).
import dnaa_observables as dnaa
import pbg_plot_style as ps  # shared house style: minutes stitch, lineage lines, no gridlines

L = dnaa.L
M = "listeners__mass__"
BOUND_ATP = dnaa.BOUND_ATP_COLS
BOUND_ADP = dnaa.BOUND_ADP_COLS
PROM = [L + "promoter_high_free", L + "promoter_high_bound_atp", L + "promoter_high_bound_adp"]
BULK = {"apo": dnaa.APO_ID, "atp": dnaa.ATP_ID, "adp": dnaa.ADP_ID}
CELL_DENSITY = 1.1  # fg/fL — volume_fL = cell_mass_fg / density
NM = 1e9 / 6.022e23 / 1e-15  # molecules/fL -> nM (~1.661)


def _frame(run_dir: str) -> pl.DataFrame:
    fs = sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))
    if not fs:
        raise FileNotFoundError(f"no parquet under {run_dir}")
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {k: ids.index(v) for k, v in BULK.items()}
    cols = list(dict.fromkeys(
        ["generation", "global_time", L + "number_of_oric", M + "cell_mass"]
        + BOUND_ATP + BOUND_ADP + PROM))
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        [pl.col(c) for c in cols]
        + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
    ).collect().sort(["generation", "global_time"])
    a = lambda c: np.asarray(df[c].to_list(), dtype=float)
    # canonical decomposition (free + bound) — single source of truth
    dec = dnaa.decompose(a("apo"), a("atp"), a("adp"),
                         [a(c) for c in BOUND_ATP], [a(c) for c in BOUND_ADP])
    total = dec["total_dnaa"]
    gen = a("generation").astype(int)
    gtime = a("global_time")
    # stitch global_time (resets each gen) into cumulative minutes (shared house style)
    abs_min = ps.stitch_minutes(gen, gtime)
    cell_mass = a(M + "cell_mass")
    volume = cell_mass / CELL_DENSITY
    prom_bound = a(PROM[1]) + a(PROM[2])
    return pl.DataFrame({
        "generation": gen,
        "abs_min": abs_min,
        "total_dnaa": total,
        "atp_fraction": dec["atp_fraction"],
        "adp_fraction": dec["adp_fraction"],
        "n_oric": a(L + "number_of_oric"),
        "cell_mass": cell_mass,
        "dnaa_conc_nM": total / np.maximum(volume, 1e-9) * NM,
        "promoter_occ": prom_bound / np.maximum(a(PROM[0]) + prom_bound, 1.0),
    })


def _gen_boundaries(df: pl.DataFrame):
    """abs_min values where the generation increments (lineage boundaries).
    Delegates to the shared house style (pbg_plot_style.gen_boundaries)."""
    return ps.gen_boundaries(df["generation"].to_numpy(), df["abs_min"].to_numpy())


def metrics(run_dir: str, ss_gen: int = 3) -> dict:
    df = _frame(run_dir)
    ss = df.filter(pl.col("generation") >= ss_gen)
    total = ss["total_dnaa"].to_numpy()
    g = ss["generation"].to_numpy().astype(int)
    gmeans = [total[g == gg].mean() for gg in sorted(set(g))]
    oric = ss["n_oric"].to_numpy()
    atpfr = ss["atp_fraction"].to_numpy()
    return {
        "reinit_ticks": int((oric > 2).sum()), "oric_max": int(oric.max()),
        "dnaa_peak": float(total.max()),
        "dnaa_gmean_min": float(min(gmeans)), "dnaa_gmean_max": float(max(gmeans)),
        "atpfr_min": float(atpfr.min()), "atpfr_max": float(atpfr.max()),
    }


def _print(tag: str, m: dict) -> None:
    print(f"\n{tag}:")
    print(f"  re-init ticks (oriC>2): {m['reinit_ticks']:5d}  | oriC max: {m['oric_max']}")
    print(f"  DnaA peak:            {m['dnaa_peak']:7.0f}  (<800)")
    print(f"  DnaA gen-mean range:  {m['dnaa_gmean_min']:.0f}-{m['dnaa_gmean_max']:.0f}  ([300,800])")
    print(f"  ATP-fraction:         {m['atpfr_min']:.3f}-{m['atpfr_max']:.3f}  ([0.2,0.5])")


def render_charts(autoreg_dir: str, control_dir: str, out_dir: str) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    ad, cd = _frame(autoreg_dir), _frame(control_dir)

    def _gridlines(ax, df):
        ps.mark_lineages(ax, _gen_boundaries(df))

    # 1) DnaA pool band — minutes x-axis + lineage boundaries
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhspan(300, 800, color="green", alpha=0.10, label="target band [300,800]")
    for df, lab, col in ((cd, "control (s=0)", "#94a3b8"), (ad, "autoreg (Hill)", "#1f77b4")):
        sub = df.filter(pl.col("generation") >= 3)
        ax.plot(sub["abs_min"].to_numpy(), sub["total_dnaa"].to_numpy(), lw=0.8, color=col, label=lab)
    _gridlines(ax, ad.filter(pl.col("generation") >= 3))
    ax.axhline(800, color="#dc2626", ls="--", lw=1)
    ps.style_axes(ax, "simulation time (min)", "total DnaA (bulk+bound, molecule count)")
    ax.set_title("dnaa-4: DnaA pool — autoregulation vs control (dotted = lineage boundaries)")
    ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_pool_band.{e}", dpi=140)
    plt.close(fig)

    # 2) The feedback loop, made legible — a 2-panel ZOOM to a few representative
    #    generations (the old all-16-gen overlay filled into solid blocks). Top:
    #    DnaA pool. Bottom: dnaA-promoter occupancy f. Aligned in time so the loop
    #    reads directly: DnaA up -> promoter fills -> transcription repressed -> pool capped.
    gens_all = sorted(set(int(g) for g in ad["generation"].to_list()))
    win = [g for g in gens_all if g >= 6][:4] or [g for g in gens_all if g >= 3][:4]
    sub = ad.filter(pl.col("generation").is_in(win))
    t = sub["abs_min"].to_numpy(); bnd = _gen_boundaries(sub)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.axhspan(300, 800, color="green", alpha=0.10, label="target band [300,800]")
    ax1.plot(t, sub["total_dnaa"].to_numpy(), lw=1.2, color="#1f77b4")
    ax1.set_ylabel("total DnaA\n(bulk+bound)"); ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("dnaa-4: the autoregulation feedback loop (" + str(len(win)) + " representative generations)\n"
                  "DnaA accumulates (top) → dnaA-promoter fills (bottom, f) → transcription repressed → DnaA pool capped",
                  fontsize=10)
    ax2.plot(t, sub["promoter_occ"].to_numpy(), lw=1.2, color="#7c3aed")
    ax2.axhline(0.5, color="#9ca3af", ls=":", lw=0.8)
    ax2.text(t[0], 0.53, "Hill K=0.5 (repression switch)", fontsize=7, color="#6b7280")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("dnaA-promoter\noccupancy f")
    ax2.set_xlabel("simulation time (min)  ·  dotted = cell division (lineage boundary)")
    for ax in (ax1, ax2):
        ps.mark_lineages(ax, bnd)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_promoter_swing.{e}", dpi=140)
    plt.close(fig)

    # 3) lineage-stability panel (autoreg run): oriC, cell mass, DnaA conc, ATP/ADP fraction
    sub = ad.filter(pl.col("generation") >= 3)
    t = sub["abs_min"].to_numpy(); bnd = _gen_boundaries(sub)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    panels = [
        ("number_of_oriC", sub["n_oric"].to_numpy(), "#dc2626", None),
        ("cell mass (fg)", sub["cell_mass"].to_numpy(), "#0891b2", None),
        ("DnaA concentration (nM)", sub["dnaa_conc_nM"].to_numpy(), "#7c3aed", None),
        ("DnaA-ATP / DnaA-ADP fraction", None, None, None),
    ]
    for ax, (lab, y, col, _) in zip(axes, panels):
        ps.mark_lineages(ax, bnd)
        if lab.startswith("DnaA-ATP"):
            ax.plot(t, sub["atp_fraction"].to_numpy(), lw=0.8, color="#16a34a", label="DnaA-ATP frac")
            ax.plot(t, sub["adp_fraction"].to_numpy(), lw=0.8, color="#f59e0b", label="DnaA-ADP frac")
            ax.axhspan(0.2, 0.5, color="green", alpha=0.08)
            ax.set_ylabel("fraction"); ax.legend(fontsize=7, loc="upper right")
        else:
            ax.plot(t, y, lw=0.8, color=col); ax.set_ylabel(lab, fontsize=8)
    axes[-1].set_xlabel("simulation time (min)")
    axes[0].set_title("dnaa-4: lineage stability under autoregulation (dotted = lineage boundaries)")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_lineage_stability.{e}", dpi=140)
    plt.close(fig)
    print(f"  charts -> {out_dir}/dnaa4_pool_band.png, dnaa4_promoter_swing.png, dnaa4_lineage_stability.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoreg", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--charts", default=None)
    args = ap.parse_args()
    mc, ma = metrics(args.control), metrics(args.autoreg)
    _print("CONTROL (autoreg=0)", mc)
    _print("AUTOREG (Hill)", ma)
    if args.charts:
        render_charts(args.autoreg, args.control, args.charts)


if __name__ == "__main__":
    main()
