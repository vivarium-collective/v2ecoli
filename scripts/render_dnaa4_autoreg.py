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

L = "listeners__replication_data__"
M = "listeners__mass__"
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
PROM = [L + "promoter_high_free", L + "promoter_high_bound_atp", L + "promoter_high_bound_adp"]
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}
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
    bound_atp = sum(a(c) for c in BOUND_ATP)
    bound_adp = sum(a(c) for c in BOUND_ADP)
    total = a("apo") + a("atp") + a("adp") + bound_atp + bound_adp
    gen = a("generation").astype(int)
    gtime = a("global_time")
    # stitch global_time (resets each gen) into cumulative minutes
    abs_min = np.zeros_like(gtime)
    offset = 0.0
    for gg in sorted(set(gen)):
        m = gen == gg
        abs_min[m] = (offset + gtime[m]) / 60.0
        offset += gtime[m].max()
    cell_mass = a(M + "cell_mass")
    volume = cell_mass / CELL_DENSITY
    prom_bound = a(PROM[1]) + a(PROM[2])
    return pl.DataFrame({
        "generation": gen,
        "abs_min": abs_min,
        "total_dnaa": total,
        "atp_fraction": (a("atp") + bound_atp) / np.maximum(total, 1.0),
        "adp_fraction": (a("adp") + bound_adp) / np.maximum(total, 1.0),
        "n_oric": a(L + "number_of_oric"),
        "cell_mass": cell_mass,
        "dnaa_conc_nM": total / np.maximum(volume, 1e-9) * NM,
        "promoter_occ": prom_bound / np.maximum(a(PROM[0]) + prom_bound, 1.0),
    })


def _gen_boundaries(df: pl.DataFrame):
    """abs_min values where the generation increments (lineage boundaries)."""
    gen = df["generation"].to_numpy()
    t = df["abs_min"].to_numpy()
    return [t[i] for i in range(1, len(gen)) if gen[i] != gen[i - 1]]


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
        for b in _gen_boundaries(df):
            ax.axvline(b, color="#cbd5e1", lw=0.6, ls=":", zorder=0)

    # 1) DnaA pool band — minutes x-axis + lineage boundaries
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhspan(300, 800, color="green", alpha=0.10, label="target band [300,800]")
    for df, lab, col in ((cd, "control (s=0)", "#94a3b8"), (ad, "autoreg (Hill)", "#1f77b4")):
        sub = df.filter(pl.col("generation") >= 3)
        ax.plot(sub["abs_min"].to_numpy(), sub["total_dnaa"].to_numpy(), lw=0.8, color=col, label=lab)
    _gridlines(ax, ad.filter(pl.col("generation") >= 3))
    ax.axhline(800, color="#dc2626", ls="--", lw=1)
    ax.set_xlabel("simulation time (min)"); ax.set_ylabel("total DnaA (bulk+bound)")
    ax.set_title("dnaa-4: DnaA pool — autoregulation vs control (dotted = lineage boundaries)")
    ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_pool_band.{e}", dpi=140)
    plt.close(fig)

    # 2) promoter occupancy swing + transcription factor — minutes + boundaries
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sub = ad.filter(pl.col("generation") >= 3)
    t = sub["abs_min"].to_numpy(); occ = sub["promoter_occ"].to_numpy()
    ax.plot(t, occ, lw=0.8, color="#7c3aed", label="dnaA-promoter occupancy f")
    ax.plot(t, 1.0 - 0.8 * occ, lw=0.8, color="#16a34a", label="transcription factor (Hill)")
    _gridlines(ax, sub)
    ax.set_ylim(0, 1.05); ax.set_xlabel("simulation time (min)"); ax.set_ylabel("fraction")
    ax.set_title("dnaa-4: promoter occupancy drives transcription repression")
    ax.legend(fontsize=8); fig.tight_layout()
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
        for b in bnd:
            ax.axvline(b, color="#cbd5e1", lw=0.6, ls=":", zorder=0)
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
