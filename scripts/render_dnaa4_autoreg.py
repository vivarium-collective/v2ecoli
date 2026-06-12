#!/usr/bin/env python
"""dnaa-4 autoregulation: metric extractor + autoreg-vs-control charts.

Reads the box-binding total-DnaA pools + the dnaA-promoter occupancy from a run's
parquet history and reports the five acceptance metrics, then (optionally) renders
the study charts. Reuses the dnaa-3 reader conventions.

Usage:
  render_dnaa4_autoreg.py --autoreg <run_dir> --control <run_dir> [--charts <out_dir>]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import polars as pl

L = "listeners__replication_data__"
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
PROM = [L + "promoter_high_free", L + "promoter_high_bound_atp", L + "promoter_high_bound_adp"]
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}


def _pq(run_dir: str) -> list[str]:
    return sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))


def _frame(run_dir: str) -> pl.DataFrame:
    """Per-tick frame with total DnaA, ATP-fraction, oriC, promoter occupancy."""
    fs = _pq(run_dir)
    if not fs:
        raise FileNotFoundError(f"no parquet under {run_dir}")
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {k: ids.index(v) for k, v in BULK.items()}
    cols = list(dict.fromkeys(["generation", L + "number_of_oric"] + BOUND_ATP + BOUND_ADP + PROM))
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        [pl.col(c) for c in cols]
        + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
    ).collect().sort("generation")
    a = lambda c: np.asarray(df[c].to_list(), dtype=float)
    bound_atp = sum(a(c) for c in BOUND_ATP)
    bound_adp = sum(a(c) for c in BOUND_ADP)
    total = a("apo") + a("atp") + a("adp") + bound_atp + bound_adp
    prom_bound = a(PROM[1]) + a(PROM[2])
    return pl.DataFrame({
        "generation": df["generation"],
        "total_dnaa": total,
        "atp_fraction": (a("atp") + bound_atp) / np.maximum(total, 1.0),
        "n_oric": a(L + "number_of_oric"),
        "promoter_occ": prom_bound / np.maximum(a(PROM[0]) + prom_bound, 1.0),
    })


def metrics(run_dir: str, ss_gen: int = 3) -> dict:
    """The five acceptance metrics over steady-state generations (>= ss_gen)."""
    df = _frame(run_dir)
    ss = df.filter(pl.col("generation") >= ss_gen)
    total = ss["total_dnaa"].to_numpy()
    g = ss["generation"].to_numpy().astype(int)
    gmeans = [total[g == gg].mean() for gg in sorted(set(g))]
    oric = ss["n_oric"].to_numpy()
    atpfr = ss["atp_fraction"].to_numpy()
    occ = ss["promoter_occ"].to_numpy()
    return {
        "reinit_ticks": int((oric > 2).sum()),
        "oric_max": int(oric.max()),
        "dnaa_peak": float(total.max()),
        "dnaa_gmean_min": float(min(gmeans)),
        "dnaa_gmean_max": float(max(gmeans)),
        "atpfr_min": float(atpfr.min()),
        "atpfr_max": float(atpfr.max()),
        "promoter_min": float(occ.min()),
        "promoter_max": float(occ.max()),
    }


def _verdicts(m: dict) -> dict:
    return {
        "reinit-events-zero": m["reinit_ticks"] == 0,
        "dnaa-peak-under-800": m["dnaa_peak"] <= 800,
        "dnaa-pool-within-band": 300 <= m["dnaa_gmean_min"] and m["dnaa_gmean_max"] <= 800,
        "dnaa-atp-fraction-in-band": 0.2 <= m["atpfr_min"] and m["atpfr_max"] <= 0.5,
    }


def _print(tag: str, m: dict) -> None:
    print(f"\n{tag}:")
    print(f"  re-init ticks (oriC>2): {m['reinit_ticks']:5d}  | oriC max: {m['oric_max']}")
    print(f"  DnaA peak:            {m['dnaa_peak']:7.0f}  (<800)")
    print(f"  DnaA gen-mean range:  {m['dnaa_gmean_min']:.0f}-{m['dnaa_gmean_max']:.0f}  ([300,800])")
    print(f"  ATP-fraction:         {m['atpfr_min']:.3f}-{m['atpfr_max']:.3f}  ([0.2,0.5])")
    print(f"  promoter occ swing:   {m['promoter_min']:.2f}-{m['promoter_max']:.2f}")


def render_charts(autoreg_dir: str, control_dir: str, out_dir: str) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    ad, cd = _frame(autoreg_dir), _frame(control_dir)

    # 1) DnaA pool band: total DnaA per tick, both runs, [300,800] shaded
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhspan(300, 800, color="green", alpha=0.10, label="target band [300,800]")
    for df, lab, col in ((cd, "control (s=0)", "#94a3b8"), (ad, "autoreg (s=0.8)", "#1f77b4")):
        ss = df.filter(pl.col("generation") >= 3)
        ax.plot(range(len(ss)), ss["total_dnaa"].to_numpy(), lw=0.8, color=col, label=lab)
    ax.axhline(800, color="#dc2626", ls="--", lw=1)
    ax.set_xlabel("tick (steady-state gens)"); ax.set_ylabel("total DnaA (bulk+bound)")
    ax.set_title("dnaa-4: DnaA pool — autoregulation vs no-autoregulation control")
    ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_pool_band.{e}", dpi=140)
    plt.close(fig)

    # 2) Promoter occupancy swing + the transcription scaling (1 - 0.8 f)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ss = ad.filter(pl.col("generation") >= 3)
    occ = ss["promoter_occ"].to_numpy()
    ax.plot(range(len(occ)), occ, lw=0.8, color="#7c3aed", label="dnaA-promoter occupancy f")
    ax.plot(range(len(occ)), 1.0 - 0.8 * occ, lw=0.8, color="#16a34a",
            label="transcription factor (1 - 0.8·f)")
    ax.set_ylim(0, 1.05); ax.set_xlabel("tick (steady-state gens)")
    ax.set_ylabel("fraction"); ax.set_title("dnaa-4: promoter occupancy drives transcription repression")
    ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{out_dir}/dnaa4_promoter_swing.{e}", dpi=140)
    plt.close(fig)
    print(f"  charts -> {out_dir}/dnaa4_pool_band.png, dnaa4_promoter_swing.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoreg", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--charts", default=None, help="output dir for charts")
    args = ap.parse_args()
    ma, mc = metrics(args.autoreg), metrics(args.control)
    _print("CONTROL (autoreg=0)", mc)
    _print("AUTOREG (s=0.8)", ma)
    va, vc = _verdicts(ma), _verdicts(mc)
    print("\nAUTOREG acceptance:", {k: ("PASS" if v else "FAIL") for k, v in va.items()})
    if args.charts:
        render_charts(args.autoreg, args.control, args.charts)


if __name__ == "__main__":
    main()
