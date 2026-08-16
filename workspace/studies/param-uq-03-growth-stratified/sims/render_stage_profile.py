#!/usr/bin/env python
"""Render the REAL per-cell-cycle-stage (θ-binned) Sobol profile for
param-uq-03-growth-stratified from results_n24/sobol.json.

Line/area plot: total-order Sobol of each knob (+ inert decoy) vs cell-cycle
progress θ (birth→division), 10 bins. Shows whether knob sensitivity varies
across the cycle (the strategy-4 deliverable)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOBS = ["basal_elongation_rate", "kS", "chrom_basal_elongation_rate",
         "inert_decoy"]
LABEL = {"basal_elongation_rate": "basal_elongation_rate (translation)",
         "kS": "kS (ppGpp/charging saturation)",
         "chrom_basal_elongation_rate": "chrom basal_elongation_rate (replication)",
         "inert_decoy": "inert decoy (adversarial null)"}
COLOR = {"basal_elongation_rate": "#2E5EAA", "kS": "#E07B39",
         "chrom_basal_elongation_rate": "#3F9B57", "inert_decoy": "#9AA0A6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--gen2-cover", type=float, default=0.6,
                    help="θ up to which gen-2 contributes (coverage boundary)")
    args = ap.parse_args()
    s = json.load(open(os.path.join(args.results, "sobol.json")))
    s4 = s["strategy_4_theta_binned"]
    n_bins = s.get("meta", {}).get("n_bins", 10)

    centers, series, errs = [], {k: [] for k in KNOBS}, []
    for b in range(n_bins):
        e = s4.get(f"theta_bin_{b}")
        if not e or "total_order" not in e:
            continue
        centers.append((b + 0.5) / n_bins)
        for k in KNOBS:
            series[k].append(e["total_order"][k])
        errs.append(e.get("rel_test_error", np.nan))
    centers = np.array(centers)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    # gen-2 coverage shading
    ax.axvspan(0, args.gen2_cover, color="#000000", alpha=0.035, zorder=0)
    ax.text(args.gen2_cover / 2, 0.96, "gen-1 + gen-2 pooled", ha="center",
            fontsize=8, color="#555")
    ax.text((args.gen2_cover + 1) / 2, 0.96, "gen-1 only", ha="center",
            fontsize=8, color="#555")
    ax.axvline(args.gen2_cover, color="#888", ls="--", lw=1, alpha=0.6)

    for k in KNOBS:
        lw = 2.6 if k != "inert_decoy" else 1.6
        ls = "-" if k != "inert_decoy" else ":"
        ax.plot(centers, series[k], marker="o", ms=5, lw=lw, ls=ls,
                color=COLOR[k], label=LABEL[k], zorder=3)
    ax.axhline(0.05, color="#B00020", ls=":", lw=1, alpha=0.7)
    ax.text(0.01, 0.065, "adversarial null band (0.05)", fontsize=7.5,
            color="#B00020")

    ax.set_xlabel("cell-cycle progress θ  (0 = birth → 1 = division)")
    ax.set_ylabel("total-order Sobol index (growth rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"REAL — per-stage Sobol vs cell-cycle progress θ  ·  {args.label}\n"
                 "basal_elongation_rate leads early cycle; kS takes over mid-late "
                 "(sensitivity is stage-dependent)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
