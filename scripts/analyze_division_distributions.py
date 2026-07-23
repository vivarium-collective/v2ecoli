#!/usr/bin/env python
"""Distributions of time-to-division and size-at-division under the mechanistic
(sat-init) replication initiation.

Reads per-generation summaries (``<exp>_summary.json``) — each divided
generation contributes one (duration_min, final_dry_mass_fg) sample — pools
them across the given experiments, and plots two histograms plus a size-vs-time
scatter. Multiple --summary may be given (e.g. many seeds) to build a real
distribution. Optionally --cell-mass reads the true cell mass at the last tick
of each generation from the parquet history instead of the summary's dry mass.

Usage:
    python scripts/analyze_division_distributions.py \
        --summary out/dnaa5_succ_valid_seed4_parquet/dnaa5_succ_valid_seed4_summary.json \
        --out out/analysis/division_succinate.svg --title "succinate"
"""
import argparse, glob, json, os
import numpy as np


def load_samples(summary_paths):
    tau, size = [], []
    for p in summary_paths:
        try:
            s = json.load(open(p))
        except Exception as e:
            print(f"  warn: {p}: {e}")
            continue
        for g in (s.get("gens") or []):
            if g.get("divided"):
                if g.get("duration_min") is not None:
                    tau.append(float(g["duration_min"]))
                if g.get("final_dry_mass_fg") is not None:
                    size.append(float(g["final_dry_mass_fg"]))
    return np.array(tau), np.array(size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="append", required=True,
                    help="one or more <exp>_summary.json (glob ok)")
    ap.add_argument("--out", default="out/analysis/division.svg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    paths = []
    for s in args.summary:
        paths += sorted(glob.glob(s)) or [s]
    tau, size = load_samples(paths)
    print(f"{args.title or 'division'}: {len(tau)} divided generations "
          + (f"| tau median {np.median(tau):.1f} min, size median {np.median(size):.0f} fg"
             if len(tau) else "| none (no divisions yet)"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cond = args.title

    def hist(ax, data, xlabel, color, unit):
        if len(data):
            ax.hist(data, bins=max(6, min(20, len(data))), color=color,
                    edgecolor="#334155", alpha=0.85)
            m = np.median(data)
            ax.axvline(m, color="#dc2626", ls="--", lw=1.6,
                       label=f"median = {m:.1f} {unit}  (n={len(data)})")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "no divisions yet", ha="center", va="center",
                    transform=ax.transAxes, color="#6b7280")
        ax.set_xlabel(xlabel); ax.set_ylabel("count")

    hist(axes[0], tau, "time to division (min)", "#16a34a", "min")
    axes[0].set_title("Time-to-division distribution")
    hist(axes[1], size, "dry mass at division (fg)", "#7c3aed", "fg")
    axes[1].set_title("Size-at-division distribution")
    if len(tau) and len(size):
        n = min(len(tau), len(size))
        axes[2].scatter(tau[:n], size[:n], s=28, color="#2563eb",
                        edgecolors="#1e3a8a", alpha=0.8)
    axes[2].set_xlabel("time to division (min)")
    axes[2].set_ylabel("dry mass at division (fg)")
    axes[2].set_title("Size vs. time at division")
    fig.suptitle(f"Division dynamics — {cond}" if cond else "Division dynamics",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    meta = {"title": f"Division dynamics ({cond})" if cond else "Division dynamics",
            "caption": "Time-to-division and dry-mass-at-division distributions "
                       "under the mechanistic sat-init replication initiation, "
                       f"pooled over {len(tau)} divided generations.",
            "n_divisions": int(len(tau)),
            "tau_median_min": (float(np.median(tau)) if len(tau) else None),
            "size_median_fg": (float(np.median(size)) if len(size) else None)}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
