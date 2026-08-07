#!/usr/bin/env python
"""Distributions of time-to-division and size-at-division under the mechanistic
(sat-init) replication initiation, parsed from the multigen runner logs.

Each generation that divides emits one line

    DIVISION at t=<sec>s (dry_mass=<fg> fg, threshold=... fg, chromosomes=2)

which is one (time-to-division = t/60 min, size-at-division = dry_mass fg)
sample. We pool them across the given logs (many seeds → a real distribution)
and plot two histograms plus a size-vs-time scatter. Pass --label to title the
condition; pass several --log globs to pool seeds.

Usage:
    python scripts/analyze_division_distributions.py \
        --log 'out/dnaa5_succ_mech_seed*.log' \
        --out out/analysis/division_succinate.svg --title "succinate (sat-init)"
"""
import argparse, glob, json, os, re
import numpy as np

_DIV = re.compile(r"DIVISION at t=([0-9.]+)s \(dry_mass=([0-9.]+) fg")


def load_samples(log_paths):
    tau, size = [], []
    for p in log_paths:
        try:
            txt = open(p).read()
        except Exception as e:
            print(f"  warn: {p}: {e}")
            continue
        for m in _DIV.finditer(txt):
            tau.append(float(m.group(1)) / 60.0)
            size.append(float(m.group(2)))
    return np.array(tau), np.array(size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True,
                    help="one or more run-log globs (pool seeds)")
    ap.add_argument("--out", default="out/analysis/division.svg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    paths = []
    for s in args.log:
        paths += sorted(glob.glob(s)) or [s]
    tau, size = load_samples(paths)
    print(f"{args.title or 'division'}: {len(tau)} division events "
          + (f"| tau median {np.median(tau):.1f} min, size median {np.median(size):.0f} fg"
             if len(tau) else "| none (no divisions logged yet)"))

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
                       f"pooled over {len(tau)} division events.",
            "n_divisions": int(len(tau)),
            "tau_median_min": (float(np.median(tau)) if len(tau) else None),
            "tau_cv": (float(np.std(tau) / np.mean(tau)) if len(tau) else None),
            "size_median_fg": (float(np.median(size)) if len(size) else None),
            "size_cv": (float(np.std(size) / np.mean(size)) if len(size) else None)}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
