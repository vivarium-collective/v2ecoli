"""Plot plasmid copy-number trajectories across seeds from a multiseed run.

Reads ``out/plasmid/multiseed_timeseries.json`` (written by
``run_plasmid_multiseed.py``), plots every seed's copy-number trajectory
as a faint line, overlays the per-timepoint median, and highlights the
single seed whose full trajectory is closest to the median in L² norm.

    uv run python scripts/plot_plasmid_multiseed.py
    uv run python scripts/plot_plasmid_multiseed.py \\
        --input out/plasmid/multiseed_timeseries.json \\
        --output out/plasmid/multiseed_copy_number.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _common_time_grid(seeds_data: list[dict]) -> np.ndarray:
    """Return a time grid covering every seed, to the longest seed's final
    time. Seeds that divide earlier leave NaN after their last snapshot —
    nan-aware reductions (nanmedian, nanmean) handle the ragged tail.
    """
    longest = max(seeds_data, key=lambda s: len(s["snapshots"]))
    return np.array([snap["time"] for snap in longest["snapshots"]])


def _trajectory_matrix(seeds_data: list[dict], field: str,
                       n_timepoints: int) -> np.ndarray:
    """Build an (n_seeds, n_timepoints) array for ``field``. Seeds
    shorter than n_timepoints leave NaN in their tail."""
    mat = np.full((len(seeds_data), n_timepoints), np.nan)
    for i, s in enumerate(seeds_data):
        snaps = s["snapshots"][:n_timepoints]
        for j, snap in enumerate(snaps):
            mat[i, j] = float(snap.get(field, 0) or 0)
    return mat


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="out/plasmid/multiseed_timeseries.json")
    parser.add_argument("--output",
        default="out/plasmid/multiseed_copy_number.png")
    parser.add_argument("--field", default="n_full_plasmids",
        help="Trajectory field to plot (default n_full_plasmids). "
             "Any numeric snapshot field works — e.g. dnaG, cell_mass.")
    parser.add_argument("--ymax", type=float, default=None,
        help="Optional y-axis upper bound (default: auto).")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    seeds_data = data["seeds"]
    if len(seeds_data) == 0:
        raise SystemExit("No seeds in input.")

    t = _common_time_grid(seeds_data)
    n_t = len(t)
    traj = _trajectory_matrix(seeds_data, args.field, n_t)

    # Median is only meaningful where every seed is still alive;
    # truncate it at the shortest seed's division (individual seed
    # trajectories still extend to their own division).
    common_len = min(len(s["snapshots"]) for s in seeds_data)
    median = np.full(n_t, np.nan)
    median[:common_len] = np.nanmedian(traj[:, :common_len], axis=0)

    # L² distance computed only over the common-time region.
    diffs_common = (traj[:, :common_len] - median[None, :common_len]) ** 2
    deviations = np.sqrt(np.nanmean(diffs_common, axis=1))
    closest_idx = int(np.argmin(deviations))
    closest_seed = seeds_data[closest_idx]["seed"]

    # Each seed divides at a different time → take each seed's last
    # recorded value (last non-NaN), not a common column.
    finals = np.array([
        float(s["snapshots"][-1].get(args.field, 0) or 0)
        for s in seeds_data
    ])
    final_mean = float(np.mean(finals))
    final_median = float(np.median(finals))
    final_std = float(np.std(finals, ddof=0))
    final_min = float(np.min(finals))
    final_max = float(np.max(finals))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Faint lines for all seeds.
    for i, seed in enumerate(seeds_data):
        label = None
        color = "0.7"
        lw = 0.9
        alpha = 0.55
        zorder = 1
        if i == closest_idx:
            color = "#c0392b"
            lw = 2.2
            alpha = 1.0
            zorder = 3
            label = f"seed {seed['seed']} (closest to median)"
        ax.plot(t / 60.0, traj[i], color=color, lw=lw, alpha=alpha,
                zorder=zorder, label=label)

    # Median overlay.
    ax.plot(t / 60.0, median, color="black", lw=1.6, ls="--",
            zorder=2, label="per-timepoint median")

    ax.set_xlabel("Time (min)")
    field_label = {
        "n_full_plasmids": "Plasmid copy number",
        "n_plasmid_active_replisomes": "Active plasmid replisomes",
        "dnaG": "DnaG (primase) count",
        "cell_mass": "Cell mass (fg)",
        "dry_mass": "Dry mass (fg)",
    }.get(args.field, args.field)
    ax.set_ylabel(field_label)
    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)
    n_seeds = len(seeds_data)
    duration_min = (t[-1] - t[0]) / 60.0 if n_t > 1 else 0.0
    ax.set_title(
        f"Uncontrolled plasmid replication — {n_seeds} seeds, "
        f"{duration_min:.0f} min\n"
        f"Final {args.field}: median={final_median:.1f}, "
        f"mean={final_mean:.1f}±{final_std:.1f} "
        f"[{final_min:.0f}, {final_max:.0f}]"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    print(f"  field: {args.field}")
    print(f"  seeds: {n_seeds}, duration: {duration_min:.1f} min")
    print(f"  final values: median={final_median:.1f}, "
          f"mean={final_mean:.1f}±{final_std:.1f}, "
          f"range=[{final_min:.0f}, {final_max:.0f}]")
    print(f"  closest-to-median seed: {closest_seed} "
          f"(L² distance {deviations[closest_idx]:.2f})")


if __name__ == "__main__":
    main()
