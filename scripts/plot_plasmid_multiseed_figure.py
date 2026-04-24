"""Four-panel figure for the v2ecoli uncontrolled-plasmid multiseed run.

Reads ``out/plasmid/multiseed_timeseries.json`` and draws:
    1. Plasmid copy number vs. time
    2. Chromosome count vs. time
    3. Cell mass vs. time (with critical-initiation-mass threshold line)
    4. Replisome subunit bulk counts vs. time (DnaG, dnaB, pol_core,
       beta_clamp, holA)

Panels 1-3 use the same visual grammar as ``plot_plasmid_multiseed.py``:
faint gray lines per seed, black dashed per-timepoint median, and a
single highlighted seed (closest to plasmid-copy-number median) in red.
The same highlighted seed is used across all panels.

Panel 4 overlays all five subunits for the highlighted seed; DnaG is
emphasized to make the bottleneck visible against the ample pool of the
others.

    uv run python scripts/plot_plasmid_multiseed_figure.py
    uv run python scripts/plot_plasmid_multiseed_figure.py \\
        --input out/plasmid/multiseed_timeseries.json \\
        --output out/plasmid/multiseed_figure.png \\
        --critical-initiation-mass 975.0
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


SUBUNIT_FIELDS = ["dnaG", "dnaB", "holA", "EG11500", "pol_core", "beta_clamp"]
SUBUNIT_COLORS = {
    "dnaG": "#c0392b",
    "dnaB": "#2980b9",
    "holA": "#d35400",
    "EG11500": "#7f8c8d",
    "pol_core": "#16a085",
    "beta_clamp": "#8e44ad",
}
SUBUNIT_LABELS = {
    "dnaG": "DnaG (primase, monomer)",
    "dnaB": "DnaB (helicase, monomer)",
    "holA": "HolA (δ, monomer)",
    "EG11500": "DnaX (τ/γ, monomer)",
    "pol_core": "Pol III core (trimer)",
    "beta_clamp": "β-clamp (trimer)",
}


def _common_time_grid(seeds_data):
    longest = max(seeds_data, key=lambda s: len(s["snapshots"]))
    return np.array([snap["time"] for snap in longest["snapshots"]])


def _trajectory_matrix(seeds_data, field, n_timepoints):
    """Build an (n_seeds, n_timepoints) array for ``field``. For subunit
    fields the current run_plasmid_multiseed.py emits ``bulk_<name>`` but
    older JSONs use the bare name — try both.
    """
    mat = np.full((len(seeds_data), n_timepoints), np.nan)
    alt = f"bulk_{field}" if not field.startswith("bulk_") else None
    for i, s in enumerate(seeds_data):
        snaps = s["snapshots"][:n_timepoints]
        for j, snap in enumerate(snaps):
            v = snap.get(field)
            if v is None and alt is not None:
                v = snap.get(alt)
            mat[i, j] = float(v or 0)
    return mat


def _field_present(seeds_data, field):
    snap = seeds_data[0]["snapshots"][0]
    return field in snap or f"bulk_{field}" in snap


def _closest_to_median_seed(traj, common_len):
    common_median = np.nanmedian(traj[:, :common_len], axis=0)
    diffs = (traj[:, :common_len] - common_median[None, :]) ** 2
    deviations = np.sqrt(np.nanmean(diffs, axis=1))
    return int(np.argmin(deviations)), deviations


def _plot_seeded_panel(ax, t_min, traj, highlight_idx, highlight_seed_id,
                       median, ylabel, title):
    for i in range(traj.shape[0]):
        if i == highlight_idx:
            continue
        ax.plot(t_min, traj[i], color="0.7", lw=0.9, alpha=0.55, zorder=1)
    ax.plot(t_min, median, color="black", lw=1.6, ls="--", zorder=2,
            label="per-timepoint median")
    ax.plot(t_min, traj[highlight_idx], color="#c0392b", lw=2.2, alpha=1.0,
            zorder=3, label=f"seed {highlight_seed_id}")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
        default="out/plasmid/multiseed_timeseries.json")
    parser.add_argument("--output",
        default="out/plasmid/multiseed_figure.png")
    parser.add_argument("--critical-initiation-mass", type=float, default=975.0,
        help="Critical initiation mass per oriC in fg (default 975.0, "
             "matches chromosome_replication.py default).")
    parser.add_argument("--n-oric", type=int, default=2,
        help="Number of oriCs used to compute the re-initiation threshold "
             "line on the cell-mass panel (default 2: two oriCs persist "
             "after the initial round terminates under uncontrolled "
             "plasmid load).")
    parser.add_argument("--subunit-smooth", type=int, default=1501,
        help="Rolling-median window (in snapshots) applied to the subunit "
             "panel only. After the inherited chromosome round "
             "terminates, DnaG settles into a persistent 0<->8 "
             "oscillation; a large window collapses that envelope to its "
             "median so the trajectory reads as a thin line. Set 0 to "
             "disable.")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    seeds_data = data["seeds"]
    if not seeds_data:
        raise SystemExit("No seeds in input.")

    t = _common_time_grid(seeds_data)
    n_t = len(t)
    t_min = t / 60.0
    common_len = min(len(s["snapshots"]) for s in seeds_data)

    # Build trajectory matrices for every field we plot. Drop subunits
    # that aren't in the JSON so older runs still produce a figure.
    available_subunits = [f for f in SUBUNIT_FIELDS
                          if _field_present(seeds_data, f)]
    missing = [f for f in SUBUNIT_FIELDS if f not in available_subunits]
    if missing:
        print(f"WARNING: subunit fields missing from input JSON: {missing}. "
              f"Re-run scripts/run_plasmid_multiseed.py to include them.")
    fields = (["n_full_plasmids", "n_full_chromosomes",
               "n_active_replisomes", "cell_mass"]
              + available_subunits)
    traj = {f: _trajectory_matrix(seeds_data, f, n_t) for f in fields}

    # Highlight the seed whose plasmid copy number is closest to the median.
    highlight_idx, _ = _closest_to_median_seed(traj["n_full_plasmids"],
                                                common_len)
    highlight_seed_id = seeds_data[highlight_idx]["seed"]

    # Per-timepoint medians (truncated at common_len, NaN beyond).
    def _median(field):
        m = np.full(n_t, np.nan)
        m[:common_len] = np.nanmedian(traj[field][:, :common_len], axis=0)
        return m

    med_plasmid = _median("n_full_plasmids")
    med_chrom = _median("n_full_chromosomes")
    med_mass = _median("cell_mass")

    # Layout: left column is the "chromosome story" (chromosome count on
    # top, cell mass directly below so the viewer can match the
    # termination event in panel 1 against the cell-mass threshold in
    # panel 3). Right column is the "plasmid + resource" story (copy
    # number on top, subunit availability below).
    # Layout: left column stacks plasmid copy (top) above subunits
    # (bottom) — the plasmid + resource story. Right column stacks
    # three smaller panels: chromosome count, active replisomes, and
    # cell mass — the chromosome story. Splitting chromosome vs.
    # replisome into separate small panels makes the "termination +
    # no re-initiation" pattern read at a glance without squinting at
    # a twin-axis plot.
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.25)
    ax_copy = fig.add_subplot(gs[0:3, 0])
    ax_sub = fig.add_subplot(gs[3:6, 0], sharex=ax_copy)
    ax_chrom = fig.add_subplot(gs[0:2, 1], sharex=ax_copy)
    ax_repl = fig.add_subplot(gs[2:4, 1], sharex=ax_copy)
    ax_mass = fig.add_subplot(gs[4:6, 1], sharex=ax_copy)

    # Panel (top-left): plasmid copy number
    _plot_seeded_panel(
        ax_copy, t_min, traj["n_full_plasmids"], highlight_idx,
        highlight_seed_id, med_plasmid,
        "Plasmid copy number",
        "Plasmid copy number (runaway under uncontrolled replication)")
    ax_copy.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Panel (top-right): chromosome count — highlighted seed only
    ax_chrom.plot(t_min, traj["n_full_chromosomes"][highlight_idx],
                  color="#c0392b", lw=1.6)
    ax_chrom.set_ylabel("Full chromosomes")
    ax_chrom.set_ylim(-0.2, 2.6)
    ax_chrom.grid(False)
    ax_chrom.set_title(f"Chromosome count (seed {highlight_seed_id})",
                       fontsize=10)

    # Panel (middle-right): active replisomes — highlighted seed only
    ax_repl.plot(t_min, traj["n_active_replisomes"][highlight_idx],
                 color="#2980b9", lw=1.6)
    ax_repl.set_ylabel("Active replisomes")
    ax_repl.set_ylim(-0.2, 2.6)
    ax_repl.grid(False)
    ax_repl.set_title(
        "Active replisomes (inherited round terminates; "
        "no re-initiation)", fontsize=10)

    # Panel (bottom-right): cell mass with re-initiation threshold
    _plot_seeded_panel(
        ax_mass, t_min, traj["cell_mass"], highlight_idx,
        highlight_seed_id, med_mass,
        "Cell mass (fg)",
        "Cell mass vs. re-initiation threshold")
    threshold = args.n_oric * args.critical_initiation_mass
    ax_mass.axhline(threshold, color="#2c3e50",
                    ls=":", lw=1.5, zorder=4,
                    label=f"re-initiation threshold "
                          f"({args.n_oric}×{args.critical_initiation_mass:.0f} "
                          f"= {threshold:.0f} fg)")
    ax_mass.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_mass.set_xlabel("Time (min)")

    # Panel (bottom-left): replisome subunit counts for the highlighted
    # seed — one thin line per subunit. Labels mark monomers vs. trimers
    # so the bottleneck is interpretable without extra visual encoding.
    # DnaG in red flags it as the depleted pool. Rolling median smooths
    # out high-frequency 0<->N flicker that would otherwise render as a
    # solid block on the log axis.
    def _rolling_median(y, w):
        # Smooth only the valid (non-NaN) prefix; leave NaN in the tail
        # so seeds that divided early don't get artificially extended.
        # The NaN padding in traj comes from ragged per-seed durations,
        # and np.median over any window containing NaN returns NaN — so
        # a naive rolling median clips each seed half-a-window before
        # its actual end.
        y = np.asarray(y, dtype=float)
        valid_mask = ~np.isnan(y)
        if not valid_mask.any():
            return y.copy()
        last = int(np.where(valid_mask)[0][-1]) + 1  # exclusive end
        if w <= 1:
            return y.copy()
        w_eff = min(w, last)
        if w_eff % 2 == 0:
            w_eff += 1
        half = w_eff // 2
        y_valid = y[:last]
        padded = np.pad(y_valid, (half, half), mode="edge")
        smoothed = np.empty_like(y_valid)
        for k in range(last):
            smoothed[k] = np.median(padded[k:k + w_eff])
        out = np.full_like(y, np.nan)
        out[:last] = smoothed
        return out

    # Faint gray background: all non-highlighted seeds per subunit,
    # smoothed the same way as the highlighted trajectory.
    for field in available_subunits:
        for i in range(traj[field].shape[0]):
            if i == highlight_idx:
                continue
            y_bg = _rolling_median(traj[field][i], args.subunit_smooth)
            ax_sub.plot(t_min, y_bg, color="0.75", lw=0.5,
                        alpha=0.35, zorder=1, solid_capstyle="butt")
    for field in available_subunits:
        y = traj[field][highlight_idx]
        y_smooth = _rolling_median(y, args.subunit_smooth)
        ax_sub.plot(t_min, y_smooth,
                    color=SUBUNIT_COLORS[field], lw=0.6, alpha=0.95,
                    zorder=3, solid_capstyle="butt",
                    label=SUBUNIT_LABELS[field])
    ax_sub.set_ylabel("Bulk count")
    ax_sub.set_xlabel("Time (min)")
    ax_sub.set_title(
        f"Replisome subunit bulk counts (seed {highlight_seed_id}) — "
        f"DnaG is the bottleneck",
        fontsize=10)
    # Linear y-axis (not log) so sub-1 smoothed values for DnaG don't
    # look like "going negative." DnaG's near-zero depletion reads as
    # a flat line along the x-axis; the other subunits (20-600 range)
    # still fit comfortably on the same linear scale.
    # Legend in the upper-left corner of the panel.
    ax_sub.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_sub.grid(False)

    n_seeds = len(seeds_data)
    duration_min = (t[-1] - t[0]) / 60.0 if n_t > 1 else 0.0
    # Hide x-tick labels on panels that aren't at the bottom of their
    # column — ax_sub (left) and ax_mass (right) carry the x-axis labels.
    for ax in (ax_copy, ax_chrom, ax_repl):
        plt.setp(ax.get_xticklabels(), visible=False)

    # Subpanel labels (A–E) in reading order: top-left, top-right,
    # middle-right, bottom-right, bottom-left.
    for ax, label in [
        (ax_copy, "A"), (ax_chrom, "B"), (ax_repl, "C"),
        (ax_mass, "D"), (ax_sub, "E"),
    ]:
        ax.text(-0.08, 1.02, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        f"v2ecoli uncontrolled plasmid replication — {n_seeds} seeds, "
        f"{duration_min:.0f} min (highlighted: seed {highlight_seed_id}, "
        f"closest to plasmid-copy-number median)",
        fontsize=11)
    gs.tight_layout(fig, rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    print(f"  highlighted seed: {highlight_seed_id}")
    print(f"  seeds: {n_seeds}, duration: {duration_min:.1f} min")
    print(f"  critical initiation mass: {args.critical_initiation_mass} fg / oriC")
    print(f"  re-initiation threshold drawn: "
          f"{args.n_oric} oriCs × {args.critical_initiation_mass} = "
          f"{args.n_oric * args.critical_initiation_mass} fg")
    finals_plasmid = np.array([
        float(s["snapshots"][-1].get("n_full_plasmids", 0) or 0)
        for s in seeds_data])
    print(f"  final plasmid copy number: "
          f"median={np.median(finals_plasmid):.0f}, "
          f"range=[{finals_plasmid.min():.0f}, {finals_plasmid.max():.0f}]")


if __name__ == "__main__":
    main()
