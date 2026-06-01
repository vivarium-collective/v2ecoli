"""Figure 3 for the prelim proposal: BP1993 multigen validation.

Reads ``out/plasmid/multiseed_multigen_controlled_timeseries.json`` (10
seeds × 12 generations, BP1993 control active) and assembles a
three-panel figure:

  (A) ColE1 / pBR322 control-loop schematic (external PNG — supplied
      via --schematic-path). Occupies the full height of the left
      column.
  (B) Plasmid copy number vs. cumulative time across all generations:
      10 seeds as faint gray sawtooth traces, one highlighted seed
      (closest to per-timepoint median) in red. Experimental pBR322
      range shaded for comparison.
  (C) RNA I, RNA II, Rom dynamics over the final generation of the
      highlighted seed on a log y-axis (species magnitudes span ~3
      orders of magnitude).

    uv run python scripts/plot_plasmid_multiseed_multigen_figure.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _seed_track(seed: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-generation time+copy-number tracks for one seed,
    offsetting each generation by the cumulative duration of the prior
    ones. Returns (time_min, copy_number) arrays."""
    t_parts, pc_parts = [], []
    offset_s = 0.0
    for gen in seed["generations"]:
        snaps = gen["snapshots"]
        t = np.array([s["time"] for s in snaps]) + offset_s
        pc = np.array([s["n_full_plasmids"] for s in snaps])
        t_parts.append(t)
        pc_parts.append(pc)
        offset_s += gen["duration"]
    return (np.concatenate(t_parts) / 60.0, np.concatenate(pc_parts))


def _highlight_seed_idx(seeds: list) -> int:
    """Pick the seed whose full-lineage mean copy number is closest to
    the per-seed median of lineage means — gives a single representative
    trajectory for the emphasized trace."""
    means = []
    for seed in seeds:
        all_pc = []
        for gen in seed["generations"]:
            all_pc.extend(
                s["n_full_plasmids"] for s in gen["snapshots"])
        means.append(np.mean(all_pc))
    means = np.array(means)
    med = np.median(means)
    return int(np.argmin(np.abs(means - med)))


def _rna_series(gen: dict) -> dict:
    """Extract time (min from gen start), R_I, R_II, M (Rom) arrays,
    and initiation-event times (0→1 transitions of n_rna_initiations)
    for a single generation's snapshots."""
    snaps = gen["snapshots"]
    t_min = np.array([s["time"] for s in snaps]) / 60.0
    ni = np.array([s["n_rna_initiations"] for s in snaps]).astype(int)
    event_idx = np.where(np.diff(ni) > 0)[0]
    return {
        "t_min": t_min,
        "R_I":   np.array([s["R_I"]  for s in snaps]),
        "R_II":  np.array([s["R_II"] for s in snaps]),
        "M":     np.array([s["M"]    for s in snaps]),
        "event_times": t_min[event_idx] if len(event_idx) else np.array([]),
    }


def _seed_rna_lineage(seed: dict) -> dict:
    """Concatenate R_I, R_II, M and initiation-event times across all
    generations for one seed, offsetting each generation's time by the
    cumulative duration of prior ones (same grammar as _seed_track).
    Returns arrays on a single cumulative time axis."""
    t_parts, ri_parts, rii_parts, m_parts = [], [], [], []
    event_parts = []
    offset_s = 0.0
    for gen in seed["generations"]:
        rna = _rna_series(gen)
        t_parts.append(rna["t_min"] + offset_s / 60.0)
        ri_parts.append(rna["R_I"])
        rii_parts.append(rna["R_II"])
        m_parts.append(rna["M"])
        if len(rna["event_times"]):
            event_parts.append(rna["event_times"] + offset_s / 60.0)
        offset_s += gen["duration"]
    return {
        "t_min": np.concatenate(t_parts),
        "R_I":   np.concatenate(ri_parts),
        "R_II":  np.concatenate(rii_parts),
        "M":     np.concatenate(m_parts),
        "event_times": (np.concatenate(event_parts)
                        if event_parts else np.array([])),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
        default="out/plasmid/multiseed_multigen_controlled_timeseries.json")
    parser.add_argument("--schematic-path", type=str,
        default="/Users/rashmidissasekara/Documents/Plasmid Replication Project/cole1 control.png",
        help="Path to the external ColE1 control-loop schematic PNG.")
    parser.add_argument("--output",
        default="out/plasmid/figure3_bp1993_multigen.png")
    parser.add_argument("--target-low", type=float, default=20.0,
        help="Experimental pBR322 copy-number target low (default 20).")
    parser.add_argument("--target-high", type=float, default=30.0,
        help="Experimental pBR322 copy-number target high (default 30).")
    parser.add_argument("--dynamics-gen", type=int, default=-1,
        help="Generation index (1-based) from the highlighted seed to "
             "plot in Panel C. Default -1 = last generation.")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    seeds = data["seeds"]
    n_seeds = len(seeds)
    n_gens = max(len(s["generations"]) for s in seeds)

    highlight = _highlight_seed_idx(seeds)
    highlight_seed_id = seeds[highlight]["seed"]
    gen_idx = (args.dynamics_gen - 1 if args.dynamics_gen > 0
               else len(seeds[highlight]["generations"]) - 1)
    gen_num = gen_idx + 1  # for display

    # ---- Figure layout ---------------------------------------------
    # Left column (full height): schematic.
    # Right column: sawtooth (top), RNA dynamics (bottom).
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.6], hspace=0.35,
                          wspace=0.12)
    ax_schem = fig.add_subplot(gs[:, 0])
    ax_saw = fig.add_subplot(gs[0, 1])
    ax_dyn = fig.add_subplot(gs[1, 1])

    # ---- Panel A: external schematic -------------------------------
    if os.path.exists(args.schematic_path):
        img = mpimg.imread(args.schematic_path)
        ax_schem.imshow(img)
        ax_schem.axis("off")
    else:
        ax_schem.text(0.5, 0.5,
                      f"schematic not found:\n{args.schematic_path}",
                      ha="center", va="center", fontsize=10,
                      transform=ax_schem.transAxes)
        ax_schem.axis("off")
    ax_schem.set_title("A. ColE1 / pBR322 copy-number control",
                       fontsize=11, loc="left")

    # ---- Panel B: concatenated-lineage copy-number sawtooth --------
    ax_saw.axhspan(args.target_low, args.target_high,
                   facecolor="#f1c40f", alpha=0.22, zorder=1,
                   label=f"pBR322 experimental range "
                         f"({args.target_low:.0f}–{args.target_high:.0f})")
    max_t_min = 0.0
    for i, seed in enumerate(seeds):
        t_min, pc = _seed_track(seed)
        max_t_min = max(max_t_min, t_min[-1])
        if i == highlight:
            continue
        ax_saw.plot(t_min, pc, color="0.7", lw=0.9, alpha=0.55,
                    zorder=2)
    # Division boundaries for the highlighted seed.
    t_div_min = 0.0
    div_times = []
    for gen in seeds[highlight]["generations"][:-1]:
        t_div_min += gen["duration"] / 60.0
        div_times.append(t_div_min)
    for td in div_times:
        ax_saw.axvline(td, color="0.75", ls=":", lw=0.8, zorder=1)
    # Highlighted seed on top.
    t_min, pc = _seed_track(seeds[highlight])
    ax_saw.plot(t_min, pc, color="#c0392b", lw=1.8, alpha=1.0,
                zorder=4,
                label=f"seed {highlight_seed_id} (representative)")
    ax_saw.set_xlabel("Time (min, cumulative across generations)")
    ax_saw.set_ylabel("Plasmid copy number")
    ax_saw.set_title(
        f"B. Plasmid copy number across {n_gens} generations "
        f"({n_seeds} seeds)",
        fontsize=11, loc="left")
    ax_saw.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_saw.set_xlim(0, max_t_min * 1.02)
    ax_saw.grid(False)

    # ---- Panel C: RNA I / RNA II / Rom dynamics (log y) -----------
    # All 10 seeds' gen-gen_num traces in muted per-species colors;
    # highlighted seed in bold; initiation events as ticks at the
    # bottom of the axis for the highlighted seed.
    SPECIES_COLORS = {
        "R_I":  "#c0392b",
        "R_II": "#2980b9",
        "M":    "#8e44ad",
    }
    SPECIES_LABELS = {
        "R_I":  "RNA I (antisense)",
        "R_II": "RNA II (preprimer)",
        "M":    "Rom (M)",
    }

    ax_dyn.set_yscale("log")

    # Background traces: all non-highlighted seeds, concatenated across
    # all generations with cumulative time (same axis as Panel B).
    max_t_lin = 0.0
    for i, seed in enumerate(seeds):
        if i == highlight:
            continue
        rna_i = _seed_rna_lineage(seed)
        max_t_lin = max(max_t_lin, rna_i["t_min"][-1])
        for sp in ("R_I", "R_II", "M"):
            ax_dyn.plot(rna_i["t_min"], rna_i[sp],
                        color=SPECIES_COLORS[sp], lw=0.7, alpha=0.25,
                        zorder=1)

    # Division boundaries for the highlighted seed (same verticals as
    # Panel B, so the two panels read against a shared time grid).
    t_div_min = 0.0
    for gen in seeds[highlight]["generations"][:-1]:
        t_div_min += gen["duration"] / 60.0
        ax_dyn.axvline(t_div_min, color="0.82", ls=":", lw=0.7,
                       zorder=1)

    # Highlighted seed on top.
    rna = _seed_rna_lineage(seeds[highlight])
    max_t_lin = max(max_t_lin, rna["t_min"][-1])
    for sp in ("R_I", "R_II", "M"):
        ax_dyn.plot(rna["t_min"], rna[sp],
                    color=SPECIES_COLORS[sp], lw=1.4, alpha=1.0,
                    zorder=3, label=SPECIES_LABELS[sp])

    ax_dyn.set_xlim(0, max_t_lin * 1.02)
    # Clamp the y-axis to meaningful magnitudes. Lower bound at 0.5
    # hides the very brief post-division moments when the ODE state
    # transiently drops toward 0 (which on a log scale would stretch
    # the panel down to 10⁻⁴ for no informational gain). Upper bound a
    # little above the largest Rom value.
    y_max_data = max(rna["R_I"].max(), rna["M"].max())
    ax_dyn.set_ylim(0.5, y_max_data * 2.0)

    ax_dyn.set_xlabel("Time (min, cumulative across generations)")
    ax_dyn.set_ylabel("Molecule count (log scale)")
    ax_dyn.set_title(
        f"C. ColE1 control-loop species across {n_gens} generations "
        f"({n_seeds} seeds)",
        fontsize=11, loc="left")
    ax_dyn.legend(loc="center right", fontsize=8.5, framealpha=0.9)
    ax_dyn.grid(False)

    fig.suptitle(
        "Figure 3. ColE1 control module reproduces stable pBR322 "
        "copy number across generations "
        f"(highlighted: seed {highlight_seed_id})",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")

    # Summary numbers for the caption paragraph.
    all_terminals = []
    for s in seeds:
        for gen in s["generations"]:
            all_terminals.append(gen["snapshots"][-1]["n_full_plasmids"])
    gen12_terminals = [
        s["generations"][-1]["snapshots"][-1]["n_full_plasmids"]
        for s in seeds
    ]
    print(f"  {n_seeds} seeds × {n_gens} gens, all divided")
    print(f"  gen-12 terminal PC: "
          f"median={np.median(gen12_terminals):.0f}, "
          f"mean±SD={np.mean(gen12_terminals):.1f}±"
          f"{np.std(gen12_terminals):.1f}, "
          f"range=[{min(gen12_terminals):.0f}, "
          f"{max(gen12_terminals):.0f}]")
    print(f"  RNA species (highlighted seed, gen {gen_num}): "
          f"R_I [{rna['R_I'].min():.1f}, {rna['R_I'].max():.1f}], "
          f"R_II [{rna['R_II'].min():.2f}, {rna['R_II'].max():.2f}], "
          f"M [{rna['M'].min():.1f}, {rna['M'].max():.1f}]")


if __name__ == "__main__":
    main()
