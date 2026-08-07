"""Stage 1 sanity-check diagnostic plotting.

Reads the SQLite history written by ``run_sim.py`` and produces an
expected-vs-observed HTML report comparing v2ecoli's simulated behavior
under the Stage 1 inputs to the Stage 1 PDF's prescribed values.

Output: ``docs/stage1_sanity_check.html`` plus per-panel PNG files
embedded inline (base64).

Usage:
    python studies/stage1_sanity_check/plot_diagnostics.py \\
        --db   out/stage1_sanity_runs.db \\
        --cache out/cache_stage1 \\
        --out  docs/stage1_sanity_check.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Bulk indices resolved from the ParCa cache (Stage 1 fixture)
BULK_DNAA_MONOMER = 11565  # PD03831[c]    — unbound DnaA
BULK_DNAA_ATP = 10822      # MONOMER0-160[c]
BULK_DNAA_ADP = 11114      # MONOMER0-4565[c]
BULK_DNAA_MRNA = 15272     # TU00259[c]    — dnaA-containing TU

# Stage 1 brief values (what the heuristic-parameter PDF prescribes)
STAGE1 = {
    "doubling_time_min": 150.0,    # ABT minimal glycerol
    "c_period_min": 70.0,
    "d_period_min": 30.0,
    "replication_speed_kbpm": 66.31,
    "dnaa_translation_eff": 1.0,
    "dnaa_decay_rate": 0.0,
    "dnaa_mrna_per_gene_per_min": 1.5,
    "oric_lo_box_count": 8,
    "dnaa_atp_kd_nM": 30.0,
    "dnaa_adp_kd_nM": 100.0,
    "dnaa_atp_hydrolysis_rate_inv_min": 0.046,
    "oric_lo_box_kd_nM_range": "50 → 1",
    "hda_hydrolysis_rate_inv_min_per_fork": 40,
    "data_hydrolysis_rate_inv_min": 12,
    "dars1_rate_inv_min": 5,
    "dars2_rate_inv_min": 10,
    "seqa_sequestration_min": 0,
}
# v2ecoli canonical acetate condition defaults
CANONICAL = {
    "doubling_time_min": 136.0,    # condition_defs.tsv acetate row
    "c_period_min": 40.0,          # genome_len / (967 nt/s × 2)
    "d_period_min": 20.0,          # parameters.tsv d_period
    "replication_speed_kbpm": 116.0, # 967 nt/s × 60 × 2 forks / 1000
    "dnaa_translation_eff": 0.35,   # Li 2014 Table S4
    "dnaa_decay_rate_per_min": 0.00248,  # 4.13e-5 /s × 60
    "dnaa_atp_kd_nM": 29.0,         # equilibrium_reaction_rates.tsv
    "dnaa_adp_kd_nM": None,         # species exists but no rate wired
}


@dataclass
class GenRow:
    gen: int
    t: float            # global_time, sec
    cell_mass: float    # fg
    dry_mass: float     # fg
    dna_mass: float     # fg
    protein_mass: float # fg
    volume: float       # fL
    growth_rate: float  # 1/s
    dnaa_monomer: int
    dnaa_atp: int
    dnaa_adp: int
    dnaa_mrna: int
    n_oric: int
    free_boxes: int
    total_boxes: int
    n_fork: int
    n_chrom: int
    fork_coords: list       # list[int] — positions along the genome (-2.3M to +2.3M nt)
    fork_domains: list      # list[int] — domain index for each fork (groups sister forks)


def read_db(db_path: str, subsample: int = 1) -> dict[int, list[GenRow]]:
    """Stream-read the SQLite history into per-gen lists of compact rows.

    ``subsample=N`` keeps every Nth tick (default 1 = keep all). Used for
    the multi-condition cross-comparison plots where 1-second resolution
    isn't needed and JSON parsing dominates wall time."""
    print(f"[{time.strftime('%H:%M:%S')}] Reading {db_path} (subsample={subsample}) ...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sim_to_gen: dict[str, int] = {}
    for r in conn.execute(
        "SELECT simulation_id, name FROM simulations ORDER BY started_at"
    ):
        # e.g. name = 'stage1-sanity-seed0-gen3'
        name = r["name"]
        try:
            g = int(name.rsplit("gen", 1)[1])
        except (ValueError, IndexError):
            continue
        sim_to_gen[r["simulation_id"]] = g

    rows_per_gen: dict[int, list[GenRow]] = {g: [] for g in sim_to_gen.values()}
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    print(f"  reading {total} history rows ...")
    t0 = time.time()

    cur = conn.execute(
        "SELECT simulation_id, step, global_time, state FROM history "
        "ORDER BY simulation_id, step"
    )
    n_done = 0
    for sid, step, t, state_json in cur:
        g = sim_to_gen.get(sid)
        if g is None:
            continue
        if subsample > 1 and (step % subsample) != 0:
            continue
        s = json.loads(state_json)
        bulk = s.get("bulk", [])
        listeners = s.get("listeners", {}) or {}
        mass = listeners.get("mass", {}) or {}
        repl = listeners.get("replication_data", {}) or {}
        # Authoritative fork count from the replication_data listener
        # (the active_replisome unique-molecule list can include stale,
        # already-terminated replisomes for a brief window after a round ends).
        fork_coords_list = list(repl.get("fork_coordinates", []) or [])
        n_fork = len(fork_coords_list)
        n_chrom = len(s.get("full_chromosome", []) or [])

        row = GenRow(
            gen=g,
            t=float(t),
            cell_mass=float(mass.get("cell_mass", 0)),
            dry_mass=float(mass.get("dry_mass", 0)),
            dna_mass=float(mass.get("dna_mass", 0)),
            protein_mass=float(mass.get("protein_mass", 0)),
            volume=float(mass.get("volume", 0)),
            growth_rate=float(mass.get("instantaneous_growth_rate", 0)),
            # Each bulk entry is [id, count, *submasses_per_compartment]
            dnaa_monomer=int(bulk[BULK_DNAA_MONOMER][1]) if bulk else 0,
            dnaa_atp=int(bulk[BULK_DNAA_ATP][1]) if bulk else 0,
            dnaa_adp=int(bulk[BULK_DNAA_ADP][1]) if bulk else 0,
            dnaa_mrna=int(bulk[BULK_DNAA_MRNA][1]) if bulk else 0,
            n_oric=int(repl.get("number_of_oric", 0) or 0),
            free_boxes=int(repl.get("free_DnaA_boxes", 0) or 0),
            total_boxes=int(repl.get("total_DnaA_boxes", 0) or 0),
            n_fork=n_fork,
            n_chrom=n_chrom,
            fork_coords=fork_coords_list,
            fork_domains=list(repl.get("fork_domains", []) or []),
        )
        rows_per_gen[g].append(row)
        n_done += 1
        if n_done % 2000 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (total - n_done) / rate
            print(f"    {n_done}/{total}  ({rate:.0f} rows/s, ETA {eta:.0f}s)")

    print(f"  done in {time.time() - t0:.1f}s")
    conn.close()
    return rows_per_gen


def _rows_to_np(rows: list[GenRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "t":             np.array([r.t for r in rows]),
        "cell_mass":     np.array([r.cell_mass for r in rows]),
        "dry_mass":      np.array([r.dry_mass for r in rows]),
        "dna_mass":      np.array([r.dna_mass for r in rows]),
        "protein_mass":  np.array([r.protein_mass for r in rows]),
        "volume":        np.array([r.volume for r in rows]),
        "growth_rate":   np.array([r.growth_rate for r in rows]),
        "dnaa_monomer":  np.array([r.dnaa_monomer for r in rows]),
        "dnaa_atp":      np.array([r.dnaa_atp for r in rows]),
        "dnaa_adp":      np.array([r.dnaa_adp for r in rows]),
        "dnaa_mrna":     np.array([r.dnaa_mrna for r in rows]),
        "n_oric":        np.array([r.n_oric for r in rows]),
        "free_boxes":    np.array([r.free_boxes for r in rows]),
        "total_boxes":   np.array([r.total_boxes for r in rows]),
        "n_fork":        np.array([r.n_fork for r in rows]),
        "n_chrom":       np.array([r.n_chrom for r in rows]),
        # Variable-length per tick — keep as Python lists, not arrays
        "fork_coords":   [r.fork_coords for r in rows],
        "fork_domains":  [r.fork_domains for r in rows],
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def panel_cell_mass(per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    t_offset = 0.0
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        ax.plot(
            (d["t"] + t_offset) / 60, d["cell_mass"],
            color=c, label=f"gen {g}", lw=1.4,
        )
        t_offset += d["t"][-1]
    ax.set_xlabel("Time (min, lineage cumulative)")
    ax.set_ylabel("Cell mass (fg)")
    ax.set_title("Cell mass over the 5-generation lineage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return fig_to_b64(fig)


def panel_doubling_time(per_gen):
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = sorted(per_gen.keys())
    obs_dt = [per_gen[g]["t"][-1] / 60 for g in gens]
    ax.bar(gens, obs_dt, color="#4682B4", alpha=0.8, label="Observed")
    ax.axhline(CANONICAL["doubling_time_min"], color="#FF7F0E",
               ls=":", lw=1.5, label=f'v2ecoli canonical acetate target ({CANONICAL["doubling_time_min"]:.0f} min)')
    ax.axhline(STAGE1["doubling_time_min"], color="#D62728",
               ls="--", lw=1.5, label=f'Stage 1 brief target ({STAGE1["doubling_time_min"]:.0f} min)')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cell cycle duration (min)")
    ax.set_title("Per-generation cell cycle: observed vs canonical vs Stage 1 brief")
    ax.set_xticks(gens)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    return fig_to_b64(fig)


def panel_dnaa_species(per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    g_first = min(per_gen)
    d = per_gen[g_first]
    t_min = d["t"] / 60
    ax.plot(t_min, d["dnaa_monomer"], label="DnaA (unbound) — PD03831", color="#2CA02C", lw=1.4)
    ax.plot(t_min, d["dnaa_atp"], label="DnaA-ATP — MONOMER0-160", color="#1F77B4", lw=1.4)
    ax.plot(t_min, d["dnaa_adp"], label="DnaA-ADP — MONOMER0-4565", color="#D62728", lw=1.4)
    total = d["dnaa_monomer"] + d["dnaa_atp"] + d["dnaa_adp"]
    ax.plot(t_min, total, label="Total DnaA", color="black", lw=2, ls="--")
    ax.set_xlabel("Time within gen 1 (min)")
    ax.set_ylabel("Molecule count")
    ax.set_title("DnaA species over gen 1 — unbound / ATP / ADP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return fig_to_b64(fig)


def panel_dnaa_per_oric(per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        total = d["dnaa_monomer"] + d["dnaa_atp"] + d["dnaa_adp"]
        oric = np.maximum(d["n_oric"], 1)
        ax.plot(d["t"] / 60, total / oric, color=c, label=f"gen {g}", lw=1.4)
    ax.set_xlabel("Time within generation (min)")
    ax.set_ylabel("Total DnaA / oriC")
    ax.set_title("DnaA per oriC across generations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return fig_to_b64(fig)


def panel_dnaa_mrna(per_gen):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        ax1.plot(d["t"] / 60, d["dnaa_mrna"], color=c, label=f"gen {g}", lw=1.4)
    ax1.set_xlabel("Time within generation (min)")
    ax1.set_ylabel("dnaA mRNA count (TU00259)")
    ax1.set_title("dnaA mRNA over time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # Mean mRNA per generation (rough)
    gens = sorted(per_gen.keys())
    means = [per_gen[g]["dnaa_mrna"].mean() for g in gens]
    ax2.bar(gens, means, color="#4682B4", alpha=0.8)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean dnaA mRNA count")
    ax2.set_title("Per-gen average dnaA mRNA (TU00259)")
    ax2.set_xticks(gens)
    ax2.grid(True, alpha=0.3, axis="y")
    return fig_to_b64(fig)


def panel_dnaa_decay(per_gen):
    """If DnaA degradation = 0, the unbound DnaA pool should be stable
    (only diluted by division and consumed by binding events). Plot
    total DnaA over time and overlay an exponential decay reference."""
    fig, ax = plt.subplots(figsize=(9, 4))
    g_first = min(per_gen)
    d = per_gen[g_first]
    total = d["dnaa_monomer"] + d["dnaa_atp"] + d["dnaa_adp"]
    ax.plot(d["t"] / 60, total, label="Observed total DnaA", color="black", lw=1.6)
    # Reference: half-life infinite (Stage 1 says decay = 0)
    ax.axhline(total[0], color="#1F77B4", ls="--", lw=1,
               label=f"Stage 1 prediction (zero decay → flat = {total[0]:.0f})")
    ax.set_xlabel("Time within gen 1 (min)")
    ax.set_ylabel("Total DnaA count")
    ax.set_title("DnaA stability check — Stage 1 brief says decay rate = 0")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return fig_to_b64(fig)


def _detect_events(d: dict) -> dict:
    """For one generation, find initiation events (oriC count jumps up),
    termination events (active_replisome count drops), and division (final t).
    Returns dict with arrays of event times (min)."""
    t = d["t"] / 60
    oric = d["n_oric"]
    forks = d["n_fork"]
    chroms = d["n_chrom"]

    init_times = t[1:][np.diff(oric) > 0]
    # Termination: number of forks drops AND chromosome count increases (or stays)
    term_times = t[1:][(np.diff(forks) < 0) & (np.diff(chroms) >= 0)]
    division_t = t[-1]

    return {
        "initiations_min": init_times.tolist(),
        "terminations_min": term_times.tolist(),
        "division_min": float(division_t),
    }


def panel_replication_timeline(per_gen):
    """Per-gen lifeline: t=0 (birth), initiation(s), termination(s), division.
    Annotates C-period (init→term) and D-period (term→division) for each
    gen, with Stage 1 brief + canonical reference bars."""
    fig, ax = plt.subplots(figsize=(11, max(3, 1.0 * len(per_gen) + 1.5)))
    y_pos = 0
    yticks, ylabels = [], []
    for g, d in sorted(per_gen.items()):
        ev = _detect_events(d)
        # Birth → division horizontal line
        ax.plot([0, ev["division_min"]], [y_pos, y_pos], color="#888", lw=2, alpha=0.5)
        # Initiations
        for t_init in ev["initiations_min"]:
            ax.scatter(t_init, y_pos, marker="^", s=110, color="#2CA02C", zorder=5,
                       label="Initiation" if (g == min(per_gen) and t_init == ev["initiations_min"][0]) else None)
        # Terminations
        for t_term in ev["terminations_min"]:
            ax.scatter(t_term, y_pos, marker="v", s=110, color="#D62728", zorder=5,
                       label="Termination" if (g == min(per_gen) and t_term == ev["terminations_min"][0]) else None)
        # Division marker
        ax.scatter(ev["division_min"], y_pos, marker="s", s=120, color="black", zorder=5,
                   label="Division" if g == min(per_gen) else None)
        yticks.append(y_pos)
        ylabels.append(f"gen {g}")
        y_pos += 1

    # Reference timing bars (top)
    bar_y = y_pos + 0.5
    # Canonical: C=40, D=20
    ax.barh(bar_y, CANONICAL["c_period_min"], left=0, height=0.3, color="#FF7F0E",
            alpha=0.4, label=f'Canonical C={CANONICAL["c_period_min"]:.0f} min')
    ax.barh(bar_y, CANONICAL["d_period_min"], left=CANONICAL["c_period_min"],
            height=0.3, color="#FF7F0E", alpha=0.7, label=f'Canonical D={CANONICAL["d_period_min"]:.0f} min')
    # Stage 1: C=70, D=30
    ax.barh(bar_y + 0.4, STAGE1["c_period_min"], left=0, height=0.3, color="#D62728",
            alpha=0.4, label=f'Stage 1 C={STAGE1["c_period_min"]:.0f} min')
    ax.barh(bar_y + 0.4, STAGE1["d_period_min"], left=STAGE1["c_period_min"],
            height=0.3, color="#D62728", alpha=0.7, label=f'Stage 1 D={STAGE1["d_period_min"]:.0f} min')

    yticks.extend([bar_y, bar_y + 0.4])
    ylabels.extend(["canonical brief", "Stage 1 brief"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time within generation (min)")
    ax.set_title("Replication timeline — initiation (▲), termination (▼), division (■)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    return fig_to_b64(fig)


def panel_replication(per_gen):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        ax1.plot(d["t"] / 60, d["n_fork"], color=c, label=f"gen {g}", lw=1.4)
        ax2.plot(d["t"] / 60, d["n_oric"], color=c, lw=1.4)
        ax3.plot(d["t"] / 60, d["n_chrom"], color=c, lw=1.4)
    for ax, ylab, title in [
        (ax1, "Active replisome forks", "Fork count"),
        (ax2, "Origins of replication", "oriC count"),
        (ax3, "Full chromosomes", "Chromosome count"),
    ]:
        ax.set_xlabel("Time within generation (min)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    ax1.legend(loc="best", fontsize=8)
    return fig_to_b64(fig)


def panel_dnaa_boxes(per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        ax.plot(d["t"] / 60, d["total_boxes"], color=c, label=f"gen {g} (total)", lw=1.4)
        ax.plot(d["t"] / 60, d["free_boxes"], color=c, ls=":", label=f"gen {g} (free)", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time within generation (min)")
    ax.set_ylabel("DnaA-box count")
    ax.set_title("DnaA-box occupancy — total + free (memory: 307 boxes annotated, ~614 mid-replication)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig_to_b64(fig)


def panel_growth_rate(per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(per_gen)))
    for (g, d), c in zip(sorted(per_gen.items()), colors):
        # instantaneous_growth_rate is in 1/s; convert to doubling time min
        gr = d["growth_rate"]
        # Skip the first few ticks where growth_rate may be 0
        valid = gr > 0
        if not valid.any():
            continue
        ax.plot(d["t"][valid] / 60, gr[valid] * 3600, color=c, label=f"gen {g}", lw=1.4)
    ax.axhline(np.log(2) * 3600 / CANONICAL["doubling_time_min"] / 60,
               color="#FF7F0E", ls=":", lw=1, label=f'Canonical acetate ({CANONICAL["doubling_time_min"]:.0f} min)')
    ax.axhline(np.log(2) * 3600 / STAGE1["doubling_time_min"] / 60,
               color="#D62728", ls="--", lw=1, label=f'Stage 1 brief ({STAGE1["doubling_time_min"]:.0f} min)')
    ax.set_xlabel("Time within generation (min)")
    ax.set_ylabel("Growth rate (1/hr)")
    ax.set_title("Instantaneous specific growth rate vs Stage 1 expected")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig_to_b64(fig)


# ----------------------------------------------------------------------------
# Multi-condition baseline comparison
# ----------------------------------------------------------------------------
# Five canonical media conditions, no Stage 1 overrides. Used to show
# how v2ecoli's chromosome dynamics differ across slow / moderate / fast
# growth out of the box.

CONDITIONS_5 = [
    # (name,        db_path,                                  cache_dir,                          color,      tau_target_min)
    ("with_aa",    "out/with_aa_canonical_runs.db",           "out/cache_with_aa_canonical",      "#d62728",  25.0),   # fastest
    ("basal",      "out/basal_canonical_runs.db",             "out/cache_basal_canonical",        "#ff7f0e",  44.0),
    ("succinate",  "out/succinate_canonical_runs.db",         "out/cache_succinate_canonical",    "#bcbd22",  82.0),
    ("no_oxygen",  "out/no_oxygen_canonical_runs.db",         "out/cache_no_oxygen_canonical",    "#17becf", 100.0),
    ("acetate",    "out/acetate_apr24_test.db",               "out/cache_acetate_apr24",          "#2ca02c", 136.0),   # slowest
]

# Stage 1 PDF — a model-override scenario, not a peer canonical
# condition. It carries a NEW media file (MIX0-845: AB + 0.2% glycerol)
# we authored, plus six post-ParCa knobs (C=70, D=30, dnaA txn × 15,
# dnaA constitutive, dnaA stable, dnaA TE = 1.0). Compared against
# canonical acetate as the closest analog (both slow-growth conditions),
# not against the full 5-canonical baseline.
CONDITIONS_STAGE1 = [
    ("acetate (canonical)",
     "out/acetate_canonical_runs.db",
     "out/cache_acetate_canonical",       "#2ca02c", 136.0),
    ("glycerol (Stage 1 PDF)",
     "out/glycerol_stage1_1gen.db",
     "out/cache_glycerol_stage1_full",    "#c2410c", 150.0),
]


# MIX0-845 media composition — shown collapsibly in the Stage 1 section
# to make explicit that this is a media file we authored (not a vEcoli
# canonical recipe).
MIX0_845_RECIPE = [
    ("NA+",                  "135.15"),
    ("Pi",                   "63.851"),
    ("CL-",                  "55.389"),
    ("AMMONIUM",             "30.272"),
    ("K+",                   "21.883"),
    ("SULFATE",              "15.077"),
    ("MG+2",                 "2.1"),
    ("CA+2",                 "0.09"),
    ("FE+3",                 "0.0031"),
    ("WATER",                "Infinity"),
    ("GLYCEROL",             "13.0 (= 0.2 % w/v)"),
    ("CARBON-DIOXIDE",       "Infinity"),
    ("OXYGEN-MOLECULE",      "Infinity"),
    ("CO+2, FE+2, MN+2, NI+2, ZN+2",  "Infinity (trace metals)"),
    ("L-SELENOCYSTEINE",     "Infinity"),
]


def mix0_845_recipe_html() -> str:
    rows = "".join(
        f'<tr><td><code>{name}</code></td><td>{conc}</td></tr>'
        for name, conc in MIX0_845_RECIPE
    )
    return (
        '<details style="margin: 12px 0;">'
        '<summary style="cursor:pointer; font-weight:600; font-size:13px; '
        'color:var(--hi);">MIX0-845 media composition (mmol / L)</summary>'
        '<table style="margin-top:8px; max-width:520px;">'
        '<thead><tr><th>Molecule</th><th>Concentration</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p style="font-size:11px; color:var(--muted); margin-top:6px;">'
        'Authored as part of this work, patterned after MIX0-57 '
        '(AB + 0.2 % glucose) with GLC swapped for GLYCEROL at 13 mmol/L. '
        'No glycerol-specific RNA-seq backing — per-condition transcript fit '
        'uses the glucose baseline column. <strong>Model-override scenario, '
        'not a peer canonical condition.</strong></p>'
        '</details>'
    )


def read_conditions(conditions, subsample: int = 50) -> dict[str, dict]:
    """Read history for each (name, db, cache, color, tau) tuple."""
    out = {}
    for name, db_path, _cache, _color, _tau in conditions:
        if not os.path.isfile(db_path) or os.path.getsize(db_path) < 4096:
            print(f"  SKIP {name}: {db_path} missing or empty (likely cleaned up)")
            continue
        try:
            rows_per_gen = read_db(db_path, subsample=subsample)
        except sqlite3.OperationalError as e:
            print(f"  SKIP {name}: {db_path} not yet populated ({e})")
            continue
        gen1 = rows_per_gen.get(1, [])
        if gen1:
            out[name] = _rows_to_np(gen1)
            print(f"  {name}: {len(gen1)} subsampled rows from gen 1")
    return out


def read_all_conditions(subsample: int = 50) -> dict[str, dict]:
    """Read each condition's first-gen history with subsampling. Returns
    {condition_name: rows-as-numpy-dict}."""
    out = {}
    for name, db_path, _cache, _color, _tau in CONDITIONS_5:
        if not os.path.isfile(db_path) or os.path.getsize(db_path) < 4096:
            print(f"  SKIP {name}: {db_path} missing or empty (likely cleaned up)")
            continue
        try:
            rows_per_gen = read_db(db_path, subsample=subsample)
        except sqlite3.OperationalError as e:
            print(f"  SKIP {name}: {db_path} not yet populated ({e})")
            continue
        gen1 = rows_per_gen.get(1, [])
        if gen1:
            out[name] = _rows_to_np(gen1)
            print(f"  {name}: {len(gen1)} subsampled rows from gen 1")
    return out


def extract_condition_summary(per_cond: dict[str, dict]) -> list[dict]:
    """Per-condition summary row for the comparison table."""
    # v2ecoli's canonical C is a global scalar derived from
    # parameters.tsv replisome_elongation_rate. At 967 nt/s with the
    # 4,641,652 nt genome → C ≈ 40 min for every condition.
    C_GLOBAL_MIN = 40.0

    # vEcoli reference cycle times (from runs we executed:
    # logs/vecoli_with_aa_*, logs/vecoli_remaining_*, logs/vecoli_acetate_*).
    # Same condition_defs.tsv, same simData built by vEcoli's own ParCa.
    VECOLI_OBSERVED_MIN = {
        "with_aa":   22.5,
        "basal":     42.2,
        "succinate": 92.3,
        "no_oxygen": 119.6,
        "acetate":   130.5,
    }

    out = []
    for name, db, cache, color, tau in CONDITIONS_5:
        if name not in per_cond:
            continue
        d = per_cond[name]
        out.append({
            "name": name,
            "color": color,
            "tau_target_min": tau,
            "tau_observed_min": d["t"][-1] / 60.0,
            "tau_vecoli_min": VECOLI_OBSERVED_MIN.get(name),
            "initial_dry_mass": float(d["dry_mass"][0]),
            "final_dry_mass": float(d["dry_mass"].max()),
            "mass_fold_change": float(d["dry_mass"].max() / max(d["dry_mass"][0], 1)),
            "max_oric": int(d["n_oric"].max()),
            "max_forks": int(d["n_fork"].max()),
            "max_chroms": int(d["n_chrom"].max()) + 1,  # +1 to count the divided chromosome
            "initial_oric": int(d["n_oric"][0]),
            "initial_forks": int(d["n_fork"][0]),
            # Strict biological definition: multi-fork iff Td < C
            "multi_fork": tau < C_GLOBAL_MIN,
        })
    return out


def _plot_multi(per_cond, get_y, ylabel, title, ylim_bottom=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, db, cache, color, tau in CONDITIONS_5:
        d = per_cond.get(name)
        if d is None:
            continue
        ax.plot(d["t"] / 60, get_y(d), color=color, lw=1.5,
                label=f"{name} (τ={tau:.0f} min)")
    ax.set_xlabel("Time within generation (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return fig_to_b64(fig)


def panel_multi_mass(per_cond):
    return _plot_multi(per_cond, lambda d: d["dry_mass"], "Dry mass (fg)",
                       "Cell-mass trajectory — 5 conditions, 1 generation each")


def panel_multi_oric(per_cond):
    return _plot_multi(per_cond, lambda d: d["n_oric"], "oriC count",
                       "Origins of replication over the cell cycle",
                       ylim_bottom=0)


def panel_multi_forks(per_cond):
    return _plot_multi(per_cond, lambda d: d["n_fork"], "Active replication forks",
                       "Replication-fork count over the cell cycle",
                       ylim_bottom=0)


def panel_multi_chromosomes(per_cond):
    """Full-chromosome count makes multi-fork visible: at with_aa,
    chromosomes go 1 → 2 → 4 as overlapping rounds terminate; other
    conditions stay at chroms ≤ 2."""
    return _plot_multi(per_cond, lambda d: d["n_chrom"], "Full-chromosome count",
                       "Full chromosomes accumulating as replication rounds terminate",
                       ylim_bottom=0)


def make_replication_animation(per_cond: dict, out_gif: str,
                                 n_frames: int = 100, fps: int = 12,
                                 conditions: list | None = None,
                                 figsize: tuple = (18, 6.6)) -> str:
    """Synced theta-replication animation with fork-count strip.

    ``conditions`` defaults to the 5 canonical conditions; pass a custom
    list (same tuple shape as ``CONDITIONS_5``) for an alternate panel
    layout — e.g. the 2-condition canonical-vs-Stage-1 comparison.

    Top row: chromosome(s) as circles, oriC at top, terminus at bottom,
    active forks as red dots moving along the circle (positions read
    directly from ``fork_coords``). Bottom row: fork-count vs time with
    a vertical bar tracking the animation cursor.

    All panels share a single global time axis equal to the longest
    condition. When a faster condition divides before the global time
    ends, its theta panel freezes at the final state but the time bar
    keeps moving on its fork-count strip.
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Circle as MplCircle, Arc as MplArc

    if conditions is None:
        conditions = CONDITIONS_5

    GENOME_HALF_NT = 2_320_826  # half of K-12 genome length (4,641,652 nt)

    # Global time axis = max gen length across all conditions (acetate).
    t_max_global = max(
        float(per_cond[name]["t"][-1] / 60)
        for name, *_ in conditions if name in per_cond
    )
    frame_times = np.linspace(0, t_max_global, n_frames)

    # Precompute per-condition frame data on the SHARED time axis.
    # When global time exceeds a condition's gen length, clamp to the
    # last available row (theta panel will visually freeze).
    frame_data = {}
    for name, db, cache, color, tau in conditions:
        if name not in per_cond:
            continue
        d = per_cond[name]
        t_min = d["t"] / 60
        t_end = float(t_min[-1])
        frames = []
        for ft in frame_times:
            ft_clamped = min(ft, t_end)
            idx = int(np.argmin(np.abs(t_min - ft_clamped)))
            frames.append({
                "t":            float(ft),
                "ended":        bool(ft > t_end),
                "fork_coords":  d["fork_coords"][idx],
                "fork_domains": d["fork_domains"][idx],
                "n_chrom":      int(d["n_chrom"][idx]),
                "n_oric":       int(d["n_oric"][idx]),
                "n_fork":       int(d["n_fork"][idx]),
            })
        frame_data[name] = frames

    # 2-row layout: theta panels on top, fork-count strip charts below.
    # ncols adapts to the conditions list length.
    ncols = len(conditions)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, height_ratios=[3.0, 1.2],
                          hspace=0.32, wspace=0.18)
    theta_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    line_axes = [fig.add_subplot(gs[1, i]) for i in range(ncols)]

    # Pre-draw static strip charts on the bottom row: oriC (green),
    # active forks (red), full chromosomes (blue). Each strip uses its
    # OWN gen length on the x-axis so fast-condition dynamics aren't
    # squished. The vertical time bar tracks current sim time clamped
    # to that strip's gen end.
    global_ymax = max(
        max(int(per_cond[name]["n_fork"].max()),
            int(per_cond[name]["n_oric"].max()),
            int(per_cond[name]["n_chrom"].max()))
        for name, *_ in conditions if name in per_cond
    )
    cond_t_end: dict[str, float] = {}
    time_bars = []
    for i, (name, db, cache, color, tau) in enumerate(conditions):
        line_ax = line_axes[i]
        if name in per_cond:
            d = per_cond[name]
            tt = d["t"] / 60
            line_ax.step(tt, d["n_oric"],  where="post",
                         color="#16a34a", lw=1.4, label="oriC")
            line_ax.step(tt, d["n_fork"],  where="post",
                         color="#dc2626", lw=1.4, label="forks")
            line_ax.step(tt, d["n_chrom"], where="post",
                         color="#2563eb", lw=1.4, label="chroms")
            cond_t_end[name] = float(tt[-1])
            line_ax.set_xlim(0, cond_t_end[name])
            # Initiation events: oriC count steps up
            init_idx = np.where(np.diff(d["n_oric"]) > 0)[0] + 1
            # Termination events: full-chromosome count steps up
            term_idx = np.where(np.diff(d["n_chrom"]) > 0)[0] + 1
            y_top = global_ymax + 0.35
            y_bot = -0.05
            for idx in init_idx:
                line_ax.axvline(tt[idx], color="#16a34a", ls=":", lw=0.9, alpha=0.65)
                line_ax.scatter(tt[idx], y_top, marker="^", s=42,
                                color="#16a34a", edgecolor="black",
                                linewidths=0.5, zorder=6, clip_on=False)
            for idx in term_idx:
                line_ax.axvline(tt[idx], color="#2563eb", ls=":", lw=0.9, alpha=0.55)
                line_ax.scatter(tt[idx], y_bot, marker="v", s=38,
                                color="#2563eb", edgecolor="black",
                                linewidths=0.5, zorder=6, clip_on=False)
        else:
            line_ax.set_xlim(0, t_max_global)
        line_ax.set_ylim(-0.3, global_ymax + 0.7)
        line_ax.set_xlabel("t (min)", fontsize=8)
        if i == 0:
            line_ax.set_ylabel("count", fontsize=8)
        line_ax.tick_params(labelsize=7)
        line_ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            line_ax.spines[s].set_visible(False)
        # Pre-create the moving vertical bar; we'll just set xdata each frame
        bar = line_ax.axvline(0, color="#111827", lw=1.4, zorder=5)
        time_bars.append(bar)

    # Pre-init each theta panel's static elements (title, axis limits).
    # ylim extends below -1.5 to leave room for two lines of text.
    for i, (name, db, cache, color, tau) in enumerate(conditions):
        ax = theta_axes[i]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.85, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{name} (τ={tau:.0f} min)", color=color, fontsize=11)

    # Figure-level legend anchored to the right of the bottom strip row
    # (outside the plot area so it doesn't overlap the data).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#16a34a", lw=1.6, label="oriC"),
        Line2D([0], [0], color="#dc2626", lw=1.6, label="forks"),
        Line2D([0], [0], color="#2563eb", lw=1.6, label="chroms"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#16a34a",
               markeredgecolor="black", markersize=8, label="initiation"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#2563eb",
               markeredgecolor="black", markersize=8, label="termination"),
        Line2D([0], [0], color="#111827", lw=1.4, label="time cursor"),
    ]
    line_axes[-1].legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(1.04, 0.5), fontsize=8, frameon=False,
        handlelength=1.6, labelspacing=0.7,
    )
    fig.subplots_adjust(right=0.93)

    def fork_xy(coord_nt: float) -> tuple[float, float]:
        """Map a fork's genome coordinate to (x, y) on a unit circle.
        oriC at (0, 1), terminus at (0, -1). Positive coords go to the
        right side, negative coords to the left side.

        Adds a small angular bias based on sign so sister forks at coord≈0
        (right after initiation) don't render exactly stacked at the top.
        """
        frac = coord_nt / GENOME_HALF_NT  # in [-1, +1]
        bias = np.deg2rad(3.0) * (1 if coord_nt >= 0 else -1)
        angle_from_top = np.pi * frac + bias
        return np.sin(angle_from_top), np.cos(angle_from_top)

    def chromosome_layout(n_chrom: int) -> list[tuple[float, float, float]]:
        """Return list of (cx, cy, radius) for n_chrom chromosomes
        arranged within the unit-square panel."""
        if n_chrom <= 1:
            return [(0.0, 0.0, 1.0)]
        if n_chrom == 2:
            return [(-0.55, 0.0, 0.55), (0.55, 0.0, 0.55)]
        # 3 or 4 → 2×2 grid
        r = 0.5
        return [
            (-0.6,  0.65, r), (0.6,  0.65, r),
            (-0.6, -0.65, r), (0.6, -0.65, r),
        ][:max(n_chrom, 4)]

    def update(frame_idx: int):
        artists = []
        # Move the vertical time bar on every bottom strip; clamp to
        # the strip's own gen-end so it parks at the right edge once
        # the condition has divided.
        global_t = float(frame_times[frame_idx])
        for j, (name, *_) in enumerate(conditions):
            bar_t = min(global_t, cond_t_end.get(name, global_t))
            time_bars[j].set_xdata([bar_t, bar_t])
            artists.append(time_bars[j])
        for i, (name, db, cache, color, tau) in enumerate(conditions):
            ax = theta_axes[i]
            ax.clear()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.85, 1.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{name} (τ={tau:.0f} min)", color=color, fontsize=11)

            if name not in frame_data:
                continue
            frame = frame_data[name][frame_idx]
            n_chrom = max(frame["n_chrom"], 1)

            chrom_positions = chromosome_layout(n_chrom)
            # Draw each chromosome circle + markers
            for cx, cy, r in chrom_positions:
                circle = MplCircle((cx, cy), r, fill=False, edgecolor="#374151",
                                   linewidth=1.4)
                ax.add_patch(circle)
                # oriC marker (top)
                ax.scatter([cx], [cy + r], s=55, color="#16a34a",
                           edgecolor="black", linewidths=0.7, zorder=4)
                # terminus marker (bottom)
                ax.scatter([cx], [cy - r], s=30, color="#1f2937",
                           edgecolor="black", linewidths=0.5, zorder=4)

            # Pair sister forks by domain index. Each domain = one
            # in-progress replication round. The biology: in multi-fork
            # cycles, rounds are nested on the parent chromosome and
            # SURVIVE the split — each daughter inherits the still-active
            # rounds. So we draw every active round on every chromosome
            # circle to convey ongoing activity across all chromosomes.
            forks_by_domain: dict[int, list[float]] = {}
            for coord, dom in zip(frame["fork_coords"], frame["fork_domains"]):
                forks_by_domain.setdefault(int(dom), []).append(float(coord))
            BUBBLE_COLORS = ["#fdba74", "#86efac", "#93c5fd", "#fda4af"]

            # Map each unique domain to a SPECIFIC chromosome circle. v2ecoli
            # doesn't expose the hierarchical domain → chromosome lineage tree
            # in the listener, so we approximate: assign domains in sort order
            # to chromosomes round-robin. When there are more rounds than
            # chromosomes (multi-fork on a single chromosome — e.g. with_aa
            # born with 3 rounds before any split), multiple domains share
            # the same chromosome and nest at progressively inner radii.
            sorted_rounds = sorted(forks_by_domain.items())
            chrom_to_rounds: dict[int, list[tuple[int, list[float]]]] = {
                i: [] for i in range(len(chrom_positions))
            }
            for j, (dom, forks) in enumerate(sorted_rounds):
                chrom_to_rounds[j % len(chrom_positions)].append((dom, forks))

            fork_size = max(50, 75 - 12 * (n_chrom - 1))
            for c_idx, (cx, cy, r) in enumerate(chrom_positions):
                rounds_here = chrom_to_rounds.get(c_idx, [])
                # Within a chromosome, nest each round at decreasing radius.
                # Color is tied to DOMAIN ID (round identity) — so the green
                # arc that was round 1 stays green throughout, even after the
                # chromosome splits and round 1 ends up alone on a daughter.
                for nest_i, (dom, forks) in enumerate(rounds_here):
                    inner_r = max(r * (0.85 - 0.15 * nest_i), 0.3 * r)
                    col = BUBBLE_COLORS[int(dom) % len(BUBBLE_COLORS)]
                    for coord in forks:
                        frac = min(abs(coord) / GENOME_HALF_NT, 1.0)
                        if coord >= 0:
                            fork_ang = 90.0 - 180.0 * frac
                            theta1, theta2 = fork_ang, 90.0
                        else:
                            fork_ang = 90.0 + 180.0 * frac
                            theta1, theta2 = 90.0, fork_ang
                        arc = MplArc((cx, cy), 2 * inner_r, 2 * inner_r,
                                     theta1=theta1, theta2=theta2,
                                     color=col, lw=3.0, alpha=0.95, zorder=2)
                        ax.add_patch(arc)
                        # Fork dot at end of this arc (the fork tip)
                        fx, fy = fork_xy(coord)
                        ax.scatter([cx + inner_r * fx], [cy + inner_r * fy],
                                   s=fork_size, color="#dc2626",
                                   edgecolor="black", linewidths=0.6, zorder=5)

            # Counters at the bottom — two lines so long times + counts
            # don't overflow the subplot width. When this condition has
            # already divided (its gen ended before global time), append
            # a "divided" marker so the viewer knows the frozen panel is
            # post-division.
            t_label = f"t = {frame['t']:5.1f} min"
            if frame["ended"]:
                t_label += "  (divided)"
            ax.text(0, -1.45, t_label,
                    ha="center", fontsize=9, color="#374151", family="monospace")
            ax.text(0, -1.72,
                    f"chroms {frame['n_chrom']}  ·  "
                    f"oriC {frame['n_oric']}  ·  "
                    f"forks {frame['n_fork']}",
                    ha="center", fontsize=8.5, color="#374151", family="monospace")

        return artists

    print(f"  rendering {n_frames}-frame animation → {out_gif} ...")
    t0 = time.time()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    ani.save(out_gif, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  done in {time.time() - t0:.1f}s")
    return out_gif


def panel_bulk_metrics_grid(per_cond, conditions=None):
    """Cross-condition overlays of bulk metrics: cell mass, total DnaA,
    total DnaA boxes, occupied DnaA boxes. ``conditions`` defaults to
    ``CONDITIONS_5``; pass a custom list for an alternate overlay (e.g.
    canonical-vs-Stage-1 acetate)."""
    if conditions is None:
        conditions = CONDITIONS_5
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    metrics = [
        (lambda d: d["dry_mass"], "Dry mass (fg)", "Cell mass"),
        (lambda d: d["dnaa_monomer"] + d["dnaa_atp"] + d["dnaa_adp"],
         "Total DnaA", "DnaA pool"),
        (lambda d: d["total_boxes"],
         "Total DnaA boxes", "DnaA boxes (chromosome-wide)"),
        (lambda d: d["total_boxes"] - d["free_boxes"],
         "Occupied DnaA boxes", "DnaA-box occupancy"),
    ]
    legend_hl = None
    for i, (get_y, ylabel, title) in enumerate(metrics):
        ax = axes[i]
        for name, db, cache, color, tau in conditions:
            d = per_cond.get(name)
            if d is None:
                continue
            ax.plot(d["t"] / 60, get_y(d), color=color, lw=1.6,
                    label=f"{name} (τ={tau:.0f} min)")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if legend_hl is None:
            legend_hl = ax.get_legend_handles_labels()
    handles, labels = legend_hl
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=len(conditions),
               fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig_to_b64(fig)


def panel_replication_dynamics_grid(per_cond):
    """One subplot per condition. Shows oriC, forks, and full chromosomes
    on shared axes per panel, with ▲ markers at initiation events (oriC
    count steps up) and ▼ at termination events (chromosome count
    steps up). Multi-fork is visible as oriC staying ≥4 with multiple
    chromosome-count step-ups in one generation."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
    # Compute GLOBAL ymax across all conditions so sharey doesn't hide
    # the larger-y conditions (with_aa has forks=6 vs acetate's 2)
    global_ymax = 1
    for name in [n for n, *_ in CONDITIONS_5]:
        d = per_cond.get(name)
        if d is None:
            continue
        global_ymax = max(global_ymax,
                          int(d["n_fork"].max()),
                          int(d["n_oric"].max()),
                          int(d["n_chrom"].max()))
    global_ymax += 1

    for i, (name, db, cache, color, tau) in enumerate(CONDITIONS_5):
        ax = axes[i]
        d = per_cond.get(name)
        if d is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            continue
        t_min = d["t"] / 60
        # Use step plots so the count transitions are visible
        ax.step(t_min, d["n_oric"],  where="post", color="#2ca02c", lw=2, label="oriC")
        ax.step(t_min, d["n_fork"],  where="post", color="#d62728", lw=2, label="forks")
        ax.step(t_min, d["n_chrom"], where="post", color="#1f77b4", lw=2, label="chromosomes")

        # Initiation events: oriC count steps up
        d_oric = np.diff(d["n_oric"])
        init_idx = np.where(d_oric > 0)[0] + 1
        # Termination events: chromosome count steps up
        d_chrom = np.diff(d["n_chrom"])
        term_idx = np.where(d_chrom > 0)[0] + 1

        # Vertical dashed lines + markers — use the GLOBAL ymax so marker
        # positions are consistent across all subplots
        ymax = global_ymax
        for idx in init_idx:
            ax.axvline(t_min[idx], color="#2ca02c", ls=":", lw=1, alpha=0.7)
            ax.scatter(t_min[idx], ymax * 0.92, marker="^", s=90,
                       color="#2ca02c", zorder=5, edgecolor="black", linewidths=0.6)
        for idx in term_idx:
            ax.axvline(t_min[idx], color="#1f77b4", ls=":", lw=1, alpha=0.5)
            ax.scatter(t_min[idx], ymax * 0.05, marker="v", s=80,
                       color="#1f77b4", zorder=5, edgecolor="black", linewidths=0.6)

        # Birth-state callout if cell starts multi-fork
        if int(d["n_oric"][0]) > 1:
            ax.text(0.02, 0.97, f"born with oriC={int(d['n_oric'][0])}, forks={int(d['n_fork'][0])}\n(inherited from mother)",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed",
                              edgecolor="#c2410c", linewidth=0.8),
                    color="#7c2d12")

        ax.set_title(f"{name} (τ={tau:.0f} min)", fontsize=11, color=color)
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylim(bottom=0, top=ymax)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Count", fontsize=10)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    # Add legend entries for event markers
    from matplotlib.lines import Line2D
    handles.extend([
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=10, label="initiation"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=9, label="termination"),
    ])
    labels.extend(["initiation", "termination"])
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig_to_b64(fig)


def panel_multi_dnaa(per_cond):
    def total_dnaa(d):
        return d["dnaa_monomer"] + d["dnaa_atp"] + d["dnaa_adp"]
    return _plot_multi(per_cond, total_dnaa, "Total DnaA (monomer + ATP + ADP)",
                       "DnaA pool dynamics across conditions")


def panel_multi_boxes(per_cond):
    def occupied(d):
        return d["total_boxes"] - d["free_boxes"]
    return _plot_multi(per_cond, occupied, "Occupied DnaA boxes (total − free)",
                       "DnaA-box occupancy across conditions")


STAGE1_PDF_PARAMS = [
    # (group, label, pdf_value, canonical_value, stage1_value, status_code, note)
    # status_code: "applied" | "default" | "no_consumer"
    ("Cell-cycle", "Cell birth volume", "1 µm³",
     "~0.43 µm³ (basal) / ~0.95 µm³ (acetate)", "same as canonical",
     "default",
     "Derived by ParCa from condition-specific dry-mass × ratio. Not safe to inject post-ParCa because the mass fit drives downstream ParCa stages."),
    ("Cell-cycle", "Doubling time τ", "150 min (ABT-glycerol)",
     "136 min (acetate)", "136 min (acetate)",
     "default",
     "PDF growth condition is ABT-glycerol; we run acetate (closest match) at v2ecoli's canonical 136 min. Doubling time drives most ParCa fits — not safe to override post-ParCa."),
    ("Cell-cycle", "C period", "70 min",
     "40 min", "70 min",
     "applied",
     "Post-ParCa knob: <code>--c-period-minutes 70</code> sets <code>basal_elongation_rate = replichore_length / (C × 60) ≈ 553 nt/s</code> in the chromosome_replication config."),
    ("Cell-cycle", "D period", "30 min",
     "20 min", "30 min",
     "applied",
     "Post-ParCa knob: <code>--d-period-minutes 30</code> replaces the D_period entry in the chromosome_replication config. Bypasses ParCa Step 6's global ECOS solver (which was infeasible under raw_data C+D=100)."),
    ("Cell-cycle", "Replication speed", "66.31 kbp/min",
     "115.9 kbp/min", "66.31 kbp/min",
     "applied",
     "Same override as C period (replisome elongation rate)."),
    ("DnaA expression", "dnaA transcription rate (constitutive)", "1.5 mRNA/min/gene",
     "fit per-condition by ParCa Step 6", "scaled × 15 (≈ PDF 1.5/min)",
     "applied",
     "Post-ParCa knob: <code>--dnaa-txn-scale 15</code> multiplies <code>basal_prob</code> for the dnaA TU. Pairs with <code>--dnaa-constitutive</code> to zero the TF regulatory row (TU's row in <code>delta_prob_matrix</code>)."),
    ("DnaA expression", "DnaA translation efficiency", "1.0 protein/mRNA (Hansen & Atlung 2018)",
     "0.35", "1.0",
     "applied",
     "Post-ParCa knob: <code>--dnaa-translation-efficiency 1.0</code> sets the DnaA monomer's per-monomer TE before the normalize() in get_polypeptide_initiation_config."),
    ("DnaA expression", "DnaA degradation rate", "0 (fully stable)",
     "decay via canonical half-life", "0 (fully stable)",
     "applied",
     "Post-ParCa knob: <code>--dnaa-stable</code> zeros <code>raw_degradation_rate</code> for the DnaA monomer in protein_degradation config."),
    ("DnaA biochemistry", "DnaA → ATP Kd", "30 nM (Sekimizu 1987)",
     "—", "—",
     "no_consumer",
     "No DnaA-ATP/ADP exchange process exists in v2ecoli yet (needs DnaA cycle build)."),
    ("DnaA biochemistry", "DnaA → ADP Kd", "100 nM (Kawakami 2006)",
     "—", "—",
     "no_consumer",
     "Same: no exchange process."),
    ("DnaA biochemistry", "DnaA-ATP intrinsic hydrolysis", "0.046 / min",
     "—", "—",
     "no_consumer",
     "No intrinsic-hydrolysis step exists yet."),
    ("DnaA boxes", "consensus DnaA-box locations", "raw TSV (Sheet \"DnaA boxes\")",
     "scanned from sequence motifs.tsv", "scanned from sequence motifs.tsv",
     "default",
     "Boxes are scanned for coordinates (~300+ sites) but only the LOCATIONS are used. No box-Kd model consumes them."),
    ("DnaA boxes", "consensus DnaA-box Kd", "1 nM",
     "—", "—",
     "no_consumer",
     "No DnaA-box equilibrium step yet."),
    ("DnaA boxes", "oriC location", "3925744–3925989",
     "as in genome annotation", "as in genome annotation",
     "default",
     "Already present in raw genome data; not overridden."),
    ("DnaA boxes", "oriC low-affinity DnaA-ATP box count", "8 (cooperative)",
     "—", "—",
     "no_consumer",
     "No oriC-loading step; current initiation gate is mass-driven."),
    ("DnaA boxes", "oriC low-affinity box Kd", "50 → 1 nM (cooperative)",
     "—", "—",
     "no_consumer",
     "Same: no oriC-loading step."),
    ("RIDA / DDAH / DARS", "Hda hydrolysis rate (per fork pair)", "40 / min",
     "—", "—",
     "no_consumer",
     "No RIDA process."),
    ("RIDA / DDAH / DARS", "datA locus", "4392732–4392914",
     "absent", "added (extragenic-site)",
     "applied",
     "DATA row added to dna_sites.tsv. Dormant data — no DDAH process consumes it yet."),
    ("RIDA / DDAH / DARS", "DnaA-ATP hydrolysis rate at datA", "12 / min / locus",
     "—", "—",
     "no_consumer",
     "No DDAH process."),
    ("RIDA / DDAH / DARS", "DARS1 location", "813086–813186",
     "813107–813141 (narrower)", "813086–813186 (widened)",
     "applied",
     "Widened to PDF spec on dna_sites.tsv. Dormant data — no DARS conversion process consumes it yet."),
    ("RIDA / DDAH / DARS", "DARS1 ADP→ATP conversion", "5 / min / locus",
     "—", "—",
     "no_consumer",
     "No DARS conversion process."),
    ("RIDA / DDAH / DARS", "DARS2 location", "2969112–2969367",
     "2969135–2969169 (narrower)", "2969112–2969367 (widened)",
     "applied",
     "Widened to PDF spec on dna_sites.tsv. Dormant data — no DARS conversion process consumes it yet."),
    ("RIDA / DDAH / DARS", "DARS2 ADP→ATP conversion", "10 / min / locus",
     "—", "—",
     "no_consumer",
     "No DARS conversion process."),
    ("SeqA", "SeqA sequestration time", "0 (not considered)",
     "—", "—",
     "no_consumer",
     "Stage 1 explicitly defers SeqA. No SeqA sequestration process in v2ecoli yet."),
]


def stage1_pdf_comparison_table_html() -> str:
    """Per-parameter comparison: PDF spec vs v2ecoli canonical default vs Stage 1
    cache value (after overrides), with status pill on every row."""
    PILL = {
        "applied":     '<span class="pill match">override applied</span>',
        "default":     '<span class="pill differs">v2ecoli default</span>',
        "no_consumer": '<span class="pill missing">no consumer yet</span>',
    }
    rows = []
    last_group = None
    for grp, label, pdf, canon, stage1, status, note in STAGE1_PDF_PARAMS:
        if grp != last_group:
            rows.append(
                f'<tr style="background:#eef2ff;">'
                f'<td colspan="5" style="font-weight:600;color:#3730a3;'
                f'font-size:11px;letter-spacing:0.4px;'
                f'text-transform:uppercase;">{grp}</td></tr>'
            )
            last_group = grp
        rows.append(
            f'<tr>'
            f'<td><strong>{label}</strong><div style="font-size:11px;color:var(--muted);margin-top:3px;">{note}</div></td>'
            f'<td style="background:#fff7ed; color:#c2410c;">{pdf}</td>'
            f'<td>{canon}</td>'
            f'<td style="background:#ecfdf5; color:#065f46;">{stage1}</td>'
            f'<td>{PILL[status]}</td>'
            f'</tr>'
        )
    return f"""
<table>
  <thead><tr>
    <th>Parameter (Stage 1 brief)</th>
    <th style="background:#fff7ed; color:#c2410c;">PDF value</th>
    <th>v2ecoli canonical default</th>
    <th style="background:#ecfdf5; color:#065f46;">Stage 1 cache value</th>
    <th>Status</th>
  </tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
"""


def multicondition_param_table_html(summary: list[dict]) -> str:
    """Build the cross-condition parameter table comparing observed vs Stage 1 brief."""

    def pill(s):
        return f'<span class="pill match">{s}</span>'

    rows_in_order = [name for name, _db, _c, _col, _t in CONDITIONS_5]
    summary_by_name = {s["name"]: s for s in summary}

    # Column header
    th = '<th>Parameter</th>' + ''.join(
        f'<th style="color: {summary_by_name[n]["color"]};">{n}</th>'
        for n in rows_in_order if n in summary_by_name
    ) + '<th style="background:#fff7ed; color:#c2410c;">Stage 1 brief</th>'

    def trow(label, values, stage1):
        cells = ''.join(f'<td>{v}</td>' for v in values)
        return (f'<tr><td><strong>{label}</strong></td>{cells}'
                f'<td style="background:#fff7ed; color:#c2410c;">{stage1}</td></tr>')

    # Per-parameter rows. The "Stage 1 brief" cell is populated ONLY when
    # the PDF explicitly states a value for that row; everything else is
    # left blank so the table never overstates what the PDF prescribes.
    rows_html = []
    # τ target (from condition_defs) — PDF: 150 min (ABT minimal glycerol)
    rows_html.append(trow("Doubling time τ (min)",
                          [f'{summary_by_name[n]["tau_target_min"]:.0f}' for n in rows_in_order if n in summary_by_name],
                          "150"))
    # τ observed in v2ecoli sim — not in PDF
    rows_html.append(trow("τ observed — v2ecoli (min)",
                          [f'{summary_by_name[n]["tau_observed_min"]:.0f}' for n in rows_in_order if n in summary_by_name],
                          ""))
    # Initial dry mass — not in PDF (PDF has cell BIRTH VOLUME = 1 µm³,
    # which is a different quantity)
    rows_html.append(trow("Initial dry mass (fg)",
                          [f'{summary_by_name[n]["initial_dry_mass"]:.0f}' for n in rows_in_order if n in summary_by_name],
                          ""))
    # Mass at division — not in PDF
    rows_html.append(trow("Mass at division (fg)",
                          [f'{summary_by_name[n]["final_dry_mass"]:.0f}' for n in rows_in_order if n in summary_by_name],
                          ""))
    # Mass fold change — not in PDF
    rows_html.append(trow("Mass fold-change",
                          [f'{summary_by_name[n]["mass_fold_change"]:.2f}×' for n in rows_in_order if n in summary_by_name],
                          ""))
    # Newborn state — not in PDF
    rows_html.append(trow("Newborn oriC count (t=0)",
                          [str(summary_by_name[n]["initial_oric"]) for n in rows_in_order if n in summary_by_name],
                          ""))
    rows_html.append(trow("Newborn active forks (t=0)",
                          [str(summary_by_name[n]["initial_forks"]) for n in rows_in_order if n in summary_by_name],
                          ""))
    # In-cycle maxima — not in PDF
    rows_html.append(trow("max oriC count in cycle",
                          [str(summary_by_name[n]["max_oric"]) for n in rows_in_order if n in summary_by_name],
                          ""))
    rows_html.append(trow("max forks in cycle",
                          [str(summary_by_name[n]["max_forks"]) for n in rows_in_order if n in summary_by_name],
                          ""))
    # Multi-fork — not directly in PDF (PDF only states "non-overlapping
    # cell cycle, which avoids synchronization issue" as the rationale
    # for choosing ABT-glycerol, not a generic prescription); leave blank.
    rows_html.append(trow("Multi-fork? (Td &lt; C)",
                          ["<strong style='color:#c2410c'>yes</strong>" if summary_by_name[n]["multi_fork"] else "no"
                           for n in rows_in_order if n in summary_by_name],
                          ""))
    # C period — PDF: 70 min
    rows_html.append(trow("C period (min) — global scalar",
                          ["40"] * sum(1 for n in rows_in_order if n in summary_by_name),
                          "70"))
    # D period — PDF: 30 min
    rows_html.append(trow("D period (min) — global scalar",
                          ["20"] * sum(1 for n in rows_in_order if n in summary_by_name),
                          "30"))
    # C + D — derived, not stated directly in PDF
    rows_html.append(trow("C + D (min)",
                          ["60"] * sum(1 for n in rows_in_order if n in summary_by_name),
                          ""))

    return f"""
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
"""


def write_html(panels: dict, gens_summary: list[dict], out_path: str,
               multi_panels: list = None, multi_table_html: str = "",
               stage1_table_html: str = "",
               stage1_panels: list = None,
               study1_table_html: str = "",
               study1_panels: list = None) -> None:
    """Render the report HTML using the same style as docs/stage1_parameter_comparison.html."""
    css = """
    :root { --fg: #212121; --muted: #607d8b; --border: #cfd8dc; --hi: #0d47a1;
            --match: #2e7d32; --match-bg: #e8f5e9; --differs: #ef6c00; --differs-bg: #fff3e0;
            --missing: #c62828; --missing-bg: #ffebee; --na: #455a64; --na-bg: #eceff1; }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           color: var(--fg); background: #fafafa; margin: 0; padding: 24px 36px;
           line-height: 1.45; max-width: 1300px; margin-left: auto; margin-right: auto; }
    h1 { margin: 0 0 4px 0; font-size: 24px; }
    h2 { margin-top: 32px; font-size: 18px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    h3 { margin-top: 18px; font-size: 14px; color: var(--hi); }
    .meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
    .meta code { background: #eceff1; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
    .panel { background: white; border: 1px solid var(--border); border-radius: 6px;
             padding: 14px 18px; margin-bottom: 14px; }
    .panel img { max-width: 100%; height: auto; display: block; margin: 8px auto; }
    table { width: 100%; border-collapse: collapse; background: white; font-size: 13px;
            border: 1px solid var(--border); }
    th, td { padding: 7px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
    th { background: #eceff1; font-weight: 600; font-size: 12px; text-transform: uppercase;
         letter-spacing: 0.4px; color: var(--muted); }
    .pill { display: inline-block; font-size: 11px; font-weight: 600;
            padding: 2px 8px; border-radius: 10px; text-transform: uppercase; letter-spacing: 0.3px; }
    .pill.match   { color: var(--match);   background: var(--match-bg); }
    .pill.differs { color: var(--differs); background: var(--differs-bg); }
    .pill.missing { color: var(--missing); background: var(--missing-bg); }
    .pill.na      { color: var(--na);      background: var(--na-bg); }
    .notes ul { padding-left: 20px; }
    .notes li { margin-bottom: 6px; font-size: 13px; }
    """

    # Per-gen summary table
    sum_rows = "".join(
        f"""<tr><td>{g["gen"]}</td><td>{g["dur_min"]:.1f}</td><td>{g["m0"]:.0f}</td>
        <td>{g["m1"]:.0f}</td><td>{g["mfold"]:.2f}×</td><td>{g["n_oric_max"]}</td>
        <td>{g["n_fork_max"]}</td></tr>"""
        for g in gens_summary
    )

    panel_blocks = "\n".join(
        f"""<div class="panel">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{b64}" alt="{title}" />
            <div style="font-size: 12px; color: var(--muted); margin-top: 6px;">{note}</div>
        </div>"""
        for title, b64, note in panels
    )

    def _render_multi_panel(title, content, note):
        # Animation panels pass content as "<animation:filename.gif>"
        # — inline as base64 data URL to side-step browser cache + path
        # resolution issues (Chrome was sometimes failing to autoplay
        # external GIFs).
        if isinstance(content, str) and content.startswith("<animation:"):
            import base64
            fname = content[len("<animation:"):-1]
            full_path = os.path.join(os.path.dirname(out_path) or ".", fname)
            with open(full_path, "rb") as f:
                gif_b64 = base64.b64encode(f.read()).decode("ascii")
            img_tag = f'<img src="data:image/gif;base64,{gif_b64}" alt="{title}" />'
        else:
            img_tag = f'<img src="data:image/png;base64,{content}" alt="{title}" />'
        return f"""<div class="panel">
            <h3>{title}</h3>
            {img_tag}
            <div style="font-size: 12px; color: var(--muted); margin-top: 6px;">{note}</div>
        </div>"""

    multi_panel_blocks = "\n".join(
        _render_multi_panel(title, content, note)
        for title, content, note in (multi_panels or [])
    )

    stage1_panel_blocks = "\n".join(
        _render_multi_panel(title, content, note)
        for title, content, note in (stage1_panels or [])
    )

    # Collapsible MIX0-845 media composition block (rendered inline in
    # the Stage 1 section to make the model-override framing explicit).
    mix0_845_recipe = mix0_845_recipe_html()

    study1_panel_blocks = "\n".join(
        _render_multi_panel(title, content, note)
        for title, content, note in (study1_panels or [])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 1 sanity-check diagnostics</title>
<style>{css}</style>
</head>
<body>

<h1>v2ecoli baseline behaviour vs Stage 1 brief</h1>
<div class="meta">
  Generated 2026-05-22 ·
  source: <code>Parameters for WCM (Stage 1: heuristic values) - Stage 1.pdf</code> ·
  ParCa fixture: canonical <code>models/parca/parca_state.pkl.gz</code> (chromosome-only, no plasmid, no Stage 1 overrides)
</div>

<h2>Pipeline architecture — how DNA-replication data flows</h2>
<p style="font-size: 13px; color: var(--muted); max-width: 950px;">
  Where every replication-related parameter lives in v2ecoli, and how it gets from raw TSV files through ParCa into the running simulation. Each labeled arrow shows specifically which raw input becomes which sim_data field and which ParCa step (or runtime process) consumes it. Stage 1's mechanistic parameters that have no consumer yet are shown in the bottom row (dashed grey).
</p>
<svg viewBox="0 0 1200 1170" xmlns="http://www.w3.org/2000/svg"
     style="width: 100%; max-width: 1200px; height: auto; display: block;
            margin: 1rem auto; border: 1px solid var(--border);
            border-radius: 6px; background: #fafafa;">
  <defs>
    <marker id="arrR" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#374151"/>
    </marker>
    <marker id="arrD" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#2563eb"/>
    </marker>
    <marker id="arrG" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#16a34a"/>
    </marker>
    <marker id="arrO" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#9a3412"/>
    </marker>
    <marker id="arrGrey" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#9ca3af"/>
    </marker>
  </defs>

  <!-- ============================== ROW 1: Raw data ============================== -->
  <text x="20" y="30" font-family="sans-serif" font-size="14"
        font-weight="700" fill="#111">1 · Raw flat files (canonical inputs at <tspan font-family="monospace" font-size="12">reconstruction/ecoli/flat/</tspan>)</text>

  <!-- 6 TSV boxes — aligned x-positions to match row 2 sub-boxes so
       arrows can be straight vertical -->
  <g font-family="sans-serif" font-size="10" fill="#111">
    <rect x="30"  y="50" width="180" height="80" fill="#fde68a" stroke="#92400e" rx="4"/>
    <text x="120" y="68" text-anchor="middle" font-weight="700" font-size="11">parameters.tsv</text>
    <text x="120" y="84" text-anchor="middle" font-family="monospace" font-size="9">d_period (20 min)</text>
    <text x="120" y="98" text-anchor="middle" font-family="monospace" font-size="9">replisome_elongation_rate</text>
    <text x="120" y="110" text-anchor="middle" font-family="monospace" font-size="9">(967 nt/s)</text>
    <text x="120" y="124" text-anchor="middle" font-size="9" fill="#92400e">global scalars</text>

    <rect x="225" y="50" width="180" height="80" fill="#bfdbfe" stroke="#1e40af" rx="4"/>
    <text x="315" y="68" text-anchor="middle" font-weight="700" font-size="11">sequence.fasta</text>
    <text x="315" y="84" text-anchor="middle" font-family="monospace" font-size="9">genome length</text>
    <text x="315" y="98" text-anchor="middle" font-family="monospace" font-size="9">4,641,652 nt</text>
    <text x="315" y="114" text-anchor="middle" font-family="monospace" font-size="9">U00096.3 (K-12)</text>

    <rect x="420" y="50" width="180" height="80" fill="#fbcfe8" stroke="#9d174d" rx="4"/>
    <text x="510" y="68" text-anchor="middle" font-weight="700" font-size="11">dna_sites.tsv</text>
    <text x="510" y="84" text-anchor="middle" font-family="monospace" font-size="9">oriC: 3925744–3925975</text>
    <text x="510" y="98" text-anchor="middle" font-family="monospace" font-size="9">DARS1: 813107–813141</text>
    <text x="510" y="112" text-anchor="middle" font-family="monospace" font-size="9">DARS2: 2969135–2969169</text>
    <text x="510" y="124" text-anchor="middle" font-size="9" fill="#9d174d">datA: NOT annotated</text>

    <rect x="615" y="50" width="180" height="80" fill="#d9f99d" stroke="#365314" rx="4"/>
    <text x="705" y="68" text-anchor="middle" font-weight="700" font-size="11">sequence_motifs.tsv</text>
    <text x="705" y="84" text-anchor="middle" font-family="monospace" font-size="9">8 DnaA-box motif variants</text>
    <text x="705" y="98" text-anchor="middle" font-family="monospace" font-size="9">→ 307 hits via genome scan</text>
    <text x="705" y="114" text-anchor="middle" font-size="9" fill="#365314">(Schaper-Messer 1995)</text>

    <rect x="810" y="50" width="180" height="80" fill="#fed7aa" stroke="#9a3412" rx="4"/>
    <text x="900" y="68" text-anchor="middle" font-weight="700" font-size="11">genes.tsv</text>
    <text x="900" y="84" text-anchor="middle" font-family="monospace" font-size="9">~4309 gene coords</text>
    <text x="900" y="98" text-anchor="middle" font-family="monospace" font-size="9">→ replication_coordinate</text>
    <text x="900" y="114" text-anchor="middle" font-size="9" fill="#9a3412">(fork → gene dosage)</text>

    <rect x="1005" y="50" width="170" height="80" fill="#e9d5ff" stroke="#581c87" rx="4"/>
    <text x="1090" y="68" text-anchor="middle" font-weight="700" font-size="11">equilibrium_*.tsv</text>
    <text x="1090" y="84" text-anchor="middle" font-family="monospace" font-size="9">MONOMER0-160_RXN</text>
    <text x="1090" y="98" text-anchor="middle" font-family="monospace" font-size="9">→ DnaA-ATP Kd = 29 nM</text>
    <text x="1090" y="114" text-anchor="middle" font-size="9" fill="#581c87">MONOMER0-4565: no rate</text>
  </g>

  <!-- Straight vertical arrows from row 1 (bottom at y=130) to row 2 sub-boxes (top at y=222).
       Endpoints pull back to y=216 so the arrowheads sit clearly above the sub-box tops. -->
  <path d="M 120  130 L 120  216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <path d="M 315  130 L 315  216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <path d="M 510  130 L 510  216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <path d="M 705  130 L 705  216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <path d="M 900  130 L 900  216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <path d="M 1090 130 L 1090 216" stroke="#374151" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>

  <!-- ============================== ROW 2: ParCa Step 1 with 6 sub-loaders ============================== -->
  <!-- Title sits BELOW the row-1→row-2 arrow band, just above the
       Step 1 container. Single line — no separate subtitle to clash. -->
  <text x="20" y="202" font-family="sans-serif" font-size="13"
        font-weight="700" fill="#1e40af">2 · ParCa Step 1 — <tspan font-family="monospace" font-size="11">sim_data.initialize(raw_data)</tspan> <tspan font-family="sans-serif" font-size="10" font-weight="400">— each file → its loader → a specific sim_data field</tspan></text>

  <!-- Visual frame for Step 1 — transparent fill so the row-1→row-2
       arrowheads (which end at y=220, inside the frame) are not hidden -->
  <rect x="15" y="208" width="1170" height="240" fill="none" stroke="#2563eb" stroke-width="1" rx="6"/>

  <!-- 6 internal sub-boxes — one per raw input -->
  <g font-family="sans-serif" font-size="10" fill="#111">
    <!-- (1) parameters.tsv -->
    <rect x="30" y="222" width="180" height="100" fill="#fde68a" stroke="#92400e" rx="3"/>
    <text x="120" y="240" text-anchor="middle" font-weight="700" font-size="10">_load_parameters()</text>
    <text x="120" y="256" text-anchor="middle" font-family="monospace" font-size="8">parameters.tsv</text>
    <text x="120" y="270" text-anchor="middle" font-family="monospace" font-size="8">→ raw.parameters dict</text>
    <text x="40" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="40" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.constants.d_period</text>
    <text x="40" y="310" font-family="monospace" font-size="8" fill="#1e40af">  .replisome_elongation_rate</text>

    <!-- (2) sequence.fasta -->
    <rect x="225" y="222" width="180" height="100" fill="#bfdbfe" stroke="#1e40af" rx="3"/>
    <text x="315" y="240" text-anchor="middle" font-weight="700" font-size="10">_load_sequence()</text>
    <text x="315" y="256" text-anchor="middle" font-family="monospace" font-size="8">sequence.fasta</text>
    <text x="315" y="270" text-anchor="middle" font-family="monospace" font-size="8">→ raw.genome_sequence</text>
    <text x="235" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="235" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.constants.c_period</text>
    <text x="235" y="311" font-family="monospace" font-size="8">    = len(seq)/rate/2 (derived)</text>

    <!-- (3) dna_sites.tsv -->
    <rect x="420" y="222" width="180" height="100" fill="#fbcfe8" stroke="#9d174d" rx="3"/>
    <text x="510" y="240" text-anchor="middle" font-weight="700" font-size="10">_build_genomic_</text>
    <text x="510" y="253" text-anchor="middle" font-weight="700" font-size="10">coordinates()</text>
    <text x="510" y="269" text-anchor="middle" font-family="monospace" font-size="8">dna_sites.tsv</text>
    <text x="430" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="430" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.process.replication</text>
    <text x="430" y="311" font-family="monospace" font-size="8" fill="#1e40af">    ._all_genomic_coordinates</text>

    <!-- (4) sequence_motifs.tsv -->
    <rect x="615" y="222" width="180" height="100" fill="#d9f99d" stroke="#365314" rx="3"/>
    <text x="705" y="240" text-anchor="middle" font-weight="700" font-size="10">_get_motif_</text>
    <text x="705" y="253" text-anchor="middle" font-weight="700" font-size="10">coordinates()</text>
    <text x="705" y="269" text-anchor="middle" font-family="monospace" font-size="8">sequence_motifs.tsv</text>
    <text x="625" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="625" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.process.replication</text>
    <text x="625" y="311" font-family="monospace" font-size="8" fill="#1e40af">    .motif_coordinates['DnaA_box']</text>

    <!-- (5) genes.tsv -->
    <rect x="810" y="222" width="180" height="100" fill="#fed7aa" stroke="#9a3412" rx="3"/>
    <text x="900" y="240" text-anchor="middle" font-weight="700" font-size="10">replication.py</text>
    <text x="900" y="253" text-anchor="middle" font-weight="700" font-size="10">  __init__()</text>
    <text x="900" y="269" text-anchor="middle" font-family="monospace" font-size="8">genes.tsv + rnas.tsv</text>
    <text x="820" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="820" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.process.replication</text>
    <text x="820" y="311" font-family="monospace" font-size="8" fill="#1e40af">    .gene_data + .replichore</text>

    <!-- (6) equilibrium_*.tsv -->
    <rect x="1005" y="222" width="170" height="100" fill="#e9d5ff" stroke="#581c87" rx="3"/>
    <text x="1090" y="240" text-anchor="middle" font-weight="700" font-size="10">equilibrium.py</text>
    <text x="1090" y="253" text-anchor="middle" font-weight="700" font-size="10">  __init__()</text>
    <text x="1090" y="269" text-anchor="middle" font-family="monospace" font-size="8">equilibrium_*.tsv</text>
    <text x="1015" y="287" font-family="monospace" font-size="8">produces:</text>
    <text x="1015" y="299" font-family="monospace" font-size="8" fill="#1e40af">  sim_data.process.equilibrium</text>
    <text x="1015" y="311" font-family="monospace" font-size="8" fill="#1e40af">    .rxn_rates (DnaA-ATP Kd)</text>
  </g>

  <!-- Inside Step 1 — show internal dependency arrow: parameters + sequence both feed c_period -->
  <text x="30" y="345" font-family="sans-serif" font-size="10" fill="#1e40af" font-weight="700">Cross-dependency inside Step 1:</text>
  <text x="30" y="360" font-family="monospace" font-size="9" fill="#1e40af">  c_period = len(genome_seq) / replisome_elongation_rate / 2  →  links parameters.tsv ↔ sequence.fasta</text>

  <!-- Downstream Steps 4 + 6 (highlighted in Step 1's container's bottom area) -->
  <text x="30" y="385" font-family="sans-serif" font-size="11" fill="#1e40af" font-weight="700">Downstream ParCa steps that re-consume these fields:</text>

  <rect x="30" y="395" width="540" height="45" fill="#dbeafe" stroke="#2563eb" rx="3"/>
  <text x="40" y="412" font-family="sans-serif" font-size="10" font-weight="700">Step 4 — TF condition specs</text>
  <text x="40" y="425" font-family="monospace" font-size="9" fill="#1e40af">reads sim_data.constants.{{c_period, d_period}}</text>
  <text x="40" y="436" font-family="monospace" font-size="9" fill="#1e40af">  → mass.getBiomassAsConcentrations(τ) → per-condition cell-mass fit</text>

  <rect x="585" y="395" width="600" height="45" fill="#dbeafe" stroke="#2563eb" rx="3"/>
  <text x="595" y="412" font-family="sans-serif" font-size="10" font-weight="700">Step 6 — Promoter binding</text>
  <text x="595" y="425" font-family="monospace" font-size="9" fill="#1e40af">reads gene_data.replication_coordinate + c_period + d_period</text>
  <text x="595" y="436" font-family="monospace" font-size="9" fill="#1e40af">  → Cooper-Helmstetter dosage:  2^((1−x)·C + D / τ)  →  rna_synth_prob[cond]</text>

  <!-- Arrow exit from Step 1 container down to Row 3 -->
  <path d="M 600 448 L 600 488" stroke="#2563eb" stroke-width="2" fill="none" marker-end="url(#arrD)"/>
  <text x="610" y="473" font-family="monospace" font-size="10" fill="#2563eb">all fields dilled into sim_data_cache.dill</text>

  <line x1="20" y1="466" x2="1180" y2="466" stroke="#d1d5db" stroke-dasharray="3,3"/>

  <!-- ============================== ROW 3: sim_data cache ============================== -->
  <text x="20" y="510" font-family="sans-serif" font-size="14"
        font-weight="700" fill="#16a34a">3 · sim_data cache — what's pickled and read by the runtime sim</text>
  <text x="20" y="528" font-family="sans-serif" font-size="10" fill="#16a34a">Stored at <tspan font-family="monospace" font-size="9">out/cache/sim_data_cache.dill</tspan> + per-condition initial state at <tspan font-family="monospace" font-size="9">initial_state.json</tspan></text>

  <g font-family="sans-serif" font-size="10" fill="#111">
    <rect x="20"  y="540" width="280" height="100" fill="#dcfce7" stroke="#16a34a" rx="4"/>
    <text x="160" y="558" text-anchor="middle" font-weight="700" font-size="11">sim_data.constants</text>
    <text x="30" y="576" font-family="monospace" font-size="9">  c_period = 40 min</text>
    <text x="30" y="589" font-family="monospace" font-size="9">  d_period = 20 min</text>
    <text x="30" y="602" font-family="monospace" font-size="9">  avg_cell_dry_mass_init = 105.8 fg</text>
    <text x="30" y="615" font-family="monospace" font-size="9">  (from parameters.tsv + sequence)</text>

    <rect x="310" y="540" width="300" height="100" fill="#dcfce7" stroke="#16a34a" rx="4"/>
    <text x="460" y="558" text-anchor="middle" font-weight="700" font-size="11">sim_data.process.replication</text>
    <text x="320" y="576" font-family="monospace" font-size="9">  .motif_coordinates['DnaA_box']</text>
    <text x="320" y="589" font-family="monospace" font-size="9">    = array of 307 box midpoints</text>
    <text x="320" y="602" font-family="monospace" font-size="9">  .oric_coordinate = 3925860</text>
    <text x="320" y="615" font-family="monospace" font-size="9">  .basal_elongation_rate = 967 nt/s</text>

    <rect x="620" y="540" width="220" height="100" fill="#dcfce7" stroke="#16a34a" rx="4"/>
    <text x="730" y="558" text-anchor="middle" font-weight="700" font-size="11">sim_data.condition</text>
    <text x="630" y="576" font-family="monospace" font-size="9">  = "acetate"</text>
    <text x="630" y="591" font-family="sans-serif" font-size="9">→ which RNA-expression vector</text>
    <text x="630" y="604" font-family="sans-serif" font-size="9">→ which initial cell mass</text>
    <text x="630" y="617" font-family="sans-serif" font-size="9">→ which TFs active (Cra at acetate)</text>

    <rect x="850" y="540" width="335" height="100" fill="#dcfce7" stroke="#16a34a" rx="4"/>
    <text x="1015" y="558" text-anchor="middle" font-weight="700" font-size="11">initial_state.json</text>
    <text x="860" y="576" font-family="monospace" font-size="9">  bulk array (16,323 species)</text>
    <text x="860" y="589" font-family="monospace" font-size="9">  unique molecules (1 oriC, 0 forks at t=0)</text>
    <text x="860" y="602" font-family="monospace" font-size="9">  dry_mass = 105.8 fg (acetate target)</text>
    <text x="860" y="615" font-family="sans-serif" font-size="9">→ seeds the cell at sim t=0</text>
  </g>

  <!-- Straight vertical arrows row 3 → row 4. Row 4 boxes resized below
       to share centers with row 3, so these are perfectly vertical.
       Endpoints pull back to y=704 to keep arrowheads visible above row 4 boxes. -->
  <path d="M 160  640 L 160  704" stroke="#16a34a" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>
  <path d="M 460  640 L 460  704" stroke="#16a34a" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>
  <path d="M 730  640 L 730  704" stroke="#16a34a" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>
  <path d="M 1017 640 L 1017 704" stroke="#16a34a" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <!-- inline labels — placed alongside each arrow body -->
  <text x="167" y="678" font-family="monospace" font-size="9" fill="#16a34a">d_period, c_period</text>
  <text x="467" y="678" font-family="monospace" font-size="9" fill="#16a34a">motif coords, elong rate</text>
  <text x="737" y="678" font-family="monospace" font-size="9" fill="#16a34a">τ + media</text>
  <text x="1023" y="678" font-family="monospace" font-size="9" fill="#16a34a">bulk init</text>

  <!-- ============================== ROW 4: Runtime sim ============================== -->
  <!-- Header just under row 3 box bottoms so arrows pass through the gap, not the text -->
  <text x="20" y="658" font-family="sans-serif" font-size="13"
        font-weight="700" fill="#9a3412">4 · Runtime sim — chromosome_replication.py</text>

  <g font-family="sans-serif" font-size="10" fill="#111">
    <!-- aligned with row 3 boxes for vertical arrows -->
    <rect x="20" y="710" width="280" height="130" fill="#fed7aa" stroke="#9a3412" rx="4"/>
    <text x="160" y="724" text-anchor="middle" font-weight="700" font-size="11">Initiation gate (line 230–246)</text>
    <text x="30" y="740" font-family="monospace" font-size="9">  massPerOrigin = cellMass / n_oriC</text>
    <text x="30" y="753" font-family="monospace" font-size="9">  criticalMass = get_dna_critical_mass(τ)</text>
    <text x="30" y="766" font-family="monospace" font-size="9">  if ratio &gt;= 1.0: initiate</text>
    <!-- compact hardcoded-heuristic callout (2 lines) -->
    <rect x="22" y="772" width="276" height="34" fill="#fff7ed" stroke="#c2410c" stroke-width="1" rx="3"/>
    <text x="30" y="785" font-family="sans-serif" font-size="9" font-weight="700" fill="#c2410c">⚠ critical_mass: HARDCODED, not from a TSV</text>
    <text x="30" y="800" font-family="monospace" font-size="8" fill="#c2410c">  975 fg cap × 1.2 slow-growth factor (gp_rate_dep.py:13)</text>
    <text x="30" y="822" font-family="sans-serif" font-size="10" font-weight="700" fill="#9a3412">⚠ MASS-DRIVEN, not DnaA-driven</text>
    <text x="30" y="836" font-family="sans-serif" font-size="9" fill="#9a3412">(DnaA in bulk is biochemically inert)</text>

    <rect x="320" y="710" width="280" height="130" fill="#fed7aa" stroke="#9a3412" rx="4"/>
    <text x="460" y="728" text-anchor="middle" font-weight="700" font-size="11">Fork elongation</text>
    <text x="330" y="746" font-family="monospace" font-size="9">  for each fork:</text>
    <text x="330" y="759" font-family="monospace" font-size="9">    advance by replisome_rate × Δt</text>
    <text x="330" y="773" font-family="monospace" font-size="9">    consume dNTPs from bulk</text>
    <text x="330" y="789" font-family="sans-serif" font-size="9">when fork meets terminus:</text>
    <text x="330" y="803" font-family="monospace" font-size="9">  add full_chromosome to unique</text>
    <text x="330" y="817" font-family="monospace" font-size="9">  division_time = t + d_period</text>

    <rect x="620" y="710" width="220" height="130" fill="#fed7aa" stroke="#9a3412" rx="4"/>
    <text x="730" y="728" text-anchor="middle" font-weight="700" font-size="11">Division gate</text>
    <text x="630" y="746" font-family="sans-serif" font-size="9">MarkDPeriod (division.py)</text>
    <text x="630" y="762" font-family="monospace" font-size="9">  if t &gt;= division_time</text>
    <text x="630" y="776" font-family="monospace" font-size="9">     AND chroms &gt;= 2</text>
    <text x="630" y="790" font-family="monospace" font-size="9">     AND dry_mass &gt;= thr:</text>
    <text x="630" y="804" font-family="monospace" font-size="9">    split state</text>
    <text x="630" y="817" font-family="monospace" font-size="9">    daughter starts t=0</text>
    <text x="630" y="831" font-family="sans-serif" font-size="9" fill="#9a3412">thr = mass_distr × init</text>

    <rect x="850" y="710" width="335" height="130" fill="#ffedd5" stroke="#c2410c" rx="4"/>
    <text x="1017" y="728" text-anchor="middle" font-weight="700" font-size="11">Listeners (SQLite emit history)</text>
    <text x="860" y="746" font-family="monospace" font-size="9">  replication_data_listener:</text>
    <text x="860" y="759" font-family="monospace" font-size="9">    number_of_oric</text>
    <text x="860" y="772" font-family="monospace" font-size="9">    fork_coordinates, fork_domains</text>
    <text x="860" y="785" font-family="monospace" font-size="9">    free_DnaA_boxes, total_DnaA_boxes</text>
    <text x="860" y="801" font-family="monospace" font-size="9">  mass_listener:</text>
    <text x="860" y="814" font-family="monospace" font-size="9">    cell_mass, dry_mass, dna_mass</text>
    <text x="860" y="829" font-family="sans-serif" font-size="9" fill="#c2410c">↑ everything in this report's plots</text>

    <!-- chromosome_structure.py — runs every tick alongside the other 4
         processes; handles structural consequences of fork passage -->
    <rect x="20" y="852" width="1165" height="80" fill="#fed7aa" stroke="#9a3412" rx="4"/>
    <text x="600" y="869" text-anchor="middle" font-weight="700" font-size="11">chromosome_structure.py — structural bookkeeper for replication-fork passage (every tick)</text>
    <text x="30" y="887" font-family="monospace" font-size="9">  for each unique motif (promoter, gene, DnaA_box, RNAP, ribosome):</text>
    <text x="30" y="900" font-family="monospace" font-size="9">    if fork passed → DUPLICATE motif (e.g. 307 DnaA boxes → ~614 mid-replication, new copies get DnaA_bound=False)</text>
    <text x="30" y="913" font-family="monospace" font-size="9">    if RNAP/ribosome collided with fork → REMOVE + recycle subunits to bulk + conserve mass</text>
    <text x="30" y="926" font-family="sans-serif" font-size="9" fill="#9a3412">⚠ DnaA boxes only track <tspan font-family="monospace" font-size="9">DnaA_bound: bool</tspan> — no Kd, no ATP/ADP state. Stage 1's cooperative binding hooks in here.</text>
  </g>

  <line x1="20" y1="945" x2="1180" y2="945" stroke="#d1d5db" stroke-dasharray="3,3"/>

  <!-- ============================== ROW 5: Stage 1 NOT implemented ============================== -->
  <text x="20" y="978" font-family="sans-serif" font-size="14"
        font-weight="700" fill="#6b7280">5 · Stage 1 mechanisms NOT in the pipeline (dashed = no consumer)</text>
  <text x="20" y="996" font-family="sans-serif" font-size="10" fill="#6b7280">These would replace the current mass-driven initiation gate and add the DnaA cycle</text>

  <g font-family="sans-serif" font-size="10" fill="#4b5563">
    <rect x="20" y="1013" width="220" height="90" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4,4" rx="4"/>
    <text x="130" y="1031" text-anchor="middle" font-weight="700" font-size="11" fill="#6b7280">DnaA-ATP → oriC binding</text>
    <text x="30" y="1048" font-size="9">  cooperative Kd ladder 50 → 1 nM</text>
    <text x="30" y="1061" font-size="9">  at 8 low-affinity oriC boxes</text>
    <text x="30" y="1076" font-size="9" font-style="italic">→ replaces the mass-driven initiation gate</text>
    <text x="30" y="1090" font-size="9" font-style="italic">  at chromosome_replication.py:230</text>

    <rect x="250" y="1013" width="220" height="90" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4,4" rx="4"/>
    <text x="360" y="1031" text-anchor="middle" font-weight="700" font-size="11" fill="#6b7280">DnaA-ATP hydrolysis</text>
    <text x="260" y="1048" font-size="9">  intrinsic: 0.046 /min</text>
    <text x="260" y="1061" font-size="9">  RIDA (Hda-coupled): 40 /min/fork</text>
    <text x="260" y="1076" font-size="9" font-style="italic">→ depletes DnaA-ATP after initiation</text>
    <text x="260" y="1090" font-size="9" font-style="italic">  prevents immediate re-initiation</text>

    <rect x="480" y="1013" width="220" height="90" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4,4" rx="4"/>
    <text x="590" y="1031" text-anchor="middle" font-weight="700" font-size="11" fill="#6b7280">DDAH (datA locus)</text>
    <text x="490" y="1048" font-size="9">  locus: 4392732–4392914</text>
    <text x="490" y="1061" font-size="9">  12 /min DnaA-ATP hydrolysis</text>
    <text x="490" y="1076" font-size="9" font-style="italic">→ post-initiation DnaA-ATP drain</text>
    <text x="490" y="1090" font-size="9" font-style="italic">  locus not yet in dna_sites.tsv</text>

    <rect x="710" y="1013" width="220" height="90" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4,4" rx="4"/>
    <text x="820" y="1031" text-anchor="middle" font-weight="700" font-size="11" fill="#6b7280">DARS1 / DARS2</text>
    <text x="720" y="1048" font-size="9">  loci annotated, no reaction</text>
    <text x="720" y="1061" font-size="9">  reactivation rates 5 + 10 /min</text>
    <text x="720" y="1076" font-size="9" font-style="italic">→ ADP → ATP refresh</text>
    <text x="720" y="1090" font-size="9" font-style="italic">  reads sim_data._all_genomic_coordinates</text>

    <rect x="940" y="1013" width="240" height="90" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4,4" rx="4"/>
    <text x="1060" y="1031" text-anchor="middle" font-weight="700" font-size="11" fill="#6b7280">SeqA sequestration</text>
    <text x="950" y="1048" font-size="9">  Stage 1 brief: 0 (deferred)</text>
    <text x="950" y="1061" font-size="9">  future refinement</text>
    <text x="950" y="1076" font-size="9" font-style="italic">→ post-initiation oriC lock-out</text>
    <text x="950" y="1090" font-size="9" font-style="italic">  prevents premature re-init</text>
  </g>

  <!-- arrows from Stage 1 boxes pointing UP into row 4 (where they'd plug in)
       Row 4 ends at y=932 now (with chromosome_structure bar), so arrows
       terminate at y=937 just below row 4 -->
  <path d="M 130 1013 L 130 937" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrGrey)"/>
  <text x="135" y="971" font-family="monospace" font-size="8" fill="#6b7280">→ initiation gate</text>

  <path d="M 360 1013 L 360 937" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrGrey)"/>
  <text x="365" y="971" font-family="monospace" font-size="8" fill="#6b7280">→ DnaA-ATP pool</text>

  <path d="M 590 1013 L 590 937" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrGrey)"/>
  <text x="595" y="971" font-family="monospace" font-size="8" fill="#6b7280">→ DnaA-ATP pool</text>

  <path d="M 820 1013 L 820 937" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrGrey)"/>
  <text x="825" y="971" font-family="monospace" font-size="8" fill="#6b7280">→ DnaA box state</text>

  <path d="M 1060 1013 L 1060 937" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrGrey)"/>
  <text x="1065" y="971" font-family="monospace" font-size="8" fill="#6b7280">→ initiation gate</text>

  <!-- bottom legend -->
  <text x="20" y="1143" font-family="sans-serif" font-size="11" fill="#6b7280" font-weight="700">Color key:</text>
  <rect x="100" y="1133" width="14" height="14" fill="#fde68a" stroke="#92400e"/>
  <text x="120" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">raw flat data</text>
  <rect x="225" y="1133" width="14" height="14" fill="#dbeafe" stroke="#2563eb"/>
  <text x="245" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">ParCa transform</text>
  <rect x="355" y="1133" width="14" height="14" fill="#dcfce7" stroke="#16a34a"/>
  <text x="375" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">cached sim_data</text>
  <rect x="490" y="1133" width="14" height="14" fill="#fed7aa" stroke="#9a3412"/>
  <text x="510" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">runtime sim</text>
  <rect x="595" y="1133" width="14" height="14" fill="#ffedd5" stroke="#c2410c"/>
  <text x="615" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">listener output</text>
  <rect x="710" y="1133" width="14" height="14" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="3,3"/>
  <text x="730" y="1144" font-family="sans-serif" font-size="11" fill="#6b7280">Stage 1 mechanism (no consumer yet)</text>
</svg>

<p style="font-size: 12px; color: var(--muted); max-width: 950px; margin-top: 0;">
  <strong>Reading the diagram, top to bottom:</strong> raw TSV files at row 1 are inputs to ParCa (row 2); ParCa transforms them into sim_data fields (row 3); the runtime simulation reads sim_data and listeners write outputs (row 4); the bottom row shows the Stage 1 DnaA-cycle mechanisms that have no plug-in point yet. The dashed grey arrows show where each missing mechanism would attach: DnaA-ATP binding and hydrolysis would replace the current mass-driven initiation gate; DDAH / DARS would attach to the DnaA-ATP pool dynamics.
</p>

<h2>Per-condition parameter comparison</h2>
<p style="font-size: 13px; color: var(--muted); max-width: 950px;">
  Per-condition observables extracted from the 5 baseline sims, with the Stage 1 brief target as the rightmost column for reference. C and D periods are global scalars in v2ecoli, so they're the same across all conditions.
</p>
{multi_table_html}

<h2>Cross-condition baseline behaviour</h2>
<p style="font-size: 13px; color: var(--muted); max-width: 950px;">
  Five canonical media conditions, no Stage 1 overrides applied, 1 generation each. Same chromosome-only baseline composite for all. Color codes: red (with_aa, fastest) → green (acetate, slowest).
</p>
{multi_panel_blocks}

<h2>Stage 1 PDF — model-override scenario (separate from canonical baseline)</h2>
<p style="font-size: 13px; color: var(--muted); max-width: 950px;">
  This section is <strong>not</strong> a sixth canonical condition. It is a Stage 1 PDF reproduction layered on top of v2ecoli through (a) a NEW media file we authored (<code>MIX0-845</code>, AB + 0.2 % glycerol) and (b) post-ParCa knobs applied at composite-build time. Compared against canonical <code>acetate</code> as the closest analog (both slow-growth conditions), not against the full canonical baseline above.
</p>

{mix0_845_recipe}

<p style="font-size: 13px; color: var(--muted); max-width: 950px; margin-top: 14px;">
  The table below maps every Stage 1 PDF parameter to (a) v2ecoli's canonical default and (b) the value present in the override cache. Pills mark each row as
  <span class="pill match">override applied</span> (PDF value plumbed into v2ecoli),
  <span class="pill differs">v2ecoli default</span> (override skipped — either intentionally or because no consumer / fit infeasibility prevents it), or
  <span class="pill missing">no consumer yet</span> (the PDF specifies a rate/Kd but no v2ecoli process currently reads it, so the parameter would be inert even if set).
</p>
{stage1_table_html}

<p style="font-size: 13px; color: var(--muted); max-width: 950px; margin-top: 18px;">
  The diagnostic plots below compare an acetate run against the <em>canonical</em> v2ecoli baseline (left panel) vs the same acetate run with the <strong>full Stage 1 PDF cache</strong> (right panel — all seven post-ParCa knobs applied). The Stage 1 sim runs multi-gen so later generations equilibrate.
</p>
{stage1_panel_blocks}

<div style="background:#ecfdf5; border-left:5px solid #16a34a; padding:14px 18px;
            margin: 28px 0 10px 0; border-radius:4px;">
  <strong style="color:#065f46; font-size:14px;">Moving forward:</strong>
  <span style="color:#065f46; font-size:13px;">
    we keep the <code>glycerol (Stage 1 PDF)</code> cache as the working baseline
    for the multi-phase DnaA workflow. It carries every Stage 1 PDF parameter
    that has a runtime consumer — C = 70 min (baked into ParCa), D = 30 min,
    dnaA basal_prob × 15 (≈ 1.5 events/min/gene), dnaA expression made constitutive
    (delta_prob_matrix row zeroed), DnaA monomer stable (degradation = 0), and
    DnaA translation efficiency = 1.0. M* is left at the ParCa-fitted value for
    τ = 150 min — we initially tried scaling it ×1.5 to push back re-initiation
    but realized re-INIT timing is set by τ_mass, not by M*, so the scale would
    only shift events in absolute time without closing the gap between them.
    <strong>Multi-fork persists</strong> because τ_mass ≈ 48 min &lt; C + D = 100 min,
    a metabolism-fit gap that needs Studies 5-7 mechanisms (SeqA / RIDA / DDAH /
    DARS) to close properly.
  </span>
</div>

<h2>Study 1 — DnaA expression dynamics read-outs (against the chosen baseline)</h2>
<p style="font-size: 13px; color: var(--muted); max-width: 950px;">
  Six read-outs prescribed by the multi-phase workflow's Study 1 (PDF: <em>DnaA / Replication Initiation Model</em>). Tasks 1–4 (transcription, mRNA dynamics, translation, DnaA dilution) use existing v2ecoli machinery; this section just visualizes what those processes are already producing on the chosen baseline. Task 5 (dnaAp DnaA-box occupancy) and Task 6 (dynamic autorepression) are not yet implemented — Task 5 is shown as a placeholder, and Task 6 is the implementation deliverable that follows these read-outs.
</p>

<h3>Validation against expected behaviour (PDF)</h3>
{study1_table_html}

<h3>Read-out panels</h3>
{study1_panel_blocks}

</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  report → {out_path}  ({len(html)/1024:.1f} KB)")


def _load_historical_html(snap_name: str) -> str | None:
    """Return the contents of ``docs/historical/<snap_name>`` if present.

    Used by the report to fall back on snapshot HTML tables (per-condition
    parameter comparison, Study 1 validation summary) when the source DBs
    have been cleaned up. Returns None if the snapshot doesn't exist."""
    path = os.path.join("docs/historical", snap_name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_historical_png_b64(title: str) -> str | None:
    """Look up a previously-extracted historical PNG by panel title.

    The earlier full-data build wrote every panel into the report as a
    base64-embedded PNG; ``docs/historical/`` contains those PNGs
    standalone, indexed by ``docs/historical/manifest.json``. When the
    source DBs are deleted to reclaim disk, we read the historical PNG
    back as base64 so the report still shows the visual.

    Returns the base64-encoded PNG body, or None if no matching snapshot.
    """
    import json
    manifest_path = "docs/historical/manifest.json"
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    png_path = manifest.get(title)
    if not png_path or not os.path.isfile(png_path):
        return None
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="out/acetate_canonical_runs.db")
    parser.add_argument("--cache", default="out/cache_acetate_canonical")
    parser.add_argument("--out", default="docs/canonical_acetate_vs_stage1.html")
    args = parser.parse_args()

    # Acetate-specific deep-dive panels removed — report now focuses on
    # the multi-condition baseline. The 31-GB acetate DB is no longer
    # read at 1-second resolution; if you want the acetate-only panels
    # back, restore from git history.
    panels = []
    gens_summary = []

    # ----- Cross-condition baseline section -----
    print(f"[{time.strftime('%H:%M:%S')}] Reading 5-condition baseline DBs (subsample=50) ...")
    per_cond = read_all_conditions(subsample=50)
    summary = extract_condition_summary(per_cond)
    anim_gif_path = os.path.join(os.path.dirname(args.out) or ".", "replication_animation.gif")
    anim_gif_filename = os.path.basename(anim_gif_path)
    have_5cond_data = len(per_cond) == len(CONDITIONS_5)
    if have_5cond_data:
        make_replication_animation(per_cond, anim_gif_path, n_frames=160, fps=6)
    elif os.path.isfile(anim_gif_path):
        print(f"  5-cond DBs missing but {anim_gif_filename} exists — "
              "preserving the previous animation as a historical snapshot.")
    else:
        print(f"  5-cond DBs missing and no cached GIF — section will be skipped.")

    multi_panels = []
    if have_5cond_data or os.path.isfile(anim_gif_path):
        anim_caption_suffix = (
            "" if have_5cond_data
            else " <em>(snapshot from a previous build; the source SQLite DBs "
                 "were cleaned up to reclaim disk space — the animation is "
                 "preserved here as a reference for the cross-condition story.)</em>"
        )
        multi_panels.append((
            "Theta-replication animation — chromosome dynamics across 5 conditions",
            f'<animation:{anim_gif_filename}>',
            "Top row: chromosome(s) as circle(s); oriC = green dot at top; terminus = dark dot at bottom; active replication forks = red dots moving along the circle. "
            "Bottom row: oriC count (green), active forks (red), and full chromosomes (blue) vs time, with ▲ at initiation events (oriC step-up) and ▼ at termination events (chromosome step-up). A vertical bar tracks the animation cursor. Each strip uses its own gen length on the x-axis. "
            "All five panels share a single global time axis (the longest gen, acetate ~143 min) — when a faster condition divides first, its theta panel freezes at the final state and is labelled (divided) while the time bar keeps moving on the strip below. "
            "Multi-fork at with_aa = three nested rounds (6 forks) at birth, dropping to 4 once the outer round terminates. "
            "Single-round elsewhere = 2 forks emerge from oriC and meet at the terminus."
            + anim_caption_suffix))
    bulk_title = "Bulk metrics — cell mass, DnaA pool, total + occupied DnaA boxes"
    if have_5cond_data:
        multi_panels.append((
            bulk_title,
            panel_bulk_metrics_grid(per_cond),
            "All 5 conditions overlaid on each panel. Faster-growth conditions carry larger absolute pools (bigger cell). The total-DnaA-boxes panel steps up at each round-of-replication initiation (more genome copies = more box copies)."))
    else:
        hist_b64 = _load_historical_png_b64(bulk_title)
        if hist_b64:
            multi_panels.append((
                bulk_title, hist_b64,
                "All 5 conditions overlaid on each panel. Faster-growth conditions carry larger absolute pools (bigger cell). The total-DnaA-boxes panel steps up at each round-of-replication initiation (more genome copies = more box copies). "
                "<em>(Snapshot from a previous build; source SQLite DBs cleaned up to reclaim disk.)</em>"))
    if summary:
        multi_table_html = multicondition_param_table_html(summary)
    else:
        # No DBs to read from — fall back to the snapshot table.
        snap = _load_historical_html("snap_table_per_condition.html")
        multi_table_html = (
            snap + '<p style="font-size:11px;color:var(--muted);'
                   'font-style:italic;margin-top:6px;">'
                   'Snapshot from a previous build; source SQLite DBs cleaned up '
                   'to reclaim disk space.</p>'
            if snap else
            '<p style="color:var(--muted);"><em>No data available.</em></p>'
        )

    # ----- Stage 1 override section -----
    stage1_table_html = stage1_pdf_comparison_table_html()
    stage1_panels: list = []
    print(f"[{time.strftime('%H:%M:%S')}] Reading Stage 1 acetate run (subsample=50) ...")
    per_cond_stage1 = read_conditions(CONDITIONS_STAGE1, subsample=50)
    stage1_gif_path = os.path.join(
        os.path.dirname(args.out) or ".", "stage1_comparison_animation.gif"
    )
    stage1_gif_filename = os.path.basename(stage1_gif_path)
    have_stage1_data = len(per_cond_stage1) == len(CONDITIONS_STAGE1)

    # ALWAYS show the historical 3-panel "multi-fork motivation" animation
    # if it's been preserved on disk. This is the visual that motivates
    # the M* × 1.5 bump: the prior run with C/D applied but M* unbumped
    # showed a mid-cycle re-initiation (second INIT at t ≈ 156 min in the
    # 190-min Stage 1 cycle).
    motivation_gif = os.path.join(
        os.path.dirname(args.out) or ".", "stage1_multifork_motivation.gif"
    )
    if not os.path.isfile(motivation_gif) and os.path.isfile(stage1_gif_path):
        # First time we run after the cleanup — snapshot the existing
        # 3-panel comparison animation as our "why M* bump" reference.
        import shutil
        shutil.copy2(stage1_gif_path, motivation_gif)
        print(f"  preserved historical 3-panel animation → {motivation_gif}")
    if os.path.isfile(motivation_gif):
        stage1_panels.append((
            "Why we bumped M* — historical 3-panel comparison",
            f'<animation:{os.path.basename(motivation_gif)}>',
            "<strong>Left:</strong> acetate with v2ecoli canonical parameters. "
            "<strong>Middle:</strong> acetate with Stage 1 C/D applied but M* unbumped — "
            "cell mass crosses 2 × M* mid-cycle and a second initiation fires "
            "at t ≈ 156 min, before the first round terminates at t ≈ 171 min. "
            "<strong>Right:</strong> Stage 1 with only the cycle-neutral expression "
            "overrides (TE 1.0, DARS/datA) — still shows a brief mid-cycle re-init "
            "because canonical M* and the slow cell growth still let mass/oriC re-hit M*. "
            "<em>This is the observation that motivates the M* × 1.5 bump in the new "
            "full Stage 1 PDF cache shown below.</em>"))

    # Re-render the live 2-panel comparison from current data when both
    # DBs are available; otherwise note the section's pending status.
    if have_stage1_data:
        make_replication_animation(per_cond_stage1, stage1_gif_path,
                                   n_frames=160, fps=6,
                                   conditions=CONDITIONS_STAGE1,
                                   figsize=(11, 6.8))
        stage1_panels.append((
            "Acetate canonical vs Stage 1 PDF (5-gen, all knobs) — theta replication + fork-count strip",
            f'<animation:{stage1_gif_filename}>',
            "<strong>Left:</strong> acetate with v2ecoli canonical parameters (C ≈ 40 min, D = 20 min, M* from ParCa fit, dnaA TE = 0.35, dnaA expression at ParCa-fit rate) — reference baseline. "
            "<strong>Right:</strong> acetate with the <em>full Stage 1 PDF</em> cache. All seven post-ParCa knobs applied: C = 70 min, D = 30 min, M* × 1.5, dnaA basal_prob × 15, dnaA delta_prob_matrix zeroed (constitutive), dnaA monomer stable (degradation = 0), dnaA TE = 1.0. "
            "Multi-gen sim so later gens equilibrate the off-equilibrium gen-1 birth state."))
        stage1_panels.append((
            "Acetate canonical vs Stage 1 PDF — bulk metrics",
            panel_bulk_metrics_grid(per_cond_stage1, conditions=CONDITIONS_STAGE1),
            "Same four metrics as the cross-condition bulk panel. The Stage 1 PDF run is expected to carry a much higher DnaA pool (TE × 15 raw expression × stable protein) and a longer cycle (C + D = 100 min vs 60 min canonical)."))
    else:
        print(f"  Stage 1 acetate run not ready yet "
              f"(have {len(per_cond_stage1)}/{len(CONDITIONS_STAGE1)} conditions); "
              "diagnostic panels deferred until 5-gen sim completes — falling "
              "back to historical PNG snapshots where available.")
        # Fall back to the snapshot bulk-metrics PNG so the visual story
        # stays intact while we wait for the 5-gen sim. The historical
        # snapshot is the 3-variant version (canonical / Stage 1 full /
        # Stage 1 no C/D); flag that in the caption.
        hist_b64 = _load_historical_png_b64(
            "Acetate canonical vs Stage 1 (two variants) — bulk metrics"
        )
        if hist_b64:
            stage1_panels.append((
                "Acetate canonical vs Stage 1 — bulk metrics (historical snapshot)",
                hist_b64,
                "Snapshot from a previous build showing the 3-variant comparison "
                "(canonical / Stage 1 full with C/D pre-ParCa / Stage 1 no C/D). "
                "The new <code>acetate_stage1_full_5gen.db</code> with all seven "
                "post-ParCa knobs will replace this panel once the 5-gen sim "
                "completes."))

    # ----- Study 1 read-outs (chosen baseline: acetate no C/D) -----
    study1_table_html = ""
    study1_panels: list = []
    try:
        from studies.study1_dnaa_expression.plot_readouts import (
            resolve_indices, read_run, to_np, validation_summary,
            render_summary_table, panel_txn_events, panel_mrna_number,
            panel_mrna_conc, panel_dnaa_number, panel_dnaa_conc,
            panel_dnaAp_occupancy_stub,
        )
    except ImportError as e:
        print(f"  Study 1 module not importable: {e}; skipping section.")
    else:
        # Study 1 read-outs run against the full Stage 1 PDF cache (all
        # seven post-ParCa knobs applied). Prefer the multi-gen DB so we
        # see steady-state DnaA pool / mRNA / mass dynamics rather than
        # gen-1 off-equilibrium birth artefacts.
        if os.path.isfile("out/acetate_stage1_full_5gen.db"):
            s1_db = "out/acetate_stage1_full_5gen.db"
        elif os.path.isfile("out/acetate_stage1_no_cd_runs.db"):
            s1_db = "out/acetate_stage1_no_cd_runs.db"
        else:
            s1_db = ""
        s1_cache = ("out/cache_acetate_stage1_full"
                    if os.path.isdir("out/cache_acetate_stage1_full")
                    else "out/cache_acetate_stage1_no_cd")

        def _db_has_rows(path: str) -> bool:
            """True when the SQLite file is non-trivially queryable.
            Guards against partially-written DBs (in-flight sim) and
            zero-byte stubs left from a previous cleanup."""
            if not path or not os.path.isfile(path) or os.path.getsize(path) < 4096:
                return False
            try:
                conn = sqlite3.connect(path)
                row = conn.execute("SELECT COUNT(*) FROM history").fetchone()
                conn.close()
                return row is not None and int(row[0]) > 0
            except sqlite3.OperationalError:
                return False

        if _db_has_rows(s1_db) and os.path.isdir(s1_cache):
            print(f"[{time.strftime('%H:%M:%S')}] Building Study 1 readouts from {s1_db} ...")
            idx = resolve_indices(s1_cache, s1_db)
            print(idx.report())
            t_start = time.time()
            rows = read_run(s1_db, idx, subsample=1)
            print(f"  read {len(rows)} ticks in {time.time()-t_start:.1f}s")
            d = to_np(rows)
            study1_table_html = render_summary_table(validation_summary(d))
            study1_panels = [
                ("Study 1 / Task 1 — dnaA transcription events vs time",
                 panel_txn_events(d),
                 "Raw stems = events in each emitted tick; line = 1-min running sum."),
                ("Study 1 / Task 2a — dnaA mRNA number vs time",
                 panel_mrna_number(d),
                 "Mature dnaA mRNA count from <code>listeners.rna_counts.mRNA_counts[TU00259]</code>."),
                ("Study 1 / Task 2b — dnaA mRNA concentration vs time",
                 panel_mrna_conc(d),
                 "Count divided by listeners.mass.volume. With v2ecoli's ParCa-fit transcription rate (~0.1/min) and the 2.5-min mRNA half-life, the steady-state count is ~0.25 copies, so most ticks are zero — concentration is therefore near-zero with infrequent spikes."),
                ("Study 1 / Task 3+4 — DnaA number (total, ATP, ADP, apo)",
                 panel_dnaa_number(d),
                 "Green shaded band = Study 1 expected window (300–800 copies/cell from mass spec)."),
                ("Study 1 / Task 3+4 — DnaA concentration + DnaA-per-oriC",
                 panel_dnaa_conc(d),
                 "Left axis: DnaA / volume. Right axis (dotted, grey): DnaA / oriC. Green dotted line at ~200/oriC is the western-blot expected value."),
                ("Study 1 / Task 5 — dnaAp DnaA-box occupancy (placeholder)",
                 panel_dnaAp_occupancy_stub(),
                 "Not yet implemented. v2ecoli emits only global DnaA-box occupancy, not per-box state, so a dnaAp-specific tracker is needed before this read-out can be populated."),
            ]
        else:
            # Source DB or cache missing — fall back to historical PNG
            # snapshots so the section is still informative while the
            # 5-gen sim finishes / fixture rebuilds.
            print(f"  Study 1 source missing ({s1_db!r} / {s1_cache!r}); "
                  "falling back to historical PNG snapshots.")
            # Validation table also has a snapshot.
            snap_table = _load_historical_html("snap_table_study1_validation.html")
            if snap_table:
                study1_table_html = (
                    snap_table
                    + '<p style="font-size:11px;color:var(--muted);'
                      'font-style:italic;margin-top:6px;">'
                      'Snapshot from a previous build against the no-C/D '
                      'baseline (1 gen). New full Stage 1 PDF 5-gen run is in '
                      'flight and will replace this table when it completes.</p>'
                )
            historical_titles = [
                ("Study 1 / Task 1 — dnaA transcription events vs time",
                 "Raw stems = events in each emitted tick; line = 1-min running sum."),
                ("Study 1 / Task 2a — dnaA mRNA number vs time",
                 "Mature dnaA mRNA count from <code>listeners.rna_counts.mRNA_counts[TU00259]</code>."),
                ("Study 1 / Task 2b — dnaA mRNA concentration vs time",
                 "Count divided by listeners.mass.volume. The historical snapshot shows v2ecoli's pre-knob expression behaviour; once the 5-gen Stage 1 PDF run completes, this panel will refresh against the new (dnaA TE = 1.0, txn × 15, constitutive, stable) cache."),
                ("Study 1 / Task 3+4 — DnaA number (total, ATP, ADP, apo)",
                 "Green shaded band = Study 1 expected window (300–800 copies/cell from mass spec)."),
                ("Study 1 / Task 3+4 — DnaA concentration + DnaA-per-oriC",
                 "Left axis: DnaA / volume. Right axis (dotted, grey): DnaA / oriC. Green dotted line at ~200/oriC is the western-blot expected value."),
                ("Study 1 / Task 5 — dnaAp DnaA-box occupancy (placeholder)",
                 "Not yet implemented. v2ecoli emits only global DnaA-box occupancy, not per-box state, so a dnaAp-specific tracker is needed before this read-out can be populated."),
            ]
            for title, base_note in historical_titles:
                b64 = _load_historical_png_b64(title)
                if b64:
                    study1_panels.append((
                        title, b64,
                        f"{base_note} <em>(Snapshot from a previous build against "
                        f"the no-C/D baseline; new full Stage 1 PDF run is in flight.)</em>"))

    write_html(panels, gens_summary, args.out,
               multi_panels=multi_panels, multi_table_html=multi_table_html,
               stage1_table_html=stage1_table_html,
               stage1_panels=stage1_panels,
               study1_table_html=study1_table_html,
               study1_panels=study1_panels)
    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
