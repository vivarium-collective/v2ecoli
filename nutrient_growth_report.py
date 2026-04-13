"""
Nutrient-Growth Report

Exercises the nutrient-growth branch's extracted metabolic-kinetics step.
Runs a single-cell simulation with Michaelis-Menten glucose uptake in
``metabolic_kinetics.py`` and reports:

  * Dry-mass and growth-rate trajectories.
  * External glucose [GLC_ext] over time (once environment depletion is
    wired; stays flat until then).
  * Observed glucose uptake rate vs. the analytical MM curve.
  * Target doubling times from Caglar et al. 2017 (Sci Rep srep45303) —
    the calibration reference for future parameterization commits.

Usage:
    python nutrient_growth_report.py                  # 2520s default
    python nutrient_growth_report.py --duration 600   # shorter
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import shutil
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


REPORT_DIR = "out/nutrient_growth"
REPORT_NAME = "nutrient_growth_report.html"
CACHE_DIR = "out/cache"
# Fast-iteration defaults: short sim, tight environment volume so
# glucose depletes within the run.
DEFAULT_DURATION = 600          # seconds of sim time
DEFAULT_SNAPSHOT_INTERVAL = 10  # seconds between snapshots
DEFAULT_ENV_VOLUME_L = 1e-14    # 10 fL — depletes ~22 mM glucose in ~2 min
CAGLAR_DOUBLING_TIMES_CSV = (
    "data/caglar2017/41598_2017_BFsrep45303_MOESM56_ESM.csv")

# Exchanges to track in the flux panel. Union of the carbon/nitrogen/
# energy-relevant molecules; anything present in environment.exchange is
# reported, these just get highlighted colors.
TRACKED_EXCHANGES = [
    "GLC", "OXYGEN-MOLECULE", "CARBON-DIOXIDE", "AMMONIUM", "WATER",
    "Pi", "SULFATE", "K+", "MG+2", "NA+", "FE+2", "L-ALPHA-ALANINE",
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _as_float(x, default=None):
    if x is None:
        return default
    if hasattr(x, "asNumber"):
        try:
            return float(x.asNumber())
        except Exception:
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _extract_snapshot(state, t):
    """Pull the metrics this report cares about out of a composite state."""
    agent = state.get("agents", {}).get("0", {})
    mass = agent.get("listeners", {}).get("mass", {})
    boundary = agent.get("boundary", {})
    external = boundary.get("external", {}) if isinstance(boundary, dict) else {}
    exchange_data = (agent.get("environment", {})
                          .get("exchange_data", {}))
    constrained = exchange_data.get("constrained", {}) if isinstance(
        exchange_data, dict) else {}

    # environment.exchange = per-step count deltas from metabolism (negative
    # for uptake). Snapshot to a plain dict so later plotting doesn't
    # reference a live-mutating map.
    raw_exchange = agent.get("environment", {}).get("exchange", {}) or {}
    exchange_counts = {k: _as_float(v, 0.0) for k, v in raw_exchange.items()}

    glc_bound = _as_float(constrained.get("GLC[p]"))

    # Carbon + mass budget (from carbon_budget_listener).
    cb = agent.get("listeners", {}).get("carbon_budget", {}) or {}
    # FBA internal mass accounting — emitted by metabolism.py; see
    # analysis of modular_fba slack pseudofluxes.
    fba_results = agent.get("listeners", {}).get("fba_results", {}) or {}
    fba_mass_out = _as_float(fba_results.get("fba_mass_exchange_out"), 0.0)
    # Dark-matter accountant (phase 1).
    dm = agent.get("listeners", {}).get("dark_matter", {}) or {}

    return {
        "time": t,
        "dry_mass": float(mass.get("dry_mass", 0.0)),
        "cell_mass": float(mass.get("cell_mass", 0.0)),
        "protein_mass": float(mass.get("protein_mass", 0.0)),
        "rna_mass": float(mass.get("rna_mass", 0.0)),
        "dna_mass": float(mass.get("dna_mass", 0.0)),
        "smallmol_mass": float(mass.get("smallMolecule_mass", 0.0)),
        "water_mass": float(mass.get("water_mass", 0.0)),
        "growth_rate": float(mass.get("instantaneous_growth_rate", 0.0)),
        "glc_ext_mM": _as_float(external.get("GLC")),
        "glc_bound_mmol_gdcw_h": glc_bound,
        "external_mM": {k: _as_float(v, 0.0) for k, v in external.items()},
        "exchange_counts": exchange_counts,
        "c_in_mmol":   _as_float(cb.get("c_in_mmol"), 0.0),
        "c_out_mmol":  _as_float(cb.get("c_out_mmol"), 0.0),
        "c_net_mmol":  _as_float(cb.get("c_net_mmol"), 0.0),
        "cum_c_in":    _as_float(cb.get("cumulative_c_in_mmol"), 0.0),
        "cum_c_out":   _as_float(cb.get("cumulative_c_out_mmol"), 0.0),
        "biomass_c":   _as_float(cb.get("biomass_c_est_mmol_per_step"), 0.0),
        "mass_in_fg":  _as_float(cb.get("mass_in_fg"), 0.0),
        "mass_out_fg": _as_float(cb.get("mass_out_fg"), 0.0),
        "water_in_fg": _as_float(cb.get("water_in_fg"), 0.0),
        "cum_mass_in":    _as_float(cb.get("cumulative_mass_in_fg"), 0.0),
        "cum_mass_out":   _as_float(cb.get("cumulative_mass_out_fg"), 0.0),
        "cum_water_in":   _as_float(cb.get("cumulative_water_in_fg"), 0.0),
        "cum_dry_gained": _as_float(
            cb.get("cumulative_dry_mass_gained_fg"), 0.0),
        "mass_balance_deficit_fg": _as_float(
            cb.get("mass_balance_deficit_fg"), 0.0),
        "fba_mass_exchange_out": fba_mass_out,
        # Phase 2: enforced dark-matter pool from metabolism.py.
        "dm_pool_enforced": _as_float(
            fba_results.get("dark_matter_pool_fg"), 0.0),
        "dm_withdraw": _as_float(
            fba_results.get("dark_matter_withdraw_fg"), 0.0),
        "dm_deposit": _as_float(
            fba_results.get("dark_matter_deposit_fg"), 0.0),
        "dm_violation": _as_float(
            fba_results.get("dark_matter_violation_fg"), 0.0),
        "dm_scale": _as_float(fba_results.get("dark_matter_scale"), 1.0),
        "dark_matter_fg": _as_float(dm.get("dark_matter_fg"), 0.0),
        "dark_matter_delta_fg": _as_float(
            dm.get("dark_matter_delta_fg"), 0.0),
        "dm_cum_bulk_in": _as_float(
            dm.get("cumulative_bulk_mass_in_fg"), 0.0),
        "dm_cum_exch_in": _as_float(
            dm.get("cumulative_exchange_mass_in_fg"), 0.0),
        "dm_cum_viol": _as_float(
            dm.get("cumulative_violations_fg"), 0.0),
        "dm_cum_unaccounted": _as_float(
            dm.get("cumulative_unaccounted_fg"), 0.0),
        "dm_cell_in": _as_float(
            dm.get("cumulative_cell_mass_in_fg"), 0.0),
    }


def _patch_env_volume(composite, env_volume_L):
    """Find the live EnvironmentUpdate instance and override its volume."""
    from v2ecoli.steps.environment_update import EnvironmentUpdate, AVOGADRO
    found = None
    for step in composite.step_paths.values():
        inst = step.get("instance") if isinstance(step, dict) else step
        if isinstance(inst, EnvironmentUpdate):
            inst.env_volume_L = float(env_volume_L)
            inst._count_to_mM = 1000.0 / (AVOGADRO * float(env_volume_L))
            found = inst
    return found


def run_single_cell(duration: int, snapshot_interval: int, env_volume_L: float,
                    label: str = ""):
    """Run the baseline composite for `duration` seconds, snapshotting.
    ``label`` is prepended to per-step console prints so parallel runs
    stay visually disambiguated."""
    from v2ecoli.composite import make_composite
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    patched = _patch_env_volume(composite, env_volume_L)
    effective_env_vol = patched.env_volume_L if patched else None

    tag = f"[{label}] " if label else ""
    snaps = [_extract_snapshot(composite.state, 0.0)]
    total = 0.0
    t0 = time.time()
    crashed_at = None
    while total < duration:
        chunk = min(snapshot_interval, duration - total)
        try:
            composite.run(chunk)
        except Exception as e:
            print(f"  {tag}sim error at t≈{total+chunk:.0f}s: "
                  f"{type(e).__name__}: {str(e)[:80]}")
            crashed_at = total + chunk
            break
        total += chunk
        snaps.append(_extract_snapshot(composite.state, total))
    wall = time.time() - t0
    print(f"  {tag}{int(total)}s sim in {wall:.0f}s wall, "
          f"{len(snaps)} snapshots"
          + (f", crashed at t={crashed_at:.0f}s" if crashed_at else ""))
    return {
        "snapshots": snaps,
        "wall_time": wall,
        "sim_time": total,
        "env_volume_L": effective_env_vol,
        "crashed_at": crashed_at,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Caglar 2017 reference doubling times
# ---------------------------------------------------------------------------

def load_caglar_doubling_times():
    """Return {carbon_source: (mean_min, n_replicates)} from MOESM56."""
    path = CAGLAR_DOUBLING_TIMES_CSV
    if not os.path.exists(path):
        return {}
    per_cond: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cond = (row.get("name") or "").replace(".tab", "")
            try:
                dt = float(row["doubling.time.minutes"])
            except (TypeError, ValueError):
                continue
            per_cond.setdefault(cond, []).append(dt)
    return {k: (float(np.mean(v)), len(v)) for k, v in per_cond.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_mass(snaps):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    t = [s["time"] / 60 for s in snaps]
    ax.plot(t, [s["dry_mass"] for s in snaps], linewidth=2.0, color="#7c3aed")
    t_shut = _glc_shutoff_min(snaps)
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle=":", alpha=0.6,
                   linewidth=1.0, label=f"glc depleted (t={t_shut:.1f}min)")
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
    ax.grid(alpha=0.3)
    ax.set_title("Single-cell dry mass")
    return _fig_to_b64(fig)


def plot_growth_rate(snaps):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    t = [s["time"] / 60 for s in snaps]
    ax.plot(t, [s["growth_rate"] for s in snaps], linewidth=1.8, color="#7c3aed")
    t_shut = _glc_shutoff_min(snaps)
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (1/s)")
    ax.set_title("Instantaneous growth rate")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def plot_glucose_trajectory(snaps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    t = [s["time"] / 60 for s in snaps]
    ax1.plot(t, [s["glc_ext_mM"] for s in snaps], linewidth=1.8, color="#7c3aed")
    ax2.plot(t, [s["glc_bound_mmol_gdcw_h"] for s in snaps], linewidth=1.8,
             color="#7c3aed")
    ax1.set_xlabel("Time (min)"); ax1.set_ylabel("[GLC]ₑₓₜ (mM)")
    ax1.set_title("External glucose"); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("MM uptake bound (mmol/gDCW/h)")
    ax2.set_title("Kinetic glucose uptake bound")
    ax2.grid(alpha=0.3)
    return _fig_to_b64(fig)


# Stable per-molecule colors so panels are cross-comparable.
_EXCHANGE_COLORS = {
    "GLC":              "#dc2626",   # red
    "HYDROGEN-MOLECULE": "#8b5cf6",  # purple
    "CARBON-DIOXIDE":   "#0ea5e9",   # sky
    "OXYGEN-MOLECULE":  "#14b8a6",   # teal
    "WATER":            "#64748b",   # slate
    "AMMONIUM":         "#f97316",   # orange
    "PROTON":           "#eab308",   # yellow
    "Pi":               "#ec4899",   # pink
    "SULFATE":          "#10b981",   # emerald
    "K+":               "#6366f1",   # indigo
    "MG+2":             "#84cc16",   # lime
    "FE+2":             "#a16207",   # amber-brown
    "NA+":              "#22d3ee",   # cyan
    "CA+2":             "#f43f5e",   # rose
}
_FALLBACK_PALETTE = [
    "#6b7280", "#78716c", "#0f766e", "#4f46e5", "#be185d",
    "#047857", "#7c3aed", "#b45309", "#db2777", "#0891b2",
]


def _color_for(molecule: str, seen: dict) -> str:
    """Stable color per molecule; falls back to cycling palette."""
    if molecule in _EXCHANGE_COLORS:
        return _EXCHANGE_COLORS[molecule]
    if molecule not in seen:
        seen[molecule] = _FALLBACK_PALETTE[len(seen) % len(_FALLBACK_PALETTE)]
    return seen[molecule]


def _glc_shutoff_min(snaps, threshold=0.01):
    """Return time (minutes) at which [GLC] first drops below threshold, or None."""
    for s in snaps:
        g = s.get("glc_ext_mM")
        if g is None:
            continue
        if g <= threshold:
            return s["time"] / 60
    return None


def _top_by_metric(snaps, metric, n, exclude=None):
    """Rank molecules by metric function applied to per-molecule series."""
    exclude = set(exclude or ())
    mols: set[str] = set()
    for s in snaps:
        mols.update(s.get("exchange_counts", {}) or {})
    scored = []
    for m in mols:
        if m in exclude:
            continue
        series = [s.get("exchange_counts", {}).get(m, 0) for s in snaps]
        scored.append((metric(series), m))
    scored.sort(reverse=True)
    return [m for _, m in scored[:n]]


def _draw_exchange_series(ax, snaps, molecules, seen_colors, *,
                           symlog_threshold=1e3):
    """Draw per-molecule import rates. Sign is flipped from the raw
    exchange convention (raw: negative = import, positive = secretion) so
    positive values on this axis mean import."""
    t = [s["time"] / 60 for s in snaps]
    for m in molecules:
        y = [-s.get("exchange_counts", {}).get(m, 0) for s in snaps]
        ax.plot(t, y, label=m, linewidth=1.6,
                color=_color_for(m, seen_colors))
    ax.axhline(0, color="#475569", linestyle="-", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Import rate (count / step)\n↑ import   secretion ↓")
    ax.grid(alpha=0.25, which="both")
    ax.set_yscale("symlog", linthresh=symlog_threshold)
    ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)


def plot_exchange_fluxes(snaps, top_n: int = 8):
    """Two stacked panels — imports (top, uptake rate positive) and
    secretions (bottom, secretion rate positive). Glucose is included
    in the import panel so the reader can see its decay right up to
    depletion. Shared x-axis, shared color palette across panels."""
    if not snaps:
        return None
    t_shut = _glc_shutoff_min(snaps)

    imports = _top_by_metric(
        snaps,
        metric=lambda s: max((-x for x in s if x < 0), default=0),
        n=top_n)
    exports = _top_by_metric(
        snaps,
        metric=lambda s: max((x for x in s if x > 0), default=0),
        n=top_n)

    fig, (axI, axE) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"hspace": 0.12})
    seen_colors: dict = {}

    t = [s["time"] / 60 for s in snaps]
    # IMPORT panel: plot as positive (negate raw exchange).
    for m in imports:
        y = [max(0, -s.get("exchange_counts", {}).get(m, 0)) for s in snaps]
        lw = 2.6 if m == "GLC" else 1.6
        axI.plot(t, y, label=m, linewidth=lw, color=_color_for(m, seen_colors))
    axI.set_yscale("symlog", linthresh=1e3)
    axI.set_ylabel("Import rate\n(count / step)")
    axI.set_title(f"Imports (top {len(imports)}) — green zone = uptake")
    axI.grid(alpha=0.25, which="both")
    axI.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    axI.set_facecolor("#f0fdf4")  # faint green tint

    # EXPORT panel: plot raw positive exchange.
    for m in exports:
        y = [max(0, s.get("exchange_counts", {}).get(m, 0)) for s in snaps]
        axE.plot(t, y, label=m, linewidth=1.6, color=_color_for(m, seen_colors))
    axE.set_yscale("symlog", linthresh=1e3)
    axE.set_ylabel("Export rate\n(count / step)")
    axE.set_xlabel("Time (min)")
    axE.set_title(f"Secretions (top {len(exports)}) — red zone = dumping out")
    axE.grid(alpha=0.25, which="both")
    axE.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    axE.set_facecolor("#fef2f2")  # faint red tint

    if t_shut is not None:
        for ax in (axI, axE):
            ax.axvline(t_shut, color="#dc2626", linestyle="--",
                       alpha=0.7, linewidth=1.4)
        axI.annotate(
            f"glucose depleted ({t_shut:.1f} min)",
            xy=(t_shut, 0), xytext=(t_shut + 0.15, 1e4),
            textcoords="data", color="#dc2626", fontsize=9)

    return _fig_to_b64(fig)


def plot_exchange_diff(snaps, n: int = 8):
    """Two-panel "what changed": left = biggest gainers (more flux after
    glucose shutoff), right = biggest losers. Shared color scheme with the
    main exchange panel.
    """
    if not snaps:
        return None
    t_shut = _glc_shutoff_min(snaps)
    if t_shut is None:
        return None  # no shutoff ⇒ nothing to diff against

    pre = [s for s in snaps if (s.get("glc_ext_mM") or 0) > 1.0]
    post = [s for s in snaps if (s.get("glc_ext_mM") or 0) <= 0.01]
    if not pre or not post:
        return None

    # Work in import-rate space (negated raw exchange). Positive = import.
    mols: set[str] = set()
    for s in snaps:
        mols.update(s.get("exchange_counts", {}) or {})
    scores = []
    for m in mols:
        if m == "GLC":
            continue  # glucose has its own trajectory panel
        pre_import = np.mean(
            [-s.get("exchange_counts", {}).get(m, 0) for s in pre])
        post_import = np.mean(
            [-s.get("exchange_counts", {}).get(m, 0) for s in post])
        scores.append((m, float(pre_import), float(post_import),
                       float(post_import - pre_import)))

    up = sorted(scores, key=lambda r: -r[3])[:n]   # biggest import increase
    down = sorted(scores, key=lambda r: r[3])[:n]  # biggest import decrease

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    seen_colors: dict = {}
    _draw_exchange_series(axL, snaps, [m for m, *_ in up], seen_colors)
    _draw_exchange_series(axR, snaps, [m for m, *_ in down], seen_colors)

    for ax in (axL, axR):
        ax.axvline(t_shut, color="#dc2626", linestyle="--",
                   alpha=0.6, linewidth=1.2)
    axL.set_title(f"Imports that INCREASE after glucose off (top {n})")
    axR.set_title(f"Imports that DECREASE after glucose off (top {n})")
    fig.suptitle(
        "Which imports pick up the slack when glucose is cut off?",
        fontsize=12, y=1.02)
    return _fig_to_b64(fig)


def plot_externals(snaps, top_n: int = 6):
    """External concentration trajectories for the most-changing species."""
    if not snaps:
        return None
    all_mols: set[str] = set()
    for s in snaps:
        all_mols.update(s.get("external_mM", {}) or {})
    # rank by absolute change (first → last)
    first = snaps[0].get("external_mM", {})
    last = snaps[-1].get("external_mM", {})
    deltas = sorted(
        ((abs((last.get(m, 0) or 0) - (first.get(m, 0) or 0)), m)
         for m in all_mols),
        reverse=True)
    top = [m for _, m in deltas[:top_n]]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    t = [s["time"] / 60 for s in snaps]
    for m in top:
        y = [s.get("external_mM", {}).get(m, 0) for s in snaps]
        ax.plot(t, y, label=m, linewidth=1.4)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("[M]ₑₓₜ (mM)")
    ax.set_title(f"Top {len(top)} external concentrations (by total change)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="upper right")
    return _fig_to_b64(fig)


def plot_mass_composition(snaps):
    """Stacked mass composition. Answers "what's still growing?" when the cell
    is starved — with DM on, nothing should grow after scale=0."""
    if not snaps:
        return None
    import numpy as np
    comps = [
        ("protein_mass",  "Protein",        "#2563eb"),
        ("rna_mass",      "RNA",            "#9333ea"),
        ("dna_mass",      "DNA",            "#dc2626"),
        ("smallmol_mass", "Small molecule", "#f97316"),
    ]
    t_shut = _glc_shutoff_min(snaps)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = np.array([s["time"] / 60 for s in snaps])
    ys = [np.array([s.get(k, 0) for s in snaps]) for k, _, _ in comps]
    ax.stackplot(t, *ys, labels=[c[1] for c in comps],
                 colors=[c[2] for c in comps], alpha=0.85)
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle="--",
                   alpha=0.6, linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dry mass (fg)")
    ax.set_title("Dry-mass composition")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    return _fig_to_b64(fig)


def plot_dark_matter_enforcement(snaps):
    """Phase 2 enforcement: pool trajectory, biomass-scale factor, and
    cumulative violations *after* scaling.
    """
    if not snaps or len(snaps) < 2:
        return None
    import numpy as np
    t = np.array([s["time"] / 60 for s in snaps])
    pool = np.array([s.get("dm_pool_enforced", 0) for s in snaps])
    scale = np.array([s.get("dm_scale", 1) for s in snaps])
    viol = np.array([s.get("dm_violation", 0) for s in snaps])
    cum_viol = np.cumsum(viol)
    t_shut = _glc_shutoff_min(snaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(t, scale, color="#7c3aed", linewidth=2.0,
             label="biomass scale (per step)")
    ax1.fill_between(t, 0, scale, alpha=0.15, color="#ddd6fe")
    ax1.set_ylim(-0.05, 1.15)
    if t_shut is not None:
        ax1.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax1.axhline(1, color="#475569", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("scale (0 = no growth, 1 = full)")
    ax1.set_title("Dark-matter biomass scaling")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="lower left")

    ax2.plot(t, pool, color="#2563eb", linewidth=2.0,
             label="enforced pool (fg)")
    ax2b = ax2.twinx()
    ax2b.plot(t, cum_viol, color="#dc2626", linewidth=1.5,
              linestyle="--", label="cumulative residual violation (fg)")
    if t_shut is not None:
        ax2.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Pool (fg)", color="#2563eb")
    ax2b.set_ylabel("Cum. violation (fg)", color="#dc2626")
    ax2.set_title("Pool state + residual (should stay near 0)")
    ax2.grid(alpha=0.3)
    h1,l1 = ax2.get_legend_handles_labels()
    h2,l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8, loc="best")
    return _fig_to_b64(fig)


def plot_dark_matter(snaps):
    """Cell-mass based dark matter: Δ cell_mass vs Δ boundary_mass.
    Positive pool = mass created; near-zero = balanced;
    negative pool = mass unaccounted (likely unknown-MW secretions).
    """
    if not snaps:
        return None
    import numpy as np
    t = np.array([s["time"] / 60 for s in snaps])
    pool = np.array([s.get("dark_matter_fg", 0) for s in snaps])
    cell = np.array([s.get("dm_cell_in", 0) for s in snaps])
    exch = np.array([s.get("dm_cum_exch_in", 0) for s in snaps])
    viol = np.array([s.get("dm_cum_viol", 0) for s in snaps])
    unacc = np.array([s.get("dm_cum_unaccounted", 0) for s in snaps])
    t_shut = _glc_shutoff_min(snaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    # Left: pool state — sign matters.
    ax1.plot(t, pool, color="#7c3aed", linewidth=2.0,
             label="Pool state")
    ax1.fill_between(t, 0, pool, where=(pool > 0),
                     color="#fecaca", alpha=0.5,
                     label="Δ > 0: mass CREATED (violation)")
    ax1.fill_between(t, 0, pool, where=(pool < 0),
                     color="#bfdbfe", alpha=0.5,
                     label="Δ < 0: mass UNACCOUNTED (secretions / lag)")
    ax1.axhline(0, color="#475569", linestyle="-", alpha=0.4,
                linewidth=1.0)
    if t_shut is not None:
        ax1.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Pool state (fg)")
    ax1.set_title(
        "Pool state = Σ(Δ cell_mass − Δ boundary_mass)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="best")

    # Right: cumulative curves + separate counters for each direction.
    ax2.plot(t, cell, color="#2563eb", linewidth=2.0,
             label="Σ Δ cell_mass (incl. unique pool)")
    ax2.plot(t, exch, color="#16a34a", linewidth=2.0,
             label="Σ Δ boundary (imports − secretions)")
    ax2.plot(t, viol, color="#dc2626", linewidth=1.4, linestyle=":",
             label="Σ violations (mass created)")
    ax2.plot(t, -unacc, color="#0ea5e9", linewidth=1.4, linestyle=":",
             label="-Σ unaccounted (mass missing)")
    if t_shut is not None:
        ax2.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Cumulative mass (fg)")
    ax2.set_title("Cumulative mass flow — cell vs boundary")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="best")
    return _fig_to_b64(fig)


def plot_fba_mass_accounting(snaps):
    """Overlay wholecell's built-in FBA mass accounting against our
    observed dry-mass growth. When the two diverge, the LP is
    producing biomass via the homeostatic quadFractionFromUnity slack
    fluxes, which are bounded ``[-inf, +inf]`` and thus act as
    unbounded mass sources/sinks.
    """
    if not snaps:
        return None
    import numpy as np
    t = np.array([s["time"] / 60 for s in snaps])
    # Cumulative FBA-reported exchange mass accumulation (pseudoflux units)
    fba_rates = np.array([s.get("fba_mass_exchange_out", 0) for s in snaps])
    cum_fba = np.cumsum(fba_rates)
    # Dry mass growth relative to first snapshot
    dry = np.array([s.get("dry_mass", 0) for s in snaps])
    if len(dry):
        dry = dry - dry[0]
    t_shut = _glc_shutoff_min(snaps)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax2 = ax.twinx()
    ax.plot(t, dry, color="#2563eb", linewidth=2.0,
            label="Dry mass gained (fg, left axis)")
    ax2.plot(t, cum_fba, color="#f97316", linewidth=1.8, linestyle="--",
             label="Σ FBA exchange mass (pseudoflux, right axis)")
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle="--", alpha=0.6,
                   linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dry mass Δ (fg)", color="#2563eb")
    ax2.set_ylabel("Σ FBA exchange mass (pseudoflux)", color="#f97316")
    ax.set_title(
        "FBA-reported exchange mass vs actual dry-mass growth")
    ax.grid(alpha=0.3)
    # Shared legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    return _fig_to_b64(fig)


def plot_mass_balance(snaps):
    """Cumulative mass in / out / into biomass, plus the deficit —
    the quantitative "carbon appearing from nowhere" signal.

    If the cell is mass-balanced:
        Σ imports = Σ secretions + Σ dry-mass-gained
    Any gap is the amount of mass the LP is producing without a
    corresponding exchange flux — the "free biomass" problem.
    """
    if not snaps:
        return None
    import numpy as np
    t = np.array([s["time"] / 60 for s in snaps])
    cum_in = np.array([s.get("cum_mass_in", 0) for s in snaps])
    cum_out = np.array([s.get("cum_mass_out", 0) for s in snaps])
    cum_gained = np.array([s.get("cum_dry_gained", 0) for s in snaps])
    deficit = np.array([s.get("mass_balance_deficit_fg", 0) for s in snaps])
    cum_water_in = np.array([s.get("cum_water_in", 0) for s in snaps])
    t_shut = _glc_shutoff_min(snaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(t, cum_in, label="Σ mass imported (non-water)",
             color="#16a34a", linewidth=1.8)
    ax1.plot(t, cum_out, label="Σ mass secreted",
             color="#dc2626", linewidth=1.8)
    ax1.plot(t, cum_gained, label="Σ dry-mass gained",
             color="#2563eb", linewidth=1.8)
    ax1.plot(t, cum_water_in, label="Σ water imported (not dry)",
             color="#64748b", linestyle=":", linewidth=1.4)
    if t_shut is not None:
        ax1.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Cumulative (fg)")
    ax1.set_title("Mass budget — what goes in, what goes out, what grows")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="best")

    # Right: the deficit. Expected = 0 for mass-balanced cell.
    ax2.plot(t, deficit, color="#dc2626", linewidth=2.0,
             label="(Σ in − Σ out) − Σ dry-gained")
    ax2.fill_between(t, deficit, 0, where=(deficit < 0),
                     color="#fecaca", alpha=0.5,
                     label="Imports can't account for growth")
    ax2.axhline(0, color="#475569", linestyle="-", alpha=0.4, linewidth=1.0)
    if t_shut is not None:
        ax2.axvline(t_shut, color="#dc2626", linestyle="--",
                    alpha=0.6, linewidth=1.0)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Deficit (fg)")
    ax2.set_title("Mass-balance check — negative = free mass")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="best")
    return _fig_to_b64(fig)


def plot_carbon_budget(snaps):
    """Per-step carbon flux + cumulative carbon budget.

    The point of this panel: a *closed* cell has C_in ≥ C_out + C_biomass
    each step. When the cell grows without a carbon source, C_biomass is
    positive while C_in is zero — a visual signature of the homeostatic
    objective "creating carbon" from internal pool drain.
    """
    if not snaps or len(snaps) < 2:
        return None
    import numpy as np
    t = np.array([s["time"] / 60 for s in snaps])
    c_in = np.array([s.get("c_in_mmol", 0) for s in snaps])
    c_out = np.array([s.get("c_out_mmol", 0) for s in snaps])
    c_biomass = np.array([s.get("biomass_c", 0) for s in snaps])
    cum_in = np.array([s.get("cum_c_in", 0) for s in snaps])
    cum_out = np.array([s.get("cum_c_out", 0) for s in snaps])
    t_shut = _glc_shutoff_min(snaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(t, c_in, color="#16a34a", label="C in (imports)",
             linewidth=1.8)
    ax1.plot(t, c_out, color="#dc2626", label="C out (secreted)",
             linewidth=1.8)
    ax1.plot(t, c_biomass, color="#2563eb", label="C into biomass (est.)",
             linewidth=1.8)
    if t_shut is not None:
        ax1.axvline(t_shut, color="#dc2626", linestyle="--", alpha=0.6,
                    linewidth=1.0)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("mmol C / step")
    ax1.set_title("Carbon fluxes per step")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="best")

    ax2.plot(t, cum_in, color="#16a34a", label="Σ C in")
    ax2.plot(t, cum_out, color="#dc2626", label="Σ C out")
    deficit = cum_out - cum_in
    ax2.fill_between(t, 0, deficit, where=(deficit > 0),
                     color="#fca5a5", alpha=0.35,
                     label="Σ out > Σ in (pool drain)")
    if t_shut is not None:
        ax2.axvline(t_shut, color="#dc2626", linestyle="--", alpha=0.6,
                    linewidth=1.0)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Cumulative mmol C")
    ax2.set_title("Cumulative carbon budget")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Carbon accounting — is the cell closed under mass balance?",
        fontsize=12, y=1.02)
    return _fig_to_b64(fig)


def plot_external_concentrations(snaps, top_n: int = 8):
    """Track how ALL boundary concentrations evolve, not just glucose.

    The environment_update step writes back to boundary.external for
    every species metabolism exchanges — so ammonium, phosphate, sulfate,
    ions, etc. all drain naturally. This panel shows which run out first.
    """
    if not snaps:
        return None
    import numpy as np
    all_mols: set[str] = set()
    for s in snaps:
        all_mols.update(s.get("external_mM", {}) or {})
    # Rank by max-min span (ignore infinities — WATER / O2 are flagged inf).
    scored = []
    for m in all_mols:
        series = [s.get("external_mM", {}).get(m, 0) for s in snaps]
        series = [v for v in series if v is not None and v != float("inf")
                  and v != float("-inf")]
        if not series:
            continue
        span = max(series) - min(series)
        scored.append((span, m))
    scored.sort(reverse=True)
    top = [m for _, m in scored[:top_n]]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    t = [s["time"] / 60 for s in snaps]
    seen: dict = {}
    for m in top:
        y = [s.get("external_mM", {}).get(m, 0) for s in snaps]
        y = [None if (v == float("inf") or v == float("-inf")) else v
             for v in y]
        ax.plot(t, y, label=m, linewidth=1.6, color=_color_for(m, seen))
    t_shut = _glc_shutoff_min(snaps)
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle="--", alpha=0.6,
                   linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("[M]ₑₓₜ (mM)")
    ax.set_title(f"External nutrient concentrations — top {len(top)} "
                 "species by range")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="best")
    return _fig_to_b64(fig)


def plot_mm_curve(vmax: float, km: float):
    """Analytical MM curve across typical [GLC] concentrations."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    glc = np.logspace(-5, 2, 400)  # 1e-5 mM to 100 mM
    v = vmax * glc / (km + glc)
    ax.semilogx(glc, v, color="#0ea5e9", linewidth=2)
    ax.axvline(km, color="#64748b", linestyle="--", alpha=0.6,
               label=f"K_m = {km} mM")
    ax.axhline(vmax / 2, color="#64748b", linestyle=":", alpha=0.4)
    ax.set_xlabel("[GLC]ₑₓₜ (mM)"); ax.set_ylabel("Uptake (mmol/gDCW/h)")
    ax.set_title(f"Michaelis-Menten: v_max = {vmax}, K_m = {km} mM")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

# Consistent color-coded run identifiers for per-variant panel headers.
_RUN_STYLES = {
    "baseline (DM off)": {"bg": "#e0e7ff", "fg": "#3730a3", "badge": "DM OFF"},
    "enforced (DM on)":  {"bg": "#ede9fe", "fg": "#6d28d9", "badge": "DM ON"},
}


def _side_by_side_plot(datasets, single_plot_fn, title_suffix=""):
    """Render `single_plot_fn(snaps)` for each (label, snaps) in datasets
    into a row of PNG images. Each panel gets a color-coded header tying
    the image back to its run so the two columns are never confused."""
    out = []
    for label, snaps in datasets:
        if not snaps:
            continue
        b = single_plot_fn(snaps)
        if not b:
            continue
        style = _RUN_STYLES.get(label, {"bg": "#f1f5f9", "fg": "#475569", "badge": label})
        header = (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:6px 10px;margin-bottom:4px;border-radius:4px 4px 0 0;'
            f'background:{style["bg"]};color:{style["fg"]};font-size:0.9em;'
            f'font-weight:600">'
            f'<span style="background:{style["fg"]};color:white;'
            f'padding:2px 8px;border-radius:3px;font-size:0.8em">'
            f'{style["badge"]}</span>'
            f'<span>{label}{title_suffix}</span></div>'
        )
        out.append(
            f'<div style="display:inline-block;width:49%;vertical-align:top;'
            f'padding-right:4px">{header}'
            f'<img src="data:image/png;base64,{b}" '
            f'style="max-width:100%;border:1px solid #e2e8f0;'
            f'border-radius:0 0 4px 4px"></div>')
    return "".join(out)


def generate_report(data, caglar, duration: int, vmax: float, km: float,
                    dark_matter: bool = False, data_enforced=None):
    from v2ecoli.library.repro_banner import banner_html
    repro = banner_html()

    snaps = data["snapshots"]
    final = snaps[-1] if snaps else {}
    initial_glc = snaps[0].get("glc_ext_mM") if snaps else None

    caglar_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v[0]:.1f}</td><td>{v[1]}</td></tr>"
        for k, v in sorted(caglar.items())
    )

    snaps_e = data_enforced["snapshots"] if data_enforced else []
    ds_pair = [
        ("baseline (DM off)", snaps),
        ("enforced (DM on)",  snaps_e),
    ]

    plots = {
        "mass": _side_by_side_plot(ds_pair, plot_mass),
        "growth_rate": _side_by_side_plot(ds_pair, plot_growth_rate),
        "glucose": _side_by_side_plot(ds_pair, plot_glucose_trajectory),
        "mm": plot_mm_curve(vmax, km),
        "exchanges": _side_by_side_plot(ds_pair, plot_exchange_fluxes),
        "exchange_diff": _side_by_side_plot(ds_pair, plot_exchange_diff),
        "externals": _side_by_side_plot(ds_pair, plot_externals),
        "carbon_budget": _side_by_side_plot(ds_pair, plot_carbon_budget),
        "externals_all": _side_by_side_plot(
            ds_pair, plot_external_concentrations),
        "mass_composition": _side_by_side_plot(ds_pair, plot_mass_composition),
        "mass_balance": _side_by_side_plot(ds_pair, plot_mass_balance),
        "fba_mass_probe": _side_by_side_plot(
            ds_pair, plot_fba_mass_accounting),
        "dark_matter": _side_by_side_plot(ds_pair, plot_dark_matter),
        "dm_enforcement": plot_dark_matter_enforcement(snaps_e) if snaps_e
                           else None,
    }

    # Pre/post-shutoff flux table — concrete answer to "what takes over
    # when glucose runs out".
    diff_rows = ""
    t_shut = _glc_shutoff_min(snaps) if snaps else None
    if snaps and t_shut is not None:
        pre = [s for s in snaps if (s.get("glc_ext_mM") or 0) > 1.0]
        post = [s for s in snaps if (s.get("glc_ext_mM") or 0) <= 0.01]
        if pre and post:
            mols: set[str] = set()
            for s in snaps:
                mols.update(s.get("exchange_counts", {}) or {})
            scored = []
            for m in mols:
                a = float(np.mean([s.get("exchange_counts", {}).get(m, 0)
                                   for s in pre]))
                b = float(np.mean([s.get("exchange_counts", {}).get(m, 0)
                                   for s in post]))
                scored.append((m, a, b, b - a))
            scored.sort(key=lambda r: abs(r[3]), reverse=True)
            rows_html = []
            for m, a, b, d in scored[:12]:
                sign = ("↑" if d > 0 else "↓")
                rows_html.append(
                    f"<tr><td>{m}</td>"
                    f"<td>{a:+,.0f}</td>"
                    f"<td>{b:+,.0f}</td>"
                    f"<td><strong>{sign} {abs(d):+,.0f}</strong></td></tr>")
            diff_rows = "\n".join(rows_html)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, REPORT_NAME)

    def img(key, alt):
        b = plots.get(key)
        if not b:
            return ""
        # Side-by-side plots already return full HTML; single-image
        # plots return base64 strings that need wrapping.
        if "<img" in b or "<div" in b:
            return f'<div class="plot">{b}</div>'
        return (f'<div class="plot"><img src="data:image/png;base64,{b}" '
                f'alt="{alt}"></div>')

    final_dry = final.get("dry_mass", 0.0)
    final_glc = final.get("glc_ext_mM")
    final_bound = final.get("glc_bound_mmol_gdcw_h")
    env_volume_L = data.get("env_volume_L") or float("nan")
    env_volume_fL = env_volume_L * 1e15 if env_volume_L == env_volume_L else float("nan")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Nutrient-Growth Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px;
       margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }}
h2 {{ color: #1e40af; margin-top: 2em; scroll-margin-top: 4em; }}
/* Sticky top navigation */
nav.toc {{ position: sticky; top: 0; background: #f8fafc;
          padding: 10px 0 6px; margin: -12px -20px 16px; z-index: 10;
          border-bottom: 1px solid #e2e8f0; }}
nav.toc .inner {{ max-width: 1200px; margin: 0 auto; padding: 0 20px;
                 display: flex; gap: 6px; flex-wrap: wrap; }}
nav.toc a {{ display: inline-block; padding: 4px 10px;
            background: #eff6ff; color: #1e40af; border-radius: 999px;
            font-size: 0.82em; text-decoration: none;
            border: 1px solid #dbeafe; white-space: nowrap; }}
nav.toc a:hover {{ background: #dbeafe; }}
h3 {{ color: #334155; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ padding: 6px 14px; border: 1px solid #e2e8f0; text-align: right; }}
th {{ background: #f1f5f9; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
.plot {{ margin: 1em 0; text-align: center; }}
.plot img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }}
.note {{ background: #fef3c7; border-left: 4px solid #f59e0b;
         padding: 10px 14px; margin: 1em 0; font-size: 0.95em; }}
.wip {{ background: #e0f2fe; border-left: 4px solid #0284c7;
        padding: 10px 14px; margin: 1em 0; font-size: 0.95em; }}
.perf {{ display: flex; gap: 1.5em; margin: 1em 0; flex-wrap: wrap; }}
.perf-card {{ background: white; border: 1px solid #e2e8f0;
              border-radius: 8px; padding: 1em 1.5em; text-align: center;
              min-width: 140px; }}
.perf-card .value {{ font-size: 1.6em; font-weight: bold; color: #1e40af; }}
.perf-card .label {{ color: #64748b; font-size: 0.85em; }}
footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
          color: #64748b; font-size: 0.9em; }}
</style></head><body>
{repro}

<nav class="toc"><div class="inner">
  <a href="#summary">Summary</a>
  <a href="#goals">Goals</a>
  <a href="#growth">Growth</a>
  <a href="#glc-kinetics">Glucose kinetics</a>
  <a href="#exchange">Exchange fluxes</a>
  <a href="#externals">External concentrations</a>
  <a href="#carbon">Carbon budget</a>
  <a href="#growing">What's growing?</a>
  <a href="#mass-balance">Mass balance</a>
  <a href="#manufacture">How biomass is made</a>
  <a href="#dark-matter">Dark matter</a>
  <a href="#caglar">Caglar targets</a>
  <a href="#todo">Not covered yet</a>
</div></nav>

<h1 id="summary">Nutrient-Growth Report</h1>

<div style="display:flex;gap:10px;margin:1em 0;flex-wrap:wrap">
  <div style="flex:1;min-width:260px;background:#e0e7ff;border-left:4px solid #3730a3;padding:10px 14px;border-radius:4px">
    <span style="background:#3730a3;color:white;padding:2px 8px;border-radius:3px;font-size:0.8em">DM OFF</span>
    <strong style="color:#3730a3">&nbsp;Baseline.</strong>
    MM glucose + environment depletion; LP homeostatic slacks unconstrained.
  </div>
  <div style="flex:1;min-width:260px;background:#ede9fe;border-left:4px solid #6d28d9;padding:10px 14px;border-radius:4px">
    <span style="background:#6d28d9;color:white;padding:2px 8px;border-radius:3px;font-size:0.8em">DM ON</span>
    <strong style="color:#6d28d9">&nbsp;Enforced.</strong>
    Baseline + dark-matter scale ∈ [0, 1] so net cell mass gain ≤ boundary imports.
  </div>
</div>

<h2 id="goals">Goals</h2>
<p>Calibrate v2ecoli growth against Caglar et al. 2017
(<a href="https://doi.org/10.1038/srep45303" target="_blank">srep45303</a>):
4 carbon sources × 10 Mg²⁺ × 3 Na⁺ × 3 growth phases. Three design
moves on this branch:</p>
<ol>
  <li><strong>Continuous glucose sensitivity</strong> — Michaelis-Menten
  uptake on <code>boundary.external.GLC</code> replaces the media-id lookup.</li>
  <li><strong>Closed depletion loop</strong> — <code>environment_update</code>
  writes exchange deltas back to <code>boundary.external</code> using a
  configurable per-cell volume.</li>
  <li><strong>Mass-conservation enforcement</strong> — a dark-matter pool
  scales LP output so net cell mass never exceeds boundary imports;
  growth freezes when carbon runs out.</li>
</ol>

<h2 id="summary-details">Run summary</h2>
<div class="perf">
  <div class="perf-card"><div class="value">{duration}s</div>
    <div class="label">sim duration</div></div>
  <div class="perf-card"><div class="value">{data.get('wall_time',0):.0f}s</div>
    <div class="label">wall time</div></div>
  <div class="perf-card"><div class="value">{final_dry:.0f} fg</div>
    <div class="label">final dry mass</div></div>
  <div class="perf-card"><div class="value">{initial_glc if initial_glc is not None else '—'}</div>
    <div class="label">initial [GLC] (mM)</div></div>
  <div class="perf-card"><div class="value">{final_glc if final_glc is not None else '—'}</div>
    <div class="label">final [GLC] (mM)</div></div>
  <div class="perf-card"><div class="value">{final_bound if final_bound is not None else '—':.2f}</div>
    <div class="label">final MM bound<br/>(mmol/gDCW/h)</div></div>
  <div class="perf-card"><div class="value">{env_volume_fL:.1f} fL</div>
    <div class="label">env volume / cell<br/>(configurable)</div></div>
</div>

<h2 id="growth">Growth trajectory</h2>
<p>Dry mass (top row) and instantaneous growth rate (bottom row) for
both runs. <strong>The DM-OFF cell keeps gaining mass well past glucose
depletion</strong> (red dotted line) because the LP's homeostatic slacks
can synthesize biomass without a matching carbon import. <strong>The
DM-ON cell flattens out</strong> once scale → 0 — growth literally
cannot proceed without boundary carbon.</p>
{img("mass", "Dry mass vs time")}
{img("growth_rate", "Growth rate vs time")}

<h2 id="glc-kinetics">Glucose uptake kinetics</h2>
<p>MM bound computed each step from <code>boundary.external.GLC</code>:
<strong>v_max = {vmax} mmol/gDCW/h</strong>, <strong>K_m = {km} mM</strong>.</p>

{img("mm", "MM curve")}
{img("glucose", "External glucose and applied uptake bound")}

<h2 id="exchange">Exchange fluxes</h2>
<p>Per-step boundary counts (negative = import, positive = secretion).
Y-axis is symlog; red dashed line marks [GLC]ₑₓₜ &lt; 0.01&nbsp;mM.</p>

{img("exchanges", "Top exchange fluxes over time")}

<h3>What picks up the slack when glucose cuts off?</h3>
<p>Mean flux in snapshots with glucose (&gt;1&nbsp;mM) vs. post-depletion
(&lt;0.01&nbsp;mM). DM OFF ramps other imports to feed the LP; DM ON
scales them down once mass balance bites.</p>

<p>Table below is the <strong>DM-OFF</strong> run.</p>

<table>
<tr><th>Molecule</th><th>pre-shutoff<br/>(count/step)</th>
    <th>post-shutoff<br/>(count/step)</th>
    <th>shift</th></tr>
{diff_rows}
</table>

{img("exchange_diff", "Pre vs post-shutoff flux comparison (both runs)")}

<h2 id="externals">External nutrient concentrations</h2>
<p><code>environment_update</code> drains every boundary species over
time. Ammonium typically depletes alongside glucose in the default 10 fL
environment; phosphate, sulfate, and trace ions decay more slowly.</p>

{img("externals_all", "All external concentrations over time")}

<h2 id="carbon">Carbon accounting</h2>
<p>Per-step C flux (mmol C). Imports weighted by per-molecule C counts;
biomass C from dry-mass delta at 48% C/g DCW. A red-filled region
(C_out &gt; C_in + biomass_C) flags pool-drain.</p>

{img("carbon_budget", "Carbon budget over time")}

<h2 id="growing">What's actually growing?</h2>
<p>Dry-mass composition. In DM OFF every pool keeps accumulating past
glucose depletion; in DM ON they flatten.</p>

{img("mass_composition", "Dry mass composition")}

<h2 id="mass-balance">Mass-balance check</h2>
<p>Closed cell invariant:
<code>Σ imports − Σ secretions = Σ dry-mass-gained</code> (fg).
Persistent negative deficit = LP synthesizing biomass faster than it
imports mass.</p>

{img("mass_balance", "Mass balance check")}

<h2 id="manufacture">How is biomass being manufactured?</h2>
<p><code>modular_fba</code>'s homeostatic objective uses quadratic slack
pseudofluxes (<code>quadFractionFromUnity</code>) bounded
<code>[−∞, +∞]</code>, so missed targets get filled in regardless of
whether carbon is available. The built-in <code>_massExchangeID</code>
pseudometabolite only audits exchange mass; bounding it to ~0 has no
effect on dry-mass growth — confirming mass enters via the slacks in
DM OFF. The DM-ON post-scale exists to correct this.</p>

{img("fba_mass_probe", "FBA exchange-mass vs dry mass")}

<h3 id="dark-matter">Dark-matter pool (diagnostic)</h3>
<p>Per step, <code>dark_matter += Σ Δbulk_mass − Σ Δexchange_mass</code>
using sim_data MWs (12,809 entries). Mass can't be created, so sustained
upward drift = LP manufacturing mass.</p>

{img("dark_matter", "Dark matter pool over time")}

<h3>Phase 2: enforcement at the metabolism step</h3>
<p>DM ON wraps the FBA solve:</p>
<pre>Δ cell_mass = Σ Δcount × MW   (from LP)
Δ boundary_mass = imports − secretions  (from exchange dict)
excess = Δ cell_mass − Δ boundary_mass
if excess > pool_fg: scale = (boundary + pool) / cell_mass</pre>
<p>All LP deltas are multiplied by <code>scale</code> before bulk write —
preserves stoichiometry, enforces <code>Δ cell_mass ≤ boundary + pool</code>.
<code>scale → 0</code> when carbon is gone and growth freezes.</p>

{img("dm_enforcement", "Dark matter enforcement effectiveness")}

{img("externals", "Top external concentrations over time (by change)")}

<div class="note">
<strong>Why does DM OFF keep growing after [GLC]ₑₓₜ hits zero?</strong>
<ol>
  <li>~17 molecules have unconstrained exchange bounds (AMMONIUM, O₂,
      Pi, SULFATE, WATER, K⁺, MG²⁺, NA⁺, Cl⁻, Fe, Mn, Zn, Ca, Co, Ni,
      selenocysteine, CO₂); the LP pulls harder on these when glucose hits 0.</li>
  <li>The homeostatic objective pins internal metabolite concentrations
      to targets that rescale with growth — LP keeps producing internals
      regardless of upstream carbon.</li>
  <li>No carbon-balance constraint on biomass: with GLC off, the LP finds
      carbon via CO₂ fixation / reverse-TCA at reduced but nonzero flux.</li>
</ol>
DM ON eliminates all three at once via the post-scale. A proper
mechanistic fix would bound the slack pseudofluxes or carbon-balance the
balance.
</div>

<h2 id="caglar">Calibration targets — Caglar 2017 doubling times</h2>
<p>Per-replicate exponential doubling times reported in MOESM56. The
simulation should hit these within the paper's 95% CI once
nutrient-specific parameterization (carbon source, Na⁺, Mg²⁺) is in
place. Currently the model is tuned only for the glucose base condition.</p>

<table>
<tr><th>Condition</th><th>Mean doubling time (min)</th><th>N</th></tr>
{caglar_rows}
</table>

<h2 id="todo">What this report does <em>not</em> cover yet</h2>
<ul>
  <li><strong>Non-glucose carbon sources:</strong> glycerol, gluconate,
  lactate. Requires media recipes + per-carbon v_max/K_m.</li>
  <li><strong>Ion stress:</strong> Na⁺, Mg²⁺ gradients from Caglar's
  figure 2B/C.</li>
  <li><strong>ppGpp-driven transition:</strong> mechanistic stationary-
  phase entry via the existing <code>ppgpp_initiation</code> pathway.</li>
</ul>

<footer>
  Generated by <code>nutrient_growth_report.py</code> &middot;
  branch <code>nutrient-growth</code>. Calibration data:
  Caglar MU et al. (2017) <em>Sci Rep</em> 7, 45303.
</footer>
</body></html>
"""

    with open(report_path, "w") as f:
        f.write(html)

    # Mirror to docs/ for GitHub Pages.
    docs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docs")
    if os.path.isdir(docs_dir):
        shutil.copy2(
            report_path, os.path.join(docs_dir, "nutrient_growth_report.html"))
    return report_path


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"simulated seconds (default {DEFAULT_DURATION})")
    parser.add_argument("--snapshot", type=int,
                        default=DEFAULT_SNAPSHOT_INTERVAL,
                        help=f"snapshot interval, s (default "
                             f"{DEFAULT_SNAPSHOT_INTERVAL})")
    parser.add_argument("--vmax", type=float, default=20.0,
                        help="MM v_max in mmol/gDCW/h (default 20.0)")
    parser.add_argument("--km", type=float, default=0.01,
                        help="MM K_m in mM (default 0.01 = 10 µM)")
    parser.add_argument("--env-volume-L", type=float,
                        default=DEFAULT_ENV_VOLUME_L, dest="env_volume_L",
                        help=f"environment volume per cell in litres "
                             f"(default {DEFAULT_ENV_VOLUME_L:g} = "
                             f"{DEFAULT_ENV_VOLUME_L*1e15:.0f} fL). "
                             f"Smaller → faster glucose depletion.")
    parser.add_argument("--single-mode", action="store_true",
                        help="Run only the dark-matter-OFF baseline. "
                             "Default is to run both baseline + enforced "
                             "and render a side-by-side comparison report.")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Nutrient-Growth Report ({args.duration}s, "
          f"{'single-mode' if args.single_mode else 'comparison mode'})")
    print(f"  MM glucose: v_max = {args.vmax} mmol/gDCW/h, K_m = {args.km} mM")
    print(f"  env volume: {args.env_volume_L:g} L "
          f"({args.env_volume_L*1e15:.1f} fL)")
    print("=" * 60)

    t0 = time.time()
    caglar = load_caglar_doubling_times()

    # Always enable the nutrient-growth feature set for the report — the
    # whole point is to show the MM + depletion + allowlist pipeline in
    # action. Dark matter is what distinguishes the two runs.
    os.environ["V2ECOLI_NUTRIENT_GROWTH"] = "1"
    os.environ["V2ECOLI_DARK_MATTER"] = "0"
    print("\n[1/2] Baseline run — dark matter OFF")
    data_baseline = run_single_cell(
        args.duration, args.snapshot, args.env_volume_L, label="baseline")

    data_enforced = None
    if not args.single_mode:
        os.environ["V2ECOLI_DARK_MATTER"] = "1"
        print("\n[2/2] Enforced run — dark matter ON (mass conservation)")
        data_enforced = run_single_cell(
            args.duration, args.snapshot, args.env_volume_L, label="enforced")
        os.environ["V2ECOLI_DARK_MATTER"] = "0"

    print("\nGenerating report HTML...")
    report_path = generate_report(
        data_baseline, caglar, args.duration,
        args.vmax, args.km,
        dark_matter=(data_enforced is not None),
        data_enforced=data_enforced)

    print(f"\nReport: {report_path}")
    print(f"Wall: {time.time() - t0:.0f}s")

    # Open in browser if possible.
    try:
        import subprocess as sp
        sp.run(["open", report_path], capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
