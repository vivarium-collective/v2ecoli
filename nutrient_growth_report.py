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

    # Carbon budget (emitted by v2ecoli/steps/listeners/carbon_budget.py).
    cb = agent.get("listeners", {}).get("carbon_budget", {}) or {}

    return {
        "time": t,
        "dry_mass": float(mass.get("dry_mass", 0.0)),
        "cell_mass": float(mass.get("cell_mass", 0.0)),
        "growth_rate": float(mass.get("instantaneous_growth_rate", 0.0)),
        "glc_ext_mM": _as_float(external.get("GLC")),
        "glc_bound_mmol_gdcw_h": glc_bound,
        # full external concentrations snapshot (for multi-species panels)
        "external_mM": {k: _as_float(v, 0.0) for k, v in external.items()},
        # per-step exchange counts snapshot (for flux panels)
        "exchange_counts": exchange_counts,
        # carbon accounting
        "c_in_mmol":   _as_float(cb.get("c_in_mmol"), 0.0),
        "c_out_mmol":  _as_float(cb.get("c_out_mmol"), 0.0),
        "c_net_mmol":  _as_float(cb.get("c_net_mmol"), 0.0),
        "cum_c_in":    _as_float(cb.get("cumulative_c_in_mmol"), 0.0),
        "cum_c_out":   _as_float(cb.get("cumulative_c_out_mmol"), 0.0),
        "biomass_c":   _as_float(cb.get("biomass_c_est_mmol_per_step"), 0.0),
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


def run_single_cell(duration: int, snapshot_interval: int, env_volume_L: float):
    """Run the baseline composite for `duration` seconds, snapshotting."""
    from v2ecoli.composite import make_composite
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    patched = _patch_env_volume(composite, env_volume_L)
    effective_env_vol = patched.env_volume_L if patched else None

    snaps = [_extract_snapshot(composite.state, 0.0)]
    total = 0.0
    t0 = time.time()
    while total < duration:
        chunk = min(snapshot_interval, duration - total)
        try:
            composite.run(chunk)
        except Exception as e:
            print(f"  sim error at t≈{total+chunk:.0f}s: "
                  f"{type(e).__name__}: {e}")
            break
        total += chunk
        snaps.append(_extract_snapshot(composite.state, total))
        print(f"  t={int(total)}s  dry={snaps[-1]['dry_mass']:.0f}fg  "
              f"glc={snaps[-1]['glc_ext_mM']}")
    wall = time.time() - t0
    return {
        "snapshots": snaps,
        "wall_time": wall,
        "sim_time": total,
        "env_volume_L": effective_env_vol,
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
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = [s["time"] / 60 for s in snaps]
    ax.plot(t, [s["dry_mass"] for s in snaps], color="#2563eb", label="dry mass")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Mass (fg)")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Single-cell growth")
    return _fig_to_b64(fig)


def plot_growth_rate(snaps):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = [s["time"] / 60 for s in snaps]
    gr = [s["growth_rate"] for s in snaps]
    ax.plot(t, gr, color="#16a34a")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("1/s")
    ax.set_title("Instantaneous growth rate")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def plot_glucose_trajectory(snaps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    t = [s["time"] / 60 for s in snaps]
    ax1.plot(t, [s["glc_ext_mM"] for s in snaps], color="#ea580c")
    ax1.set_xlabel("Time (min)"); ax1.set_ylabel("[GLC]ₑₓₜ (mM)")
    ax1.set_title("External glucose"); ax1.grid(alpha=0.3)
    ax2.plot(t, [s["glc_bound_mmol_gdcw_h"] for s in snaps], color="#9333ea")
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


def plot_exchange_fluxes(snaps, top_n: int = 10):
    """Main import-flux panel: top-N molecules by peak import rate,
    excluding glucose (which is plotted separately). Symlog y-axis;
    glucose shutoff marker. Raw exchange is negated so positive values
    on the plot mean import."""
    if not snaps:
        return None

    # Rank by peak import magnitude (largest negative raw exchange), skip
    # glucose (covered by its own panel) and any species that is purely
    # secreted (never imported).
    top = _top_by_metric(
        snaps,
        metric=lambda s: max((-x for x in s if x < 0), default=0),
        n=top_n,
        exclude={"GLC"})

    fig, ax = plt.subplots(figsize=(11, 5))
    seen_colors: dict = {}
    _draw_exchange_series(ax, snaps, top, seen_colors)

    t_shut = _glc_shutoff_min(snaps)
    if t_shut is not None:
        ax.axvline(t_shut, color="#dc2626", linestyle="--", alpha=0.6,
                   linewidth=1.2)
        ax.annotate(
            f"glucose depleted ({t_shut:.1f} min)",
            xy=(t_shut, 0), xytext=(t_shut + 0.2, 0),
            textcoords="data", color="#dc2626", fontsize=9, va="center")

    ax.set_title(f"Top {len(top)} imports, excluding glucose (symlog, by peak import rate)")
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

def generate_report(data, caglar, duration: int, vmax: float, km: float):
    from v2ecoli.library.repro_banner import banner_html
    repro = banner_html()

    snaps = data["snapshots"]
    final = snaps[-1] if snaps else {}
    initial_glc = snaps[0].get("glc_ext_mM") if snaps else None

    caglar_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v[0]:.1f}</td><td>{v[1]}</td></tr>"
        for k, v in sorted(caglar.items())
    )

    plots = {
        "mass": plot_mass(snaps) if snaps else None,
        "growth_rate": plot_growth_rate(snaps) if snaps else None,
        "glucose": plot_glucose_trajectory(snaps) if snaps else None,
        "mm": plot_mm_curve(vmax, km),
        "exchanges": plot_exchange_fluxes(snaps),
        "exchange_diff": plot_exchange_diff(snaps),
        "externals": plot_externals(snaps),
        "carbon_budget": plot_carbon_budget(snaps),
        "externals_all": plot_external_concentrations(snaps),
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
h2 {{ color: #1e40af; margin-top: 2em; }}
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

<h1>Nutrient-Growth Report</h1>
<p>Single-cell simulation exercising the extracted
<code>metabolic_kinetics</code> step with Michaelis-Menten glucose uptake.
Target: parameterize growth under varying nutrient conditions against
the Caglar et al. 2017 (<a href="https://doi.org/10.1038/srep45303"
target="_blank">srep45303</a>) multi-condition dataset
committed under <code>data/caglar2017/</code>.</p>

<h2>Run summary</h2>
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

<h2>Growth trajectory</h2>
{img("mass", "Dry mass vs time")}
{img("growth_rate", "Growth rate vs time")}

<h2>Glucose uptake kinetics</h2>
<p>With Michaelis-Menten enabled in <code>metabolic_kinetics.py</code>,
the glucose import bound is computed every step from
<code>boundary.external.GLC</code> instead of being fixed by a media lookup.
Parameters: <strong>v_max = {vmax} mmol/gDCW/h</strong>,
<strong>K_m = {km} mM</strong>.</p>

{img("mm", "MM curve")}
{img("glucose", "External glucose and applied uptake bound")}

<h2>Exchange fluxes</h2>
<p>Counts of molecules moving across the cell boundary per simulation
step (negative = cell imports, positive = cell secretes). Y-axis is
<em>symlog</em> so zero crossings and values spanning multiple decades
are both visible. Red dashed line marks when [GLC]ₑₓₜ first drops
below 0.01 mM.</p>

{img("exchanges", "Top exchange fluxes over time")}

<h3>What picks up the slack when glucose cuts off?</h3>
<p>Mean flux in snapshots with plenty of glucose (&gt;1 mM) vs. snapshots
after depletion (&lt;0.01 mM). Biggest absolute shifts first.</p>

<table>
<tr><th>Molecule</th><th>pre-shutoff<br/>(count/step)</th>
    <th>post-shutoff<br/>(count/step)</th>
    <th>shift</th></tr>
{diff_rows}
</table>

{img("exchange_diff", "Pre vs post-shutoff flux comparison")}

<h2>External nutrient concentrations</h2>
<p>Every boundary species <code>metabolism.py</code> exchanges with is
dynamically updated by <code>environment_update.py</code>. This panel
shows which nutrients drain first. <strong>Ammonium</strong> typically
runs out alongside glucose in the default 10 fL environment; phosphate,
sulfate, and trace ions decay more slowly.</p>

{img("externals_all", "All external concentrations over time")}

<h2>Carbon accounting</h2>
<p>Emitted by the <code>carbon_budget_listener</code> each step (values
in mmol C). Imports are weighted by per-molecule carbon counts from
<code>v2ecoli/library/carbon_counts.py</code>. Biomass C is estimated
from dry-mass delta at 48% C/g DCW.</p>

<p><strong>Reading the plot:</strong> in a carbon-balanced cell, the
green line (<em>C in</em>) should exceed the sum of red (<em>C out</em>)
and blue (<em>C into biomass</em>). When the red-filled region on the
right panel grows after glucose depletion, the cell is secreting more
carbon than it imports — i.e. draining internal pools to keep the
homeostatic objective happy. That pink area is the quantitative
signature of the "pool-drain" failure mode.</p>

{img("carbon_budget", "Carbon budget over time")}

{img("externals", "Top external concentrations over time (by change)")}

<div class="note">
<strong>Why does growth continue after [GLC]ₑₓₜ hits zero?</strong>
Three reasons stacked:
<ol>
  <li><strong>~17 molecules have <em>unconstrained</em> exchange bounds</strong>
      (AMMONIUM, OXYGEN-MOLECULE, Pi, SULFATE, WATER, K⁺, MG²⁺, NA⁺, Cl⁻,
      Fe²⁺/³⁺, Mn, Zn, Ca, Co, Ni, selenocysteine, CO₂).
      Once MM caps glucose uptake at 0, the LP keeps pulling on the others.</li>
  <li><strong>The homeostatic objective pins internal metabolite
      concentrations</strong> toward fixed targets (set by
      <code>getBiomassAsConcentrations</code>). As the cell grows, these
      targets rescale — the LP keeps producing internal metabolites even
      when their upstream carbon supply is off.</li>
  <li><strong>No carbon-balance constraint on biomass.</strong> The
      biomass reaction's coefficients assume carbon comes from somewhere;
      with GLC blocked the LP finds carbon in the CO₂ fixation /
      reverse-TCA routes that have catalysts present, at reduced efficiency
      but non-zero flux.</li>
</ol>
A proper exp→stat transition needs at least one of: a carbon-balanced
biomass objective (no carbon source → no biomass), a ppGpp-coupled
biomass downscaling, or hard flux bounds on the "promiscuous" inputs
above when their carbon-equivalent contribution would violate a mass
balance.
</div>

<h2>Calibration targets — Caglar 2017 doubling times</h2>
<p>Per-replicate exponential doubling times reported in MOESM56. The
simulation should hit these within the paper's 95% CI once
nutrient-specific parameterization (carbon source, Na⁺, Mg²⁺) is in
place. Currently the model is tuned only for the glucose base condition.</p>

<table>
<tr><th>Condition</th><th>Mean doubling time (min)</th><th>N</th></tr>
{caglar_rows}
</table>

<h2>What this report does <em>not</em> cover yet</h2>
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
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Nutrient-Growth Report ({args.duration}s)")
    print(f"  MM glucose: v_max = {args.vmax} mmol/gDCW/h, K_m = {args.km} mM")
    print(f"  env volume: {args.env_volume_L:g} L "
          f"({args.env_volume_L*1e15:.1f} fL)")
    print("=" * 60)

    t0 = time.time()
    data = run_single_cell(args.duration, args.snapshot, args.env_volume_L)
    caglar = load_caglar_doubling_times()
    report_path = generate_report(data, caglar, args.duration,
                                  args.vmax, args.km)

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
