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
SNAPSHOT_INTERVAL = 30  # seconds
CAGLAR_DOUBLING_TIMES_CSV = (
    "data/caglar2017/41598_2017_BFsrep45303_MOESM56_ESM.csv")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _extract_snapshot(state, t):
    """Pull the metrics this report cares about out of a composite state."""
    agent = state.get("agents", {}).get("0", {})
    mass = agent.get("listeners", {}).get("mass", {})
    boundary = agent.get("boundary", {})
    external = boundary.get("external", {}) if isinstance(boundary, dict) else {}
    env_exch = (agent.get("listeners", {})
                     .get("fba_results", {})
                     .get("external_exchange_fluxes"))
    exchange_data = (agent.get("environment", {})
                          .get("exchange_data", {}))
    constrained = exchange_data.get("constrained", {}) if isinstance(
        exchange_data, dict) else {}

    # Glucose external concentration — plain float (mM) in boundary.external.
    glc_ext = external.get("GLC", None)
    if hasattr(glc_ext, "asNumber"):
        glc_ext = float(glc_ext.asNumber())
    elif glc_ext is not None:
        glc_ext = float(glc_ext)

    # Glucose uptake bound currently applied (mmol/gDCW/h) — Unum in the
    # store because of the `node` schema. Strip to scalar for plotting.
    glc_bound = constrained.get("GLC[p]", None)
    if hasattr(glc_bound, "asNumber"):
        glc_bound = float(glc_bound.asNumber())
    elif glc_bound is not None:
        try:
            glc_bound = float(glc_bound)
        except Exception:
            glc_bound = None

    return {
        "time": t,
        "dry_mass": float(mass.get("dry_mass", 0.0)),
        "cell_mass": float(mass.get("cell_mass", 0.0)),
        "growth_rate": float(mass.get("instantaneous_growth_rate", 0.0)),
        "glc_ext_mM": glc_ext,
        "glc_bound_mmol_gdcw_h": glc_bound,
    }


def run_single_cell(duration: int, snapshot_interval: int):
    """Run the baseline composite for `duration` seconds, snapshotting."""
    from v2ecoli.composite import make_composite
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)

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
    return {"snapshots": snaps, "wall_time": wall, "sim_time": total}


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
    }

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

<div class="wip">
<strong>Work in progress:</strong> environment depletion feedback is not
yet wired — <code>boundary.external.GLC</code> stays at its initial
media value, so the left panel above is effectively flat and the uptake
bound (right panel) stays saturated. The next commit on the
<code>nutrient-growth</code> branch will subtract cellular uptake flux
from the external pool so the model naturally transitions into
stationary phase as glucose runs out.
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
  <li><strong>Environment depletion:</strong> cell uptake currently does
  not subtract from <code>boundary.external</code>. Next commit.</li>
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
    parser.add_argument("--duration", type=int, default=2520)
    parser.add_argument("--snapshot", type=int, default=SNAPSHOT_INTERVAL)
    parser.add_argument("--vmax", type=float, default=20.0,
                        help="MM v_max in mmol/gDCW/h (default 20.0)")
    parser.add_argument("--km", type=float, default=0.01,
                        help="MM K_m in mM (default 0.01 = 10 µM)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Nutrient-Growth Report ({args.duration}s)")
    print(f"  MM glucose: v_max = {args.vmax} mmol/gDCW/h, K_m = {args.km} mM")
    print("=" * 60)

    t0 = time.time()
    data = run_single_cell(args.duration, args.snapshot)
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
