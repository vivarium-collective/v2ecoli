"""
Structured composite comparison report
=======================================

A composite-agnostic comparison harness. Runs any set of engines —
vEcoli (the vivarium reference) and/or any registered v2ecoli composite by
name (baseline, millard_pdmp_baseline, …) — each in its own subprocess (to
avoid type/registry conflicts), then extracts a structured set of metrics and
renders a comparison table to HTML.

Unlike reports/v1_v2_report.py (a fixed 3-way vEcoli-vs-v2ecoli viz), this is
parameterized: name the engines on the command line and get a metrics table
plus per-metric trajectories.

Metrics per engine (derived from the shared snapshot schema each runner emits):
  * performance: load time, wall time, sim time reached, realtime factor
  * runtime:     per-step wall ms/sim-s (mean/median/p95), peak RSS, step-time
                 series over the run  (v2ecoli runners only)
  * growth:      initial / final dry mass, fold-change, mean growth rate
  * composition: final protein / RNA (r/t/m split) / DNA / small-molecule mass,
                 cell volume
  * metabolism:  FBA objective value (pinned vs unpinned)
  * dynamics:    chromosomes, replication forks at end
  * fba-bridge:  flux-pin diagnostics (reactions pinned / relaxed-as-infeasible,
                 central-flux magnitudes) for the millard_fba_bridge_harness
Each metric column shows Δ% vs the first engine (the reference) and, where a
tolerance applies, a divergence badge (within tol / drift / mismatch).
Behavioral observables are also drawn as shared-axis overlays (all engines on
one plot) so absolute divergence is directly readable.

Usage:
    # vEcoli vs v2ecoli baseline vs the FBA-bridge harness, full cycle:
    python reports/composite_comparison.py \\
        --engines vecoli baseline millard_fba_bridge_harness --duration 2520

    # quick smoke (no vEcoli; the two fast v2ecoli engines):
    python reports/composite_comparison.py \\
        --engines baseline millard_fba_bridge_harness --duration 60

Engine tokens:
    vecoli                 -> scripts/run_vecoli_v1.py   (vivarium reference)
    vecoli_composite       -> scripts/run_vecoli_composite.py
    <any other token>      -> scripts/run_v2.py <token>  (a v2ecoli composite)
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess as sp
import sys
import time

REPORT_DIR = "out/comparison"

# Tokens that map to the dedicated vEcoli runners; anything else is treated as
# a v2ecoli composite name passed to run_v2.py.
_VECOLI_RUNNERS = {
    "vecoli": "scripts/run_vecoli_v1.py",
    "vecoli_composite": "scripts/run_vecoli_composite.py",
}


def _launch(token: str, duration: int, interval: int, base: str, rpath: str):
    """Start a subprocess runner for one engine token. Returns Popen|None."""
    if token in _VECOLI_RUNNERS:
        script = os.path.join(base, _VECOLI_RUNNERS[token])
        argv = [sys.executable, script, str(duration), str(interval), rpath]
    else:
        script = os.path.join(base, "scripts/run_v2.py")
        # 4th arg = composite name, so one runner serves every v2ecoli composite
        argv = [sys.executable, script, str(duration), str(interval), rpath, token]
    if not os.path.exists(script):
        print(f"  {token}: runner not found ({script})")
        return None
    return sp.Popen(argv)


def _collect(token: str, proc, rpath: str) -> dict:
    if proc is None:
        return {"engine": f"{token} (no runner)", "snapshots": []}
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(rpath):
        return {"engine": f"{token} (FAILED rc={proc.returncode})", "snapshots": []}
    with open(rpath) as f:
        data = json.load(f)
    os.unlink(rpath)
    return data


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def _pct(vals, p):
    """Linear-interpolated percentile of a list (p in [0,100])."""
    xs = sorted(v for v in vals if v == v)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _metrics(data: dict) -> dict:
    snaps = data.get("snapshots") or []
    first = snaps[0] if snaps else {}
    last = snaps[-1] if snaps else {}

    def g(s, k):
        return float(s.get(k, 0) or 0)

    dry0, dry1 = g(first, "dry_mass"), g(last, "dry_mass")
    rates = [g(s, "instantaneous_growth_rate") for s in snaps]
    rates = [r for r in rates if r]

    # Per-step runtime distribution (only v2ecoli runners emit step_times).
    steps = data.get("step_times") or []
    step_ms = [float(s.get("ms_per_sim_s", 0) or 0) for s in steps]
    step_ms = [m for m in step_ms if m]

    # FBA-bridge diagnostics are present only on snapshots from the bridge
    # harness; fall back to 0 / not-applicable for every other engine.
    n_relaxed_series = [g(s, "n_relaxed") for s in snaps if "n_relaxed" in s]
    has_bridge = any("n_pinned" in s for s in snaps)

    return {
        "engine": data.get("engine", "?"),
        # performance
        "load_s": data.get("load_time", 0.0),
        "wall_s": data.get("wall_time", 0.0),
        "sim_s": data.get("sim_time", 0.0),
        "realtime_x": data.get("speed", 0.0),
        # runtime distribution (per-step)
        "step_ms_mean": (sum(step_ms) / len(step_ms)) if step_ms else 0.0,
        "step_ms_median": _pct(step_ms, 50),
        "step_ms_p95": _pct(step_ms, 95),
        "peak_rss_mb": float(data.get("peak_rss_mb", 0.0) or 0.0),
        # growth
        "dry_mass_0": dry0,
        "dry_mass_f": dry1,
        "dry_fold": (dry1 / dry0) if dry0 else 0.0,
        "growth_rate": (sum(rates) / len(rates)) if rates else 0.0,
        # composition (final)
        "protein_mass_f": g(last, "protein_mass"),
        "rna_mass_f": g(last, "rna_mass"),
        "rrna_mass_f": g(last, "rRna_mass"),
        "trna_mass_f": g(last, "tRna_mass"),
        "mrna_mass_f": g(last, "mRna_mass"),
        "dna_mass_f": g(last, "dna_mass"),
        "smol_mass_f": g(last, "smallMolecule_mass"),
        "water_mass_f": g(last, "water_mass"),
        "volume_f": g(last, "volume"),
        # metabolism (final)
        "fba_obj_f": g(last, "fba_objective"),
        # molecular species
        "bulk_total_f": g(last, "bulk_total"),
        "bulk_nonzero_f": g(last, "bulk_species_nonzero"),
        "bulk_n_species": g(last, "bulk_n_species"),
        # dynamics (final)
        "n_chrom_f": g(last, "n_chromosomes"),
        "n_forks_f": g(last, "n_forks"),
        "n_snapshots": len(snaps),
        # FBA-bridge diagnostics (bridge harness only)
        "_has_bridge": has_bridge,
        "n_pinned_f": g(last, "n_pinned"),
        "n_relaxed_f": g(last, "n_relaxed"),
        "n_relaxed_max": max(n_relaxed_series) if n_relaxed_series else 0.0,
        "cflux_nonzero_f": g(last, "central_flux_nonzero"),
        "cflux_absmean_f": g(last, "central_flux_absmean"),
        "pin_nonzero_f": g(last, "pin_n_nonzero"),
        # full final unique-molecule counts (per type) + the snapshot list for
        # trajectory rendering
        "unique_final": (last.get("unique_counts") or {}),
        "_snaps": snaps,
        "_steps": steps,
        # per-reaction time-mean FBA flux + labels (absent on engines without a
        # metabolism listener / flux capture)
        "_flux_ids": data.get("base_reaction_ids") or [],
        "_flux_mean": data.get("base_reaction_flux_mean") or [],
    }


# (key, label, fmt, lower-is-better-for-Δ?, verdict_tol) — Δ% computed vs the
# reference col. verdict_tol (fractional, e.g. 0.05 = 5%): when set, a
# divergence badge is shown — within_tol (|Δ|≤tol), drift (≤3·tol), mismatch.
# None tol = no verdict (performance rows, or naturally-divergent counters).
_ROWS = [
    ("__perf__", "Performance", None, None, None),
    ("load_s", "Load time (s)", "{:.1f}", None, None),
    ("wall_s", "Wall time (s)", "{:.1f}", None, None),
    ("sim_s", "Sim time reached (s)", "{:.0f}", None, None),
    ("realtime_x", "Realtime factor (×)", "{:.1f}", None, None),
    ("__runtime__", "Runtime, per-step (v2ecoli engines)", None, None, None),
    ("step_ms_mean", "Step time, mean (ms / sim-s)", "{:.1f}", None, None),
    ("step_ms_median", "Step time, median (ms / sim-s)", "{:.1f}", None, None),
    ("step_ms_p95", "Step time, p95 (ms / sim-s)", "{:.1f}", None, None),
    ("peak_rss_mb", "Peak memory, RSS (MB)", "{:.0f}", None, None),
    ("__growth__", "Growth", None, None, None),
    ("dry_mass_0", "Dry mass, initial (fg)", "{:.1f}", None, 0.02),
    ("dry_mass_f", "Dry mass, final (fg)", "{:.1f}", True, 0.05),
    ("dry_fold", "Dry-mass fold change", "{:.4f}", True, 0.05),
    ("growth_rate", "Mean growth rate (1/s)", "{:.3e}", True, 0.05),
    ("__comp__", "Composition (final)", None, None, None),
    ("protein_mass_f", "Protein mass (fg)", "{:.1f}", True, 0.05),
    ("rna_mass_f", "RNA mass, total (fg)", "{:.1f}", True, 0.05),
    ("rrna_mass_f", "  rRNA mass (fg)", "{:.1f}", True, 0.05),
    ("trna_mass_f", "  tRNA mass (fg)", "{:.1f}", True, 0.05),
    ("mrna_mass_f", "  mRNA mass (fg)", "{:.2f}", True, 0.08),
    ("dna_mass_f", "DNA mass (fg)", "{:.1f}", True, 0.05),
    ("smol_mass_f", "Small-molecule mass (fg)", "{:.1f}", True, 0.05),
    ("water_mass_f", "Water mass (fg)", "{:.1f}", True, 0.05),
    ("volume_f", "Cell volume (fL)", "{:.3f}", True, 0.05),
    ("__energy__", "Metabolism (final)", None, None, None),
    ("fba_obj_f", "FBA objective value", "{:.3e}", None, 0.05),
    ("__species__", "Molecular species (final)", None, None, None),
    ("bulk_total_f", "Bulk molecules, total count", "{:.0f}", True, 0.05),
    ("bulk_nonzero_f", "Bulk species present (count>0)", "{:.0f}", True, 0.02),
    ("bulk_n_species", "Bulk species, distinct", "{:.0f}", None, None),
    ("__dyn__", "Unique molecules (final)", None, None, None),
    ("n_chrom_f", "Chromosomes", "{:.0f}", None, None),
    ("n_forks_f", "Replication forks", "{:.0f}", None, None),
    ("n_snapshots", "Snapshots", "{:.0f}", None, None),
]

# Tolerance-badge presentation.
_VERDICTS = {
    "within_tol": ("within tol", "#15803d", "#dcfce7"),
    "drift":      ("drift",      "#b45309", "#fef3c7"),
    "mismatch":   ("mismatch",   "#b91c1c", "#fee2e2"),
    "na":         ("n/a",        "#94a3b8", "#f1f5f9"),
}


def _verdict(v, rv, tol):
    """Classify engine value v against reference rv at fractional tolerance tol."""
    if tol is None:
        return None
    if not rv:
        return "na"
    d = abs(v - rv) / abs(rv)
    if d <= tol:
        return "within_tol"
    if d <= 3 * tol:
        return "drift"
    return "mismatch"


def _verdict_badge(verdict):
    if not verdict:
        return ""
    label, fg, bg = _VERDICTS[verdict]
    return (f"<span class='verdict' style='color:{fg};background:{bg}'>"
            f"{label}</span>")

CSS = """
body{margin:0;font-family:-apple-system,system-ui,sans-serif;color:#1e293b;background:#f8fafc}
header{background:#0f172a;color:#f1f5f9;padding:24px 36px}
header h1{margin:0 0 6px;font-size:22px} header p{margin:0;color:#94a3b8;font-size:13px}
main{max-width:1100px;margin:0 auto;padding:28px 36px 70px}
h2{font-size:16px;margin:32px 0 6px;color:#0f172a}
.note{font-size:12px;color:#64748b;margin:0 0 10px}
svg{display:block}
table{border-collapse:collapse;width:100%;background:#fff;font-size:14px;margin-bottom:8px;
  box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:8px;overflow:hidden}
th,td{padding:8px 14px;text-align:right;border-bottom:1px solid #eef2f7}
th:first-child,td:first-child{text-align:left}
thead th{background:#1e293b;color:#f1f5f9;font-weight:600;position:sticky;top:0}
tr.group td{background:#eef2ff;color:#3730a3;font-weight:700;text-transform:uppercase;
  font-size:11px;letter-spacing:.04em}
td.metric{color:#475569}
.delta{font-size:11px;color:#64748b;margin-left:6px}
.delta.up{color:#15803d} .delta.down{color:#b91c1c}
.ref{font-size:11px;color:#94a3b8}
.verdict{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;
  border-radius:9px;margin-left:6px;vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#475569;margin:0 0 14px}
.legend .sw{display:inline-block;width:22px;height:3px;border-radius:2px;
  margin-right:6px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:12px 14px}
.card h3{margin:0 0 2px;font-size:13px;color:#0f172a}
.card .sub{margin:0 0 6px;font-size:11px;color:#94a3b8}
footer{max-width:1100px;margin:0 auto;padding:0 36px 40px;color:#64748b;font-size:12px}
"""

# Per-engine palette, reused by every multi-engine plot + the legend.
PALETTE = ["#3730a3", "#b45309", "#15803d", "#9d174d", "#0e7490", "#6d28d9"]


def _sparkline(snaps, key, w=260, h=44, color="#3730a3"):
    """Inline SVG sparkline of snapshot[key] over time."""
    pts = [(float(s.get("time", 0)), float(s.get(key, 0) or 0)) for s in snaps]
    pts = [(t, v) for t, v in pts if v == v]  # drop NaN
    if len(pts) < 2:
        return "<span class='ref'>—</span>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0; dy = (y1 - y0) or 1.0
    coords = " ".join(
        f"{(t-x0)/dx*(w-4)+2:.1f},{h-2-((v-y0)/dy*(h-6)):.1f}" for t, v in pts)
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='1.5' "
            f"points='{coords}'/></svg>")


def _trajectory_section(cols):
    """Mass-over-time sparklines per engine, for the key mass components."""
    palette = ["#3730a3", "#b45309", "#15803d", "#9d174d", "#0e7490"]
    traj_keys = [
        ("dry_mass", "Dry mass"), ("protein_mass", "Protein"),
        ("rna_mass", "RNA"), ("dna_mass", "DNA"),
        ("smallMolecule_mass", "Small molecules"), ("volume", "Volume"),
        ("bulk_total", "Bulk total count"),
    ]
    head = "<tr><th>Trajectory</th>" + "".join(
        f"<th>{html.escape(c['engine'])}</th>" for c in cols) + "</tr>"
    rows = []
    for key, label in traj_keys:
        cells = [f"<td class='metric'>{html.escape(label)}</td>"]
        for i, c in enumerate(cols):
            spark = _sparkline(c.get("_snaps", []), key, color=palette[i % len(palette)])
            cells.append(f"<td>{spark}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<h2>Trajectories over time</h2>"
            f"<p class='note'>Each line spans t=0 → end of run, auto-scaled per "
            f"cell (shapes comparable, absolute heights not).</p>"
            f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>")


def _species_section(cols, ref_idx=0):
    """Per-type unique-molecule count comparison across engines."""
    all_types = []
    seen = set()
    for c in cols:
        for t in c.get("unique_final", {}):
            if t not in seen:
                seen.add(t); all_types.append(t)
    if not all_types:
        return ""
    ref = cols[ref_idx]
    head = "<tr><th>Unique molecule</th>" + "".join(
        f"<th>{html.escape(c['engine'])}</th>" for c in cols) + "</tr>"
    rows = []
    for t in sorted(all_types):
        cells = [f"<td class='metric'>{html.escape(t)}</td>"]
        rv = float(ref.get("unique_final", {}).get(t, 0) or 0)
        for i, c in enumerate(cols):
            v = float(c.get("unique_final", {}).get(t, 0) or 0)
            delta = ""
            if i != ref_idx and rv:
                pct = 100.0 * (v - rv) / rv
                cls = "up" if pct >= 0 else "down"
                delta = f"<span class='delta {cls}'>{pct:+.0f}%</span>"
            cells.append(f"<td>{v:.0f}{delta}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<h2>Unique molecular species (final counts)</h2>"
            f"<p class='note'>Active count per unique-molecule type at end of "
            f"run; Δ% vs the reference engine.</p>"
            f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>")


def _legend(cols):
    items = "".join(
        f"<span><span class='sw' style='background:{PALETTE[i % len(PALETTE)]}'></span>"
        f"{html.escape(c['engine'])}</span>"
        for i, c in enumerate(cols))
    return f"<div class='legend'>{items}</div>"


def _multiline_svg(series, w=300, h=120, baseline_zero=True):
    """Shared-axis multi-line SVG. `series` is a list aligned to engine index;
    each element is a list of (x, y) points (or None/empty to skip). All lines
    share one auto-scaled x/y range so absolute divergence is directly visible.
    """
    gxs, gys = [], []
    for pts in series:
        for t, v in (pts or []):
            if v == v:
                gxs.append(t); gys.append(v)
    if len(gxs) < 2:
        return "<span class='ref'>—</span>", (0.0, 0.0)
    x0, x1 = min(gxs), max(gxs)
    y0, y1 = min(gys), max(gys)
    if baseline_zero:
        y0 = min(y0, 0.0)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    pad = 7
    lines = []
    for i, pts in enumerate(series):
        pts = [(t, v) for t, v in (pts or []) if v == v]
        if len(pts) < 2:
            continue
        coords = " ".join(
            f"{pad+(t-x0)/dx*(w-2*pad):.1f},{h-pad-((v-y0)/dy*(h-2*pad)):.1f}"
            for t, v in pts)
        lines.append(f"<polyline fill='none' stroke='{PALETTE[i % len(PALETTE)]}' "
                     f"stroke-width='1.5' points='{coords}'/>")
    svg = (f"<svg width='100%' height='{h}' viewBox='0 0 {w} {h}' "
           f"preserveAspectRatio='none'>"
           f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' "
           f"stroke='#e2e8f0'/>{''.join(lines)}</svg>")
    return svg, (y0, y1)


def _overlay_card(cols, key, title, sub, fmt="{:.3g}"):
    series = [[(float(s.get("time", 0)), float(s.get(key, 0) or 0))
               for s in c.get("_snaps", [])] for c in cols]
    svg, (lo, hi) = _multiline_svg(series)
    rng = f"range: {fmt.format(lo)} … {fmt.format(hi)}"
    return (f"<div class='card'><h3>{html.escape(title)}</h3>"
            f"<p class='sub'>{html.escape(sub)} · {html.escape(rng)}</p>{svg}</div>")


def _overlay_section(cols):
    """Shared-axis overlays — every engine on one plot per observable, so
    behavioral divergence is read in absolute terms (not per-cell auto-scaled)."""
    specs = [
        ("dry_mass", "Dry mass", "fg"),
        ("protein_mass", "Protein mass", "fg"),
        ("rna_mass", "RNA mass", "fg"),
        ("dna_mass", "DNA mass", "fg"),
        ("smallMolecule_mass", "Small molecules", "fg"),
        ("volume", "Cell volume", "fL"),
        ("instantaneous_growth_rate", "Growth rate", "1/s"),
        ("bulk_total", "Bulk total count", "molecules"),
    ]
    cards = "".join(_overlay_card(cols, k, t, u) for k, t, u in specs)
    return (f"<h2>Behavioral overlays (shared axis)</h2>"
            f"<p class='note'>All engines on one auto-scaled axis per observable; "
            f"line height is comparable across engines. Divergence between curves "
            f"is real behavioral divergence.</p>{_legend(cols)}"
            f"<div class='grid'>{cards}</div>")


def _runtime_section(cols):
    """Per-step wall-time series (ms per sim-second) over the run, shared axis —
    surfaces realtime-factor drift and per-step cost, not just the aggregate."""
    series = [[(float(s.get("time", 0)), float(s.get("ms_per_sim_s", 0) or 0))
               for s in c.get("_steps", [])] for c in cols]
    if not any(len(s) >= 2 for s in series):
        return ""  # no engine emitted per-step timing (e.g. vEcoli-only run)
    svg, (lo, hi) = _multiline_svg(series, w=520, h=150)
    return (f"<h2>Per-step runtime (wall ms / sim-second)</h2>"
            f"<p class='note'>Lower is faster. A rising curve = the cell getting "
            f"more expensive per sim-second as it grows (more molecules to update). "
            f"Range {lo:.0f}…{hi:.0f} ms/sim-s. vEcoli engines do not emit per-step "
            f"timing and are absent here.</p>{_legend(cols)}"
            f"<div class='card'>{svg}</div>")


_BRIDGE_ROWS = [
    ("n_pinned_f", "Reactions pinned (final tick)", "{:.0f}"),
    ("n_relaxed_f", "Pins relaxed as infeasible (final)", "{:.0f}"),
    ("n_relaxed_max", "Pins relaxed, peak over run", "{:.0f}"),
    ("pin_nonzero_f", "Non-zero pin targets (final)", "{:.0f}"),
    ("cflux_nonzero_f", "Central fluxes non-zero (final)", "{:.0f}"),
    ("cflux_absmean_f", "Central flux |mean| (mM/s, final)", "{:.3e}"),
    ("fba_obj_f", "FBA objective value (final)", "{:.3e}"),
]


def _bridge_section(cols):
    """FBA-bridge flux-pin diagnostics — what the Millard→FBA coupling actually
    does each tick. Only engines that carry bridge stores get a column; the
    reference (baseline / vEcoli, unpinned) shows the contrast where available."""
    bridge_cols = [c for c in cols if c.get("_has_bridge")]
    if not bridge_cols:
        return ""
    head = "<tr><th>Bridge diagnostic</th>" + "".join(
        f"<th>{html.escape(c['engine'])}</th>" for c in cols) + "</tr>"
    rows = []
    for key, label, fmt in _BRIDGE_ROWS:
        cells = [f"<td class='metric'>{html.escape(label)}</td>"]
        for c in cols:
            if key == "fba_obj_f":
                # every metabolic engine has an objective; show it for all
                v = c.get(key, 0.0)
                txt = fmt.format(v) if v else "<span class='ref'>—</span>"
            elif c.get("_has_bridge"):
                v = c.get(key, 0.0)
                txt = fmt.format(v)
            else:
                txt = "<span class='ref'>n/a</span>"
            cells.append(f"<td>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<h2>FBA-bridge flux-pin diagnostics</h2>"
            f"<p class='note'>The Millard 2017 central-carbon ODE feeds "
            f"<code>fba-flux-coupler</code>, which pins v2ecoli FBA reactions to "
            f"the ODE-derived fluxes; <code>ecoli-metabolism</code> relaxes any pin "
            f"that makes the LP infeasible. <code>n/a</code> = engine has no bridge "
            f"(unpinned FBA). Comparing the FBA objective pinned vs unpinned shows "
            f"the cost the flux-pin imposes on the metabolic solution.</p>"
            f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>")


# Materiality gates for the per-reaction flux diff: ignore numerically
# negligible reactions (abs floor) and require a real relative gap so float
# noise / tiny fluxes don't dominate the ranking.
_FLUX_ABS_FLOOR = 1e-3   # mmol/gDCW/h — below this a reaction is ~inactive
_FLUX_REL_GATE = 0.25    # ≥25% relative difference to count as "material"


def _ec_bucket(rid):
    """Coarse pathway bucket from the EC-class prefix of an EC-style id;
    EcoCyc frame ids (RXN-…, TRANS-RXN…) fall through to 'named reaction'."""
    parts = rid.split("-")[0].split(".")
    if parts and parts[0].isdigit():
        return {
            "1": "oxidoreductase (redox)", "2": "transferase", "3": "hydrolase",
            "4": "lyase", "5": "isomerase", "6": "ligase",
            "7": "translocase/transport",
        }.get(parts[0], "other (EC)")
    if rid.startswith("TRANS-RXN") or rid.startswith("RXN0-") and "TRANS" in rid:
        return "transport (named)"
    return "named reaction"


def _flux_divergence_section(cols, ref_idx=0):
    """Per-reaction FBA flux divergence across engines — opens up the single
    aggregate 'FBA objective' / 'central flux |mean|' numbers into which
    specific base reactions (and pathways) actually carry different flux.

    Each engine that captured per-reaction flux gets a column; divergence is
    measured vs the reference engine. Ranked by |Δflux| vs reference.
    """
    fcols = [(i, c) for i, c in enumerate(cols)
             if c.get("_flux_ids") and c.get("_flux_mean")]
    if len(fcols) < 2:
        return ""  # need ≥2 engines with per-reaction flux to compare

    flux_is = [i for i, _ in fcols]
    ref_i = ref_idx if ref_idx in flux_is else flux_is[0]
    refc = cols[ref_i]
    ref_map = dict(zip(refc["_flux_ids"], refc["_flux_mean"]))
    maps = {i: dict(zip(c["_flux_ids"], c["_flux_mean"])) for i, c in fcols}
    other_is = [i for i in flux_is if i != ref_i]

    # union of reaction ids (reference order first, then any extras)
    all_ids, seen = list(refc["_flux_ids"]), set(refc["_flux_ids"])
    for i in other_is:
        for r in cols[i]["_flux_ids"]:
            if r not in seen:
                seen.add(r); all_ids.append(r)

    recs = []
    for r in all_ids:
        rv = ref_map.get(r)
        maxabs = maxrel = 0.0
        flip = False
        for i in other_is:
            v = maps[i].get(r)
            if v is None or rv is None:
                continue
            d = abs(v - rv)
            maxabs = max(maxabs, d)
            maxrel = max(maxrel, d / max(abs(v), abs(rv), 1e-12))
            if (rv > _FLUX_ABS_FLOOR and v < -_FLUX_ABS_FLOOR) or \
               (rv < -_FLUX_ABS_FLOOR and v > _FLUX_ABS_FLOOR):
                flip = True
        recs.append({
            "rid": r, "ref": rv, "vals": {i: maps[i].get(r) for i in other_is},
            "bucket": _ec_bucket(r), "maxabs": maxabs, "maxrel": maxrel,
            "flip": flip,
            "material": maxabs >= _FLUX_ABS_FLOOR and maxrel >= _FLUX_REL_GATE,
        })

    material = sorted([x for x in recs if x["material"]],
                      key=lambda x: x["maxabs"], reverse=True)
    flips = [x for x in material if x["flip"]]

    # structural set diff: reactions in the reference network missing from each
    # other engine, and vice-versa
    ref_set = set(refc["_flux_ids"])
    struct = []
    for i in other_is:
        s = set(cols[i]["_flux_ids"])
        struct.append((i, sorted(ref_set - s), sorted(s - ref_set)))

    # pathway roll-up: Σ|Δflux| by EC bucket
    bucket = {}
    for x in material:
        b = bucket.setdefault(x["bucket"], [0, 0.0])
        b[0] += 1; b[1] += x["maxabs"]
    pr = sorted(bucket.items(), key=lambda kv: kv[1][1], reverse=True)

    def fmt(v):
        return "—" if v is None else f"{v:.3g}"

    # ---- structure summary ----
    struct_lines = [f"<li>base reactions: " + ", ".join(
        f"{html.escape(cols[i]['engine'])} <b>{len(cols[i]['_flux_ids'])}</b>"
        for i in flux_is) + "</li>"]
    for i, ref_missing, extra in struct:
        en = html.escape(cols[i]["engine"])
        rn = html.escape(refc["engine"])
        bits = []
        if ref_missing:
            shown = ", ".join(f"<code>{html.escape(x)}</code>" for x in ref_missing[:8])
            more = f" +{len(ref_missing)-8} more" if len(ref_missing) > 8 else ""
            bits.append(f"{len(ref_missing)} in {rn} not in {en} ({shown}{more})")
        if extra:
            shown = ", ".join(f"<code>{html.escape(x)}</code>" for x in extra[:8])
            more = f" +{len(extra)-8} more" if len(extra) > 8 else ""
            bits.append(f"{len(extra)} in {en} not in {rn} ({shown}{more})")
        struct_lines.append(f"<li>{en} vs {rn}: " +
                            ("; ".join(bits) if bits else "identical reaction set") +
                            "</li>")
    n_shared = len([x for x in recs if x["ref"] is not None
                    and all(x["vals"][i] is not None for i in other_is)])
    struct_html = (f"<ul class='note' style='margin:0 0 12px;padding-left:18px'>"
                   f"{''.join(struct_lines)}</ul>"
                   f"<p class='note'><b>{len(material)}</b> reactions carry "
                   f"materially different flux (|Δ| ≥ {_FLUX_ABS_FLOOR:g}, "
                   f"rel ≥ {_FLUX_REL_GATE:.0%}) vs "
                   f"{html.escape(refc['engine'])} (the reference).</p>")

    # ---- pathway roll-up ----
    pr_rows = "".join(
        f"<tr><td class='metric'>{html.escape(name)}</td><td>{n}</td>"
        f"<td>{tot:.3g}</td></tr>" for name, (n, tot) in pr[:14])
    pr_html = (f"<table><thead><tr><th>Pathway / EC bucket</th>"
               f"<th>#reactions</th><th>Σ|Δflux|</th></tr></thead>"
               f"<tbody>{pr_rows}</tbody></table>")

    # ---- sign-flip table ----
    if flips:
        head = "<tr><th>Reaction (direction differs)</th>" + "".join(
            f"<th>{html.escape(cols[i]['engine'])}</th>" for i in flux_is) + "</tr>"
        frows = []
        for x in flips[:20]:
            cells = [f"<td class='metric'><code>{html.escape(x['rid'])}</code></td>",
                     f"<td>{fmt(x['ref'])}</td>"]
            cells += [f"<td>{fmt(x['vals'][i])}</td>" for i in other_is]
            frows.append("<tr>" + "".join(cells) + "</tr>")
        flip_html = (f"<h3 style='font-size:14px;margin:18px 0 4px'>"
                     f"Sign-flipped reactions ({len(flips)}) — flux reverses "
                     f"direction</h3><table><thead>{head}</thead>"
                     f"<tbody>{''.join(frows)}</tbody></table>")
    else:
        flip_html = ("<p class='note'>No reaction reverses direction between the "
                     "engines.</p>")

    # ---- top-N reactions ----
    head = ("<tr><th>Reaction</th><th>Pathway/EC</th>"
            + "".join(f"<th>{html.escape(cols[i]['engine'])}"
                      + ("<div class='ref'>reference</div>" if i == ref_i else "")
                      + "</th>" for i in flux_is)
            + "<th>Δ vs ref</th><th>rel</th></tr>")
    trows = []
    for x in material[:40]:
        cells = [f"<td class='metric'><code>{html.escape(x['rid'])}</code></td>",
                 f"<td class='metric'>{html.escape(x['bucket'])}</td>",
                 f"<td>{fmt(x['ref'])}</td>"]
        cells += [f"<td>{fmt(x['vals'][i])}</td>" for i in other_is]
        cells += [f"<td>{x['maxabs']:.3g}</td>", f"<td>{x['maxrel']:.0%}</td>"]
        trows.append("<tr>" + "".join(cells) + "</tr>")
    top_html = (f"<h3 style='font-size:14px;margin:18px 0 4px'>"
                f"Top {min(40, len(material))} reactions by |Δflux| vs "
                f"{html.escape(refc['engine'])}</h3>"
                f"<table><thead>{head}</thead><tbody>{''.join(trows)}</tbody></table>")

    return (f"<h2>Per-reaction flux divergence (metabolism deep-dive)</h2>"
            f"<p class='note'>Opens up the single <em>FBA objective</em> / "
            f"<em>central flux |mean|</em> numbers above: each row is a "
            f"<b>base reaction</b> (isozyme / fwd-rev variants lumped), flux is "
            f"the per-reaction signed mean over the run (mmol·gDCW⁻¹·h⁻¹). "
            f"Caveat: FBA has <b>alternate optimal solutions</b>, so reversible "
            f"reactions and TCA/PPP branch points can differ even at matched "
            f"growth — a near-identical FBA objective can still hide per-pathway "
            f"flux reshuffling, which is exactly what this section surfaces.</p>"
            f"{struct_html}"
            f"<h3 style='font-size:14px;margin:18px 0 4px'>Pathways with the "
            f"largest flux divergence</h3>{pr_html}"
            f"{flip_html}{top_html}")


def build_html(results, duration, ref_idx=0):
    cols = [_metrics(r) for r in results]
    ref = cols[ref_idx]

    head = "<tr><th>Metric</th>" + "".join(
        f"<th>{html.escape(c['engine'])}"
        + ("<div class='ref'>reference</div>" if i == ref_idx else "")
        + "</th>"
        for i, c in enumerate(cols)
    ) + "</tr>"

    body = []
    for key, label, fmt, lower_better, tol in _ROWS:
        if fmt is None:  # section header row
            body.append(f"<tr class='group'><td colspan='{len(cols)+1}'>"
                        f"{html.escape(label)}</td></tr>")
            continue
        cells = [f"<td class='metric'>{html.escape(label)}</td>"]
        for i, c in enumerate(cols):
            v = c.get(key, 0.0)
            txt = fmt.format(v) if isinstance(v, (int, float)) else str(v)
            extra = ""
            if i != ref_idx:
                rv = ref.get(key, 0.0)
                if rv and lower_better is not None:
                    pct = 100.0 * (v - rv) / rv
                    cls = "up" if pct >= 0 else "down"
                    extra += f"<span class='delta {cls}'>{pct:+.1f}%</span>"
                extra += _verdict_badge(_verdict(v, rv, tol))
            cells.append(f"<td>{txt}{extra}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v2ecoli — composite comparison</title><style>{CSS}</style></head>
<body>
<header>
  <h1>Composite comparison</h1>
  <p>{len(cols)} engines · {duration}s requested · Δ% shown vs the reference
     (first) engine. Each engine run in an isolated subprocess.</p>
</header>
<main>
  <h2>Summary metrics</h2>
  <p class='note'>Δ% vs reference; divergence badge where a tolerance applies
     (within&nbsp;tol ≤ tol, drift ≤ 3·tol, else mismatch).</p>
  <table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>
  {_runtime_section(cols)}
  {_bridge_section(cols)}
  {_flux_divergence_section(cols, ref_idx)}
  {_overlay_section(cols)}
  {_trajectory_section(cols)}
  {_species_section(cols, ref_idx)}
</main>
<footer>
  Generated by <code>reports/composite_comparison.py</code>. v2ecoli composites
  share the same biology; differences vs vEcoli and across composites reflect
  wiring/scheduling, not the underlying model. Not bit-identical — compare by
  tolerance.
</footer>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Structured multi-composite comparison")
    ap.add_argument("--engines", nargs="+",
                    default=["vecoli", "baseline", "millard_fba_bridge_harness"],
                    help="engine tokens (vecoli / vecoli_composite / <v2ecoli composite name>)")
    ap.add_argument("--duration", type=int, default=2520)
    ap.add_argument("--interval", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)
    os.makedirs(os.path.join(base, REPORT_DIR), exist_ok=True)
    out_path = args.out or os.path.join(base, REPORT_DIR, "composite_comparison.html")

    print("=" * 64)
    print(f"Composite comparison: {', '.join(args.engines)} ({args.duration}s)")
    print("=" * 64)

    # vEcoli v1 mutates the vEcoli checkout's branch, so it must run alone and
    # last; v2ecoli composites are independent and can run in parallel.
    vecoli_tokens = [t for t in args.engines if t in _VECOLI_RUNNERS]
    v2_tokens = [t for t in args.engines if t not in _VECOLI_RUNNERS]

    results_by_token = {}
    t0 = time.time()

    # Phase 1: all v2ecoli composites in parallel.
    procs = {}
    for tok in v2_tokens:
        rpath = os.path.join(base, REPORT_DIR, f"_cmp_{tok}.json")
        print(f"  launching v2ecoli composite: {tok}")
        procs[tok] = (_launch(tok, args.duration, args.interval, base, rpath), rpath)
    for tok, (proc, rpath) in procs.items():
        results_by_token[tok] = _collect(tok, proc, rpath)
        m = results_by_token[tok]
        print(f"  {tok}: {m.get('sim_time',0)}s in {m.get('wall_time',0):.1f}s "
              f"({m.get('speed',0):.1f}x)")

    # Phase 2: vEcoli engines sequentially (they switch the vEcoli branch).
    for tok in vecoli_tokens:
        rpath = os.path.join(base, REPORT_DIR, f"_cmp_{tok}.json")
        print(f"  launching {tok} (sequential)")
        proc = _launch(tok, args.duration, args.interval, base, rpath)
        results_by_token[tok] = _collect(tok, proc, rpath)
        m = results_by_token[tok]
        print(f"  {tok}: {m.get('sim_time',0)}s in {m.get('wall_time',0):.1f}s "
              f"({m.get('speed',0):.1f}x)")

    # Preserve the user's --engines order for the table columns.
    results = [results_by_token[t] for t in args.engines if t in results_by_token]

    page = build_html(results, args.duration)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    # Mirror to docs/ for GitHub Pages (skip if --out already wrote there).
    import shutil
    mirror = os.path.join(base, "docs", "composite_comparison.html")
    if os.path.isdir(os.path.dirname(mirror)) and not (
            os.path.exists(mirror) and os.path.samefile(out_path, mirror)):
        shutil.copy2(out_path, mirror)

    print(f"\nReport: {out_path}")
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
