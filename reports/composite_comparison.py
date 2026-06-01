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
  * growth:      initial / final dry mass, fold-change, mean growth rate
  * composition: final protein / RNA / DNA / small-molecule mass
  * dynamics:    chromosomes, replication forks at end
Each metric column also shows Δ% vs the first engine (the reference).

Usage:
    # vEcoli vs v2ecoli baseline vs millard, full cycle:
    python reports/composite_comparison.py \\
        --engines vecoli baseline millard_pdmp_baseline --duration 2520

    # quick smoke:
    python reports/composite_comparison.py --engines baseline millard_pdmp_baseline --duration 60

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
def _metrics(data: dict) -> dict:
    snaps = data.get("snapshots") or []
    first = snaps[0] if snaps else {}
    last = snaps[-1] if snaps else {}

    def g(s, k):
        return float(s.get(k, 0) or 0)

    dry0, dry1 = g(first, "dry_mass"), g(last, "dry_mass")
    rates = [g(s, "instantaneous_growth_rate") for s in snaps]
    rates = [r for r in rates if r]
    return {
        "engine": data.get("engine", "?"),
        # performance
        "load_s": data.get("load_time", 0.0),
        "wall_s": data.get("wall_time", 0.0),
        "sim_s": data.get("sim_time", 0.0),
        "realtime_x": data.get("speed", 0.0),
        # growth
        "dry_mass_0": dry0,
        "dry_mass_f": dry1,
        "dry_fold": (dry1 / dry0) if dry0 else 0.0,
        "growth_rate": (sum(rates) / len(rates)) if rates else 0.0,
        # composition (final)
        "protein_mass_f": g(last, "protein_mass"),
        "rna_mass_f": g(last, "rna_mass"),
        "dna_mass_f": g(last, "dna_mass"),
        "smol_mass_f": g(last, "smallMolecule_mass"),
        "water_mass_f": g(last, "water_mass"),
        # molecular species
        "bulk_total_f": g(last, "bulk_total"),
        "bulk_nonzero_f": g(last, "bulk_species_nonzero"),
        "bulk_n_species": g(last, "bulk_n_species"),
        # dynamics (final)
        "n_chrom_f": g(last, "n_chromosomes"),
        "n_forks_f": g(last, "n_forks"),
        "n_snapshots": len(snaps),
        # full final unique-molecule counts (per type) + the snapshot list for
        # trajectory rendering
        "unique_final": (last.get("unique_counts") or {}),
        "_snaps": snaps,
    }


# (key, label, fmt, lower-is-better-for-Δ?) — Δ% computed vs the reference col.
_ROWS = [
    ("__perf__", "Performance", None, None),
    ("load_s", "Load time (s)", "{:.1f}", None),
    ("wall_s", "Wall time (s)", "{:.1f}", None),
    ("sim_s", "Sim time reached (s)", "{:.0f}", None),
    ("realtime_x", "Realtime factor (×)", "{:.1f}", None),
    ("__growth__", "Growth", None, None),
    ("dry_mass_0", "Dry mass, initial (fg)", "{:.1f}", None),
    ("dry_mass_f", "Dry mass, final (fg)", "{:.1f}", True),
    ("dry_fold", "Dry-mass fold change", "{:.4f}", True),
    ("growth_rate", "Mean growth rate (1/s)", "{:.3e}", True),
    ("__comp__", "Composition (final)", None, None),
    ("protein_mass_f", "Protein mass (fg)", "{:.1f}", True),
    ("rna_mass_f", "RNA mass (fg)", "{:.1f}", True),
    ("dna_mass_f", "DNA mass (fg)", "{:.1f}", True),
    ("smol_mass_f", "Small-molecule mass (fg)", "{:.1f}", True),
    ("water_mass_f", "Water mass (fg)", "{:.1f}", True),
    ("__species__", "Molecular species (final)", None, None),
    ("bulk_total_f", "Bulk molecules, total count", "{:.0f}", True),
    ("bulk_nonzero_f", "Bulk species present (count>0)", "{:.0f}", True),
    ("bulk_n_species", "Bulk species, distinct", "{:.0f}", None),
    ("__dyn__", "Unique molecules (final)", None, None),
    ("n_chrom_f", "Chromosomes", "{:.0f}", None),
    ("n_forks_f", "Replication forks", "{:.0f}", None),
    ("n_snapshots", "Snapshots", "{:.0f}", None),
]

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
footer{max-width:1100px;margin:0 auto;padding:0 36px 40px;color:#64748b;font-size:12px}
"""


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
    for key, label, fmt, lower_better in _ROWS:
        if fmt is None:  # section header row
            body.append(f"<tr class='group'><td colspan='{len(cols)+1}'>"
                        f"{html.escape(label)}</td></tr>")
            continue
        cells = [f"<td class='metric'>{html.escape(label)}</td>"]
        for i, c in enumerate(cols):
            v = c.get(key, 0.0)
            txt = fmt.format(v) if isinstance(v, (int, float)) else str(v)
            delta = ""
            if i != ref_idx and lower_better is not None:
                rv = ref.get(key, 0.0)
                if rv:
                    pct = 100.0 * (v - rv) / rv
                    cls = "up" if pct >= 0 else "down"
                    delta = f"<span class='delta {cls}'>{pct:+.1f}%</span>"
            cells.append(f"<td>{txt}{delta}</td>")
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
  <table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>
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
    ap.add_argument("--engines", nargs="+", default=["vecoli", "baseline", "millard_pdmp_baseline"],
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
