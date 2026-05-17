"""Render TE-sweep visualizations as standalone SVG files.

Reads studies/dnaa-01-expression-dynamics/runs.db, groups by TE multiplier,
computes per-multiplier:
  - dnaA monomer count median (gate test 1 target [300, 800])
  - autorepression Pearson r (gate test 2 target r <= -0.3)

Outputs (in this same dir):
  - 01_te_sweep_count.svg   — bar chart of dnaA count per multiplier
  - 02_te_sweep_pearson.svg — bar chart of pearson r per multiplier
  - 03_te_sweep_combined.svg — combined gate-test visualization (both axes)
  - 04_dnaa_te_percentile.svg — DnaA's TE position in proteome distribution
"""
from __future__ import annotations
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

STUDY_DIR = Path("/Users/eranagmon/code/v2ecoli/studies/dnaa-01-expression-dynamics")
OUT_DIR = Path(__file__).resolve().parent

DNAA_MONOMER_IDX = 3861
DNAA_TF_IDX = 12
DNAA_CISTRON_IDX = 227

# Counts of dnaA, binding, mrna pulled per state row.
def _extract(state_json: str):
    try:
        s = json.loads(state_json)
        listeners = s.get("listeners") or {}
        mc = listeners.get("monomer_counts")
        if not isinstance(mc, list) or len(mc) <= DNAA_MONOMER_IDX:
            return None
        dnaa = float(mc[DNAA_MONOMER_IDX])
        rsp = listeners.get("rna_synth_prob") or {}
        nb = rsp.get("n_bound_TF_per_TU")
        if isinstance(nb, list) and nb and isinstance(nb[0], list):
            binding = sum(row[DNAA_TF_IDX] for row in nb if len(row) > DNAA_TF_IDX)
        else:
            binding = 0
        rc = listeners.get("rna_counts") or {}
        mrna = rc.get("mRNA_cistron_counts")
        if not isinstance(mrna, list) or len(mrna) <= DNAA_CISTRON_IDX:
            return None
        return (dnaa, float(binding), float(mrna[DNAA_CISTRON_IDX]))
    except Exception:
        return None


def _classify(name: str):
    m = re.match(r"^baseline-te(\d+)x-seed\d+$", name)
    if m:
        return int(m.group(1))
    if re.match(r"^baseline-seed\d+$", name):
        return 1
    return None


def _pearson(xs, ys):
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sdx, sdy = statistics.stdev(xs), statistics.stdev(ys)
    if sdx == 0 or sdy == 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    return cov / (sdx * sdy * (len(xs) - 1))


def aggregate():
    """Run the sweep aggregation. Returns dict[mult] -> dict of stats."""
    conn = sqlite3.connect(str(STUDY_DIR / "runs.db"))
    sims = conn.execute(
        "SELECT simulation_id, name FROM simulations ORDER BY name"
    ).fetchall()
    by_mult = {}
    for sim_id, name in sims:
        mult = _classify(name)
        if mult is None:
            continue
        by_mult.setdefault(mult, []).append((sim_id, name))

    results = {}
    for mult, sims_in_mult in sorted(by_mult.items()):
        dnaa_all, bind_all, mrna_all = [], [], []
        for sim_id, _ in sims_in_mult:
            rows = conn.execute(
                "SELECT state FROM history WHERE simulation_id=? ORDER BY step ASC",
                (sim_id,)
            ).fetchall()
            n = len(rows)
            for (sj,) in rows[n // 2:]:
                triple = _extract(sj)
                if triple is not None:
                    dnaa_all.append(triple[0])
                    bind_all.append(triple[1])
                    mrna_all.append(triple[2])
        if not dnaa_all:
            continue
        results[mult] = {
            "seeds": len(sims_in_mult),
            "samples": len(dnaa_all),
            "dnaa_median": statistics.median(dnaa_all),
            "dnaa_q25": statistics.quantiles(dnaa_all, n=4)[0] if len(dnaa_all) >= 4 else min(dnaa_all),
            "dnaa_q75": statistics.quantiles(dnaa_all, n=4)[2] if len(dnaa_all) >= 4 else max(dnaa_all),
            "mrna_mean": statistics.mean(mrna_all),
            "pearson_r": _pearson(bind_all, mrna_all),
            "count_pass": 300 <= statistics.median(dnaa_all) <= 800,
            "autorep_pass": (_pearson(bind_all, mrna_all) is not None
                             and _pearson(bind_all, mrna_all) <= -0.3),
        }
    conn.close()
    return results


# ---- SVG builders (handcoded, no matplotlib dep) ----

SVG_W = 720
SVG_H = 400
PAD_L, PAD_R, PAD_T, PAD_B = 60, 30, 50, 50


def _svg_header(title, w=SVG_W, h=SVG_H):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="-apple-system,sans-serif" '
            f'font-size="13"><title>{title}</title>'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w/2}" y="22" text-anchor="middle" font-weight="600" '
            f'font-size="15">{title}</text>')


def _svg_footer():
    return '</svg>'


def chart_dnaa_count(results, path):
    mults = sorted(results.keys())
    if not mults:
        return
    plot_w = SVG_W - PAD_L - PAD_R
    plot_h = SVG_H - PAD_T - PAD_B
    bar_w = plot_w / max(len(mults), 1) * 0.7
    y_max = max(r["dnaa_q75"] for r in results.values()) * 1.15
    y_max = max(y_max, 900)  # always include [300, 800] band fully

    def x(i): return PAD_L + (i + 0.5) * plot_w / len(mults)
    def y(v): return PAD_T + plot_h - (v / y_max) * plot_h

    parts = [_svg_header("dnaA monomer count vs translation_efficiency multiplier")]
    # Acceptance band [300, 800] shaded green
    y_top = y(800)
    y_bot = y(300)
    parts.append(f'<rect x="{PAD_L}" y="{y_top}" width="{plot_w}" '
                 f'height="{y_bot - y_top}" fill="#86efac" fill-opacity="0.3"/>')
    parts.append(f'<text x="{PAD_L + plot_w - 5}" y="{y_top + 14}" '
                 f'text-anchor="end" fill="#16a34a" font-size="11">'
                 f'literature range [300, 800] (Schmidt 2016)</text>')
    # Y axis
    for tick in [0, 200, 400, 600, 800, 1000]:
        if tick > y_max:
            break
        yt = y(tick)
        parts.append(f'<line x1="{PAD_L}" y1="{yt}" x2="{PAD_L + plot_w}" y2="{yt}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{yt + 4}" text-anchor="end" '
                     f'fill="#64748b">{tick}</text>')
    parts.append(f'<text x="20" y="{PAD_T + plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PAD_T + plot_h/2})" fill="#334155">'
                 f'DnaA / cell (median, second-half)</text>')
    # Bars
    for i, mult in enumerate(mults):
        r = results[mult]
        cx = x(i)
        bar_x = cx - bar_w / 2
        bar_top = y(r["dnaa_median"])
        bar_bottom = y(0)
        color = "#22c55e" if r["count_pass"] else "#94a3b8"
        parts.append(f'<rect x="{bar_x}" y="{bar_top}" width="{bar_w}" '
                     f'height="{bar_bottom - bar_top}" fill="{color}"/>')
        # IQR whisker
        parts.append(f'<line x1="{cx}" y1="{y(r["dnaa_q25"])}" '
                     f'x2="{cx}" y2="{y(r["dnaa_q75"])}" '
                     f'stroke="#1e293b" stroke-width="1.5"/>')
        # Label above bar
        parts.append(f'<text x="{cx}" y="{bar_top - 4}" text-anchor="middle" '
                     f'fill="#334155" font-size="11">{int(r["dnaa_median"])}</text>')
        # X label
        parts.append(f'<text x="{cx}" y="{PAD_T + plot_h + 16}" '
                     f'text-anchor="middle" fill="#334155">{mult}×</text>')
        parts.append(f'<text x="{cx}" y="{PAD_T + plot_h + 30}" '
                     f'text-anchor="middle" fill="#94a3b8" font-size="10">'
                     f'n={r["seeds"]}</text>')
    parts.append(_svg_footer())
    path.write_text("\n".join(parts))


def chart_pearson(results, path):
    mults = sorted(results.keys())
    if not mults:
        return
    plot_w = SVG_W - PAD_L - PAD_R
    plot_h = SVG_H - PAD_T - PAD_B
    bar_w = plot_w / max(len(mults), 1) * 0.7
    # Y axis: -1 to +1
    y_min, y_max = -1.0, 1.0

    def x(i): return PAD_L + (i + 0.5) * plot_w / len(mults)
    def y(v): return PAD_T + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    parts = [_svg_header("Autorepression Pearson r (binding ↔ mRNA) vs TE multiplier")]
    # Pass band: r in [-1, -0.3]
    y_top = y(-0.3)
    y_bot = y(-1.0)
    parts.append(f'<rect x="{PAD_L}" y="{y_top}" width="{plot_w}" '
                 f'height="{y_bot - y_top}" fill="#86efac" fill-opacity="0.3"/>')
    parts.append(f'<text x="{PAD_L + plot_w - 5}" y="{y_top + 14}" '
                 f'text-anchor="end" fill="#16a34a" font-size="11">'
                 f'pass band r ≤ -0.3 (autorepression signature)</text>')
    # Zero line
    y_zero = y(0)
    parts.append(f'<line x1="{PAD_L}" y1="{y_zero}" x2="{PAD_L + plot_w}" '
                 f'y2="{y_zero}" stroke="#64748b" stroke-dasharray="4,3"/>')
    # Y ticks
    for tick in [-1.0, -0.5, -0.3, 0.0, 0.5, 1.0]:
        yt = y(tick)
        parts.append(f'<line x1="{PAD_L}" y1="{yt}" x2="{PAD_L + plot_w}" y2="{yt}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{yt + 4}" text-anchor="end" '
                     f'fill="#64748b">{tick}</text>')
    parts.append(f'<text x="20" y="{PAD_T + plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PAD_T + plot_h/2})" fill="#334155">'
                 f'Pearson r (pooled across seeds)</text>')
    for i, mult in enumerate(mults):
        r = results[mult]
        if r["pearson_r"] is None:
            continue
        cx = x(i)
        bar_x = cx - bar_w / 2
        # Bar from y(0) to y(r)
        bar_top = y(min(0, r["pearson_r"]))
        bar_bottom = y(max(0, r["pearson_r"]))
        color = "#22c55e" if r["autorep_pass"] else "#ef4444"
        parts.append(f'<rect x="{bar_x}" y="{bar_top}" width="{bar_w}" '
                     f'height="{bar_bottom - bar_top}" fill="{color}"/>')
        # Label
        ly = y(r["pearson_r"]) + (-4 if r["pearson_r"] > 0 else 12)
        parts.append(f'<text x="{cx}" y="{ly}" text-anchor="middle" '
                     f'fill="#334155" font-size="11">{r["pearson_r"]:+.3f}</text>')
        # X label
        parts.append(f'<text x="{cx}" y="{PAD_T + plot_h + 16}" '
                     f'text-anchor="middle" fill="#334155">{mult}×</text>')
        parts.append(f'<text x="{cx}" y="{PAD_T + plot_h + 30}" '
                     f'text-anchor="middle" fill="#94a3b8" font-size="10">'
                     f'n={r["seeds"]} ({r["samples"]} samples)</text>')
    parts.append(_svg_footer())
    path.write_text("\n".join(parts))


def chart_combined(results, path):
    """One chart showing both gate tests on parallel y-axes per multiplier."""
    mults = sorted(results.keys())
    if not mults:
        return
    W, H = 900, 500
    PL, PR, PT, PB = 70, 70, 60, 60
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    def x(i): return PL + (i + 0.5) * plot_w / len(mults)

    # Left axis: dnaA count (0 to 900)
    def yL(v): return PT + plot_h - (v / 900) * plot_h
    # Right axis: pearson r (-1 to +1)
    def yR(v): return PT + plot_h - ((v + 1) / 2) * plot_h

    parts = [_svg_header(
        "Gate-test landscape: dnaA count (left, bars) and autorepression r (right, points)",
        w=W, h=H)]

    # Acceptance bands
    parts.append(f'<rect x="{PL}" y="{yL(800)}" width="{plot_w}" '
                 f'height="{yL(300) - yL(800)}" fill="#86efac" fill-opacity="0.25"/>')
    parts.append(f'<rect x="{PL}" y="{yR(-0.3)}" width="{plot_w}" '
                 f'height="{yR(-1.0) - yR(-0.3)}" fill="#bfdbfe" fill-opacity="0.2"/>')

    # Y axes ticks (left)
    for tick in [0, 200, 400, 600, 800]:
        yt = yL(tick)
        parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL + plot_w}" y2="{yt}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PL - 8}" y="{yt + 4}" text-anchor="end" '
                     f'fill="#16a34a">{tick}</text>')
    # Right axis ticks
    for tick in [-1.0, -0.5, -0.3, 0.0, 0.5, 1.0]:
        yt = yR(tick)
        parts.append(f'<text x="{PL + plot_w + 8}" y="{yt + 4}" text-anchor="start" '
                     f'fill="#2563eb">{tick}</text>')
    parts.append(f'<text x="20" y="{PT + plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT + plot_h/2})" fill="#16a34a">'
                 f'DnaA/cell (median)</text>')
    parts.append(f'<text x="{W - 20}" y="{PT + plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(90 {W-20} {PT + plot_h/2})" fill="#2563eb">'
                 f'Pearson r</text>')

    # Bars + points
    bar_w = plot_w / max(len(mults), 1) * 0.45
    for i, mult in enumerate(mults):
        r = results[mult]
        cx = x(i)
        # Bar for count
        bar_top = yL(r["dnaa_median"])
        parts.append(f'<rect x="{cx - bar_w}" y="{bar_top}" width="{bar_w}" '
                     f'height="{yL(0) - bar_top}" fill="#16a34a" fill-opacity="0.6"/>')
        # Point for pearson
        if r["pearson_r"] is not None:
            py = yR(r["pearson_r"])
            color = "#22c55e" if r["autorep_pass"] else "#ef4444"
            parts.append(f'<circle cx="{cx + bar_w/2}" cy="{py}" r="6" '
                         f'fill="{color}" stroke="#1e293b" stroke-width="1.5"/>')
        # X label
        parts.append(f'<text x="{cx}" y="{PT + plot_h + 18}" text-anchor="middle" '
                     f'fill="#334155">{mult}×</text>')
        # Pass/fail summary below
        cp = "✓" if r["count_pass"] else "✗"
        ap = "✓" if r["autorep_pass"] else "✗"
        cc = "#22c55e" if r["count_pass"] else "#ef4444"
        ac = "#22c55e" if r["autorep_pass"] else "#ef4444"
        parts.append(f'<text x="{cx}" y="{PT + plot_h + 36}" text-anchor="middle" '
                     f'font-size="11"><tspan fill="{cc}">{cp} count</tspan> '
                     f'<tspan fill="{ac}">{ap} r</tspan></text>')

    # Legend
    parts.append(f'<text x="{PL}" y="{H - 12}" fill="#64748b" font-size="11">'
                 f'Green bar: dnaA count (left axis). '
                 f'Circles: autorepression r (right axis). '
                 f'Shaded: pass bands. </text>')
    parts.append(_svg_footer())
    path.write_text("\n".join(parts))


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    results = aggregate()
    if not results:
        print("No sweep data yet.")
        return
    print(f"Multipliers found: {sorted(results.keys())}")
    chart_dnaa_count(results, OUT_DIR / "01_te_sweep_count.svg")
    chart_pearson(results, OUT_DIR / "02_te_sweep_pearson.svg")
    chart_combined(results, OUT_DIR / "03_te_sweep_combined.svg")
    # Also dump JSON for downstream
    (OUT_DIR / "sweep_aggregate.json").write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    print("Wrote SVGs + sweep_aggregate.json")
    # Print summary table
    print(f"\n{'TE×':>5} | {'seeds':>5} | {'DnaA med':>9} | {'pearson r':>10} | {'count':>6} | {'autorep':>8}")
    print("-" * 70)
    for mult in sorted(results.keys()):
        r = results[mult]
        rs = f"{r['pearson_r']:+.3f}" if r['pearson_r'] is not None else "N/A"
        print(f"{mult:>5}× | {r['seeds']:>5} | {r['dnaa_median']:>9.0f} | "
              f"{rs:>10} | {'PASS' if r['count_pass'] else 'FAIL':>6} | "
              f"{'PASS' if r['autorep_pass'] else 'FAIL':>8}")


if __name__ == "__main__":
    main()
