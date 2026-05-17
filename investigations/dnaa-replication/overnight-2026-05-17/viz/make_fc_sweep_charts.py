"""Render TE × fold_change joint-sweep visualization from the fc-tagged sims.

Sim names: baseline-te{N}x-fc{F}-seed{S}.
"""
from __future__ import annotations
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

STUDY_DIR = Path("/Users/eranagmon/code/v2ecoli/studies/dnaa-01-expression-dynamics")
OUT_DIR = Path(__file__).resolve().parent.parent  # overnight dir

DNAA_MONOMER_IDX = 3861
DNAA_TF_IDX = 12
DNAA_CISTRON_IDX = 227


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
    """Parse 'baseline-te{N}x-fc{F}-seed{S}' or 'baseline-te{N}x-seed{S}' or
    'baseline-seed{S}'. Returns (te_mult, fc_mult) or None."""
    m = re.match(r"^baseline-te(\d+)x-fc([\d.]+)-seed\d+$", name)
    if m:
        return (int(m.group(1)), float(m.group(2)))
    m = re.match(r"^baseline-te(\d+)x-seed\d+$", name)
    if m:
        return (int(m.group(1)), 1.0)
    if re.match(r"^baseline-seed\d+$", name):
        return (1, 1.0)
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
    conn = sqlite3.connect(str(STUDY_DIR / "runs.db"))
    sims = conn.execute("SELECT simulation_id, name FROM simulations ORDER BY name").fetchall()
    by_pt = {}  # (te, fc) -> [(sim_id, name)]
    for sim_id, name in sims:
        c = _classify(name)
        if c is None:
            continue
        by_pt.setdefault(c, []).append((sim_id, name))

    out = {}
    for pt, sims_in_pt in by_pt.items():
        dnaa_all, bind_all, mrna_all = [], [], []
        for sim_id, _ in sims_in_pt:
            rows = conn.execute(
                "SELECT state FROM history WHERE simulation_id=? ORDER BY step ASC",
                (sim_id,)
            ).fetchall()
            n = len(rows)
            for (sj,) in rows[n // 2:]:
                t = _extract(sj)
                if t is not None:
                    dnaa_all.append(t[0]); bind_all.append(t[1]); mrna_all.append(t[2])
        if not dnaa_all:
            continue
        out[pt] = {
            "seeds": len(sims_in_pt),
            "samples": len(dnaa_all),
            "dnaa_median": statistics.median(dnaa_all),
            "pearson_r": _pearson(bind_all, mrna_all),
            "count_pass": 300 <= statistics.median(dnaa_all) <= 800,
        }
        if out[pt]["pearson_r"] is not None:
            out[pt]["autorep_pass"] = out[pt]["pearson_r"] <= -0.3
        else:
            out[pt]["autorep_pass"] = False
    conn.close()
    return out


def make_grid_heatmap(results, path, metric, vmin, vmax, cmap_low, cmap_high, title, fmt='.0f'):
    """Render a heatmap: rows = TE multipliers, cols = fc multipliers.
    metric is the dict key. Color interpolates [vmin → cmap_low, vmax → cmap_high]."""
    tes = sorted({pt[0] for pt in results})
    fcs = sorted({pt[1] for pt in results})
    if not tes or not fcs:
        return
    W, H = 760, 60 + 60 * len(tes) + 50
    PL, PT = 80, 60
    cell_w, cell_h = 100, 55

    def color_for(v):
        if v is None:
            return '#f1f5f9'
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
        # Linear interp between cmap_low and cmap_high RGB
        def hex2rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
        def rgb2hex(rgb): return '#' + ''.join(f'{int(x):02x}' for x in rgb)
        lo = hex2rgb(cmap_low)
        hi = hex2rgb(cmap_high)
        return rgb2hex([lo[i] + (hi[i] - lo[i]) * t for i in range(3)])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">{title}</text>',
    ]
    # Header row
    for j, fc in enumerate(fcs):
        cx = PL + j * cell_w + cell_w / 2
        parts.append(f'<text x="{cx}" y="{PT - 12}" text-anchor="middle" '
                     f'fill="#334155" font-weight="600">fc={fc}</text>')
    # Row labels
    for i, te in enumerate(tes):
        cy = PT + i * cell_h + cell_h / 2
        parts.append(f'<text x="{PL - 10}" y="{cy + 4}" text-anchor="end" '
                     f'fill="#334155" font-weight="600">{te}×</text>')
    # Cells
    for i, te in enumerate(tes):
        for j, fc in enumerate(fcs):
            r = results.get((te, fc))
            x = PL + j * cell_w
            y = PT + i * cell_h
            if r is None:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" '
                             f'fill="#f1f5f9" stroke="#e5e7eb"/>')
                parts.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 4}" '
                             f'text-anchor="middle" fill="#cbd5e1">—</text>')
                continue
            v = r.get(metric)
            fill = color_for(v)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" '
                         f'fill="{fill}" stroke="#94a3b8"/>')
            label = format(v, fmt) if v is not None else 'N/A'
            text_color = '#0f172a' if abs((v or 0) - (vmin + vmax) / 2) < (vmax - vmin) / 3 else '#fff'
            parts.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 - 2}" '
                         f'text-anchor="middle" fill="{text_color}" font-weight="600">{label}</text>')
            parts.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 14}" '
                         f'text-anchor="middle" fill="{text_color}" font-size="10">n={r["seeds"]}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def main():
    results = aggregate()
    if not results:
        print("No fc-tagged sims yet.")
        return
    print(f"{'TE×':>5} {'fc':>5} {'seeds':>5} {'samples':>8} {'DnaA med':>8} {'r':>8} {'count':>6} {'autorep':>8}")
    print("-" * 70)
    for pt in sorted(results.keys()):
        te, fc = pt
        r = results[pt]
        rs = f"{r['pearson_r']:+.3f}" if r['pearson_r'] is not None else 'N/A'
        print(f"{te:>5}× {fc:>5} {r['seeds']:>5} {r['samples']:>8} {r['dnaa_median']:>8.0f} {rs:>8} "
              f"{'PASS' if r['count_pass'] else 'FAIL':>6} {'PASS' if r['autorep_pass'] else 'FAIL':>8}")

    # Charts
    viz = OUT_DIR / 'viz'
    make_grid_heatmap(
        results, viz / '09_fc_grid_dnaa_count.svg',
        metric='dnaa_median', vmin=100, vmax=600,
        cmap_low='#cbd5e1', cmap_high='#16a34a',
        title='Joint (TE × fold_change) sweep — median DnaA per cell')
    make_grid_heatmap(
        results, viz / '10_fc_grid_pearson.svg',
        metric='pearson_r', vmin=-1.0, vmax=+1.0,
        cmap_low='#16a34a', cmap_high='#dc2626',
        title='Joint (TE × fold_change) sweep — autorepression Pearson r',
        fmt='+.2f')
    print(f"\nWrote {viz}/09_fc_grid_dnaa_count.svg and 10_fc_grid_pearson.svg")
    # Dump json
    (OUT_DIR / 'fc_sweep_aggregate.json').write_text(json.dumps(
        {f'{te}x_fc{fc}': v for (te, fc), v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
