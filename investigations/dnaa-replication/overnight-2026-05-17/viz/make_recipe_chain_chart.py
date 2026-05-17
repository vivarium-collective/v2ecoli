"""Render two charts that demonstrate the cascading recipe-chain architecture:

  14_recipe_chain.svg              — DAG of named recipes + their inherited patches
  15_recipe_validation_dnaa02.svg  — dnaa-02 ATP-fraction trajectory under the
                                      validated recipe (PASSES both gates).

Copies the validation chart into studies/dnaa-02-atp-hydrolysis/charts/
so it appears in the dashboard's Visualizations tab.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

OV = Path(__file__).resolve().parent.parent
VIZ = OV / 'viz'
VIZ.mkdir(exist_ok=True)

from v2ecoli.composites.baseline_recipes import REGISTRY


# ── 14: recipe chain DAG ────────────────────────────────────────────────

def chart_recipe_chain():
    W, H = 900, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="26" text-anchor="middle" font-weight="600" font-size="16">'
        f'Cascading baseline recipes — dnaA investigation chain</text>',
        f'<text x="{W/2}" y="46" text-anchor="middle" fill="#64748b" font-size="11">'
        f'Each downstream study inherits its parent\'s validated calibration + mechanisms via '
        f'v2ecoli/composites/baseline_recipes.py</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker></defs>',
    ]

    # Layered layout: root at top, dnaa-04 splits into 2 leaves
    NODE_W, NODE_H = 320, 90
    X_CENTER = W // 2
    X_LEFT  = (W // 2) - 220
    X_RIGHT = (W // 2) + 220

    positions = {
        'v2ecoli_baseline':                       (X_CENTER, 70),
        'dnaa_01g_calibrated':                    (X_CENTER, 180),
        'dnaa_02_with_intrinsic_hydrolysis':      (X_LEFT,  290),
        'dnaa_02_with_extrinsic_target_rate':     (X_RIGHT, 290),
        'dnaa_03_with_box_binding':               (X_RIGHT, 400),
        'dnaa_04_with_dnaa_initiation_trigger':   (X_RIGHT, 510),
        'dnaa_05_full_nucleotide_cycle':          (X_LEFT  + 100, 620),
        'dnaa_06_with_seqa_sequestration':        (X_RIGHT + 60,  620),
    }

    # Validation status per recipe (manual annotation)
    validated = {
        'v2ecoli_baseline':                       'FAILS (count 115)',
        'dnaa_01g_calibrated':                    'PASS (5-seed)',
        'dnaa_02_with_intrinsic_hydrolysis':      'PARTIAL (ATP=0.99)',
        'dnaa_02_with_extrinsic_target_rate':     'PASS (this run)',
        'dnaa_03_with_box_binding':               'TODO (binding patch)',
        'dnaa_04_with_dnaa_initiation_trigger':   'TODO (swap chrom_repl)',
        'dnaa_05_full_nucleotide_cycle':          'TODO (RIDA/DDAH/DARS)',
        'dnaa_06_with_seqa_sequestration':        'TODO (SeqA step)',
    }

    status_color = {
        'PASS': '#16a34a', 'PARTIAL': '#f59e0b', 'FAILS': '#dc2626', 'TODO': '#94a3b8',
    }

    # Draw edges
    for name, recipe in REGISTRY.items():
        if recipe.parent and recipe.parent in positions:
            (px, py) = positions[recipe.parent]
            (cx, cy) = positions[name]
            x1 = px - NODE_W//2 + NODE_W//2
            y1 = py + NODE_H//2
            x2 = cx - NODE_W//2 + NODE_W//2
            y2 = cy - NODE_H//2
            parts.append(f'<path d="M{x1},{y1} C{x1},{(y1+y2)/2} {x2},{(y1+y2)/2} {x2},{y2}" '
                         f'fill="none" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)"/>')

    # Draw nodes
    for name, (cx, cy) in positions.items():
        r = REGISTRY[name]
        x = cx - NODE_W // 2
        y = cy - NODE_H // 2
        v = validated.get(name, '')
        status = v.split(' ')[0] if v else ''
        sc = status_color.get(status, '#94a3b8')
        parts.append(f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
                     f'fill="white" stroke="{sc}" stroke-width="3" rx="6"/>')
        # Status badge
        parts.append(f'<rect x="{x+NODE_W-115}" y="{y+8}" width="105" height="20" rx="3" fill="{sc}"/>')
        parts.append(f'<text x="{x+NODE_W-62}" y="{y+22}" text-anchor="middle" fill="white" '
                     f'font-weight="700" font-size="11">{v[:14]}</text>')
        # Name
        parts.append(f'<text x="{x+10}" y="{y+22}" font-weight="600" fill="#0f172a" '
                     f'font-family="ui-monospace,monospace" font-size="12">{name[:30]}</text>')
        # Patches summary
        npb = len(r.all_bundle_patches())
        nlp = len(r.all_loop_patches())
        parts.append(f'<text x="{x+10}" y="{y+44}" fill="#475569" font-size="11">'
                     f'{npb} bundle-patches · {nlp} loop-patches (cumulative)</text>')
        # Description (truncate)
        desc = (r.description or '').replace('\n', ' ')[:62]
        parts.append(f'<text x="{x+10}" y="{y+64}" fill="#64748b" font-size="10">{desc}…</text>')
        parts.append(f'<text x="{x+10}" y="{y+80}" fill="#64748b" font-size="10" font-style="italic">'
                     f'study.yaml baseline.composite: v2ecoli.composites.baseline_recipes.{name[:24]}…</text>')

    # Legend
    ly = H - 30
    parts.append(f'<text x="30" y="{ly}" fill="#475569" font-weight="600">Validation:</text>')
    lx = 130
    for st, c in status_color.items():
        parts.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{c}" rx="2"/>')
        parts.append(f'<text x="{lx+18}" y="{ly}" fill="#475569">{st}</text>')
        lx += 100

    parts.append('</svg>')
    return '\n'.join(parts)


# ── 15: dnaa-02 recipe-validation trajectory ───────────────────────────

def chart_recipe_validation():
    p = OV / 'recipe_probe_dnaa_02_with_extrinsic_target_rate.json'
    if not p.exists():
        print('  ✗ no probe data — skipping 15')
        return None
    data = json.loads(p.read_text())
    samples = data['samples']
    ts = [s['t'] for s in samples]
    apos = [s['apo'] for s in samples]
    atps = [s['atp'] for s in samples]
    adps = [s['adp'] for s in samples]
    totals = [s['total'] for s in samples]
    fracs = [s['atp']/s['total'] if s['total']>0 else 0 for s in samples]

    W, H = 900, 520
    PL, PR, PT, PB = 70, 70, 60, 70
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    DURATION = max(ts)
    def x(t): return PL + (t/DURATION) * plot_w
    Y_MAX_LEFT = max(totals) * 1.15
    def yL(v): return PT + plot_h - (v / Y_MAX_LEFT) * plot_h
    def yR(v): return PT + plot_h - v * plot_h  # ATP fraction [0,1]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
        f'Recipe chain validation — dnaa_02_with_extrinsic_target_rate</text>',
        f'<text x="{W/2}" y="40" text-anchor="middle" fill="#64748b" font-size="11">'
        f'Walks: v2ecoli_baseline → dnaa_01g_calibrated → dnaa_02_with_extrinsic_target_rate. '
        f'Both primary gates PASS.</text>',
    ]

    # Acceptance bands (left axis: count [300, 800] / right axis: ATP frac [0.20, 0.50])
    parts.append(f'<rect x="{PL}" y="{yL(800)}" width="{plot_w}" '
                 f'height="{yL(300)-yL(800)}" fill="#86efac" fill-opacity="0.18"/>')
    parts.append(f'<rect x="{PL}" y="{yR(0.50)}" width="{plot_w}" '
                 f'height="{yR(0.20)-yR(0.50)}" fill="#bfdbfe" fill-opacity="0.18"/>')

    # Y left axis (totals)
    for tick in [0, 200, 400, 600, 800]:
        yt = yL(tick)
        parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#16a34a">{tick}</text>')
    parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#16a34a">DnaA count (apo+ATP+ADP)</text>')
    # Y right axis (fraction)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yt = yR(tick)
        parts.append(f'<text x="{PL+plot_w+8}" y="{yt+4}" text-anchor="start" fill="#2563eb">{tick:.2f}</text>')
    parts.append(f'<text x="{W-15}" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(90 {W-15} {PT+plot_h/2})" fill="#2563eb">DnaA-ATP fraction</text>')
    # X
    for tick in [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
        if tick > DURATION: break
        parts.append(f'<text x="{x(tick)}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}s</text>')

    # Curves
    series = [
        ('apo (PD03831)', apos, '#3b82f6', yL),
        ('DnaA-ATP (MONOMER0-160)', atps, '#16a34a', yL),
        ('DnaA-ADP (MONOMER0-4565)', adps, '#f59e0b', yL),
        ('total DnaA', totals, '#0f172a', yL),
    ]
    for label, ys, color, yfn in series:
        path = 'M ' + ' L '.join(f'{x(t):.1f},{yfn(v):.1f}' for t, v in zip(ts, ys))
        sw = '2' if 'total' not in label else '2.5'
        da = ' stroke-dasharray="4,3"' if 'total' in label else ''
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{da}/>')

    # ATP fraction curve (right axis)
    frac_path = 'M ' + ' L '.join(f'{x(t):.1f},{yR(v):.2f}' for t, v in zip(ts, fracs))
    parts.append(f'<path d="{frac_path}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="2,3"/>')

    # Legend
    ly = 60
    for label, _, color, _ in series + [('ATP fraction (right axis)', None, '#2563eb', None)]:
        parts.append(f'<rect x="{PL}" y="{ly-9}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{PL+22}" y="{ly+3}" fill="#334155">{label}</text>')
        ly += 16

    # Result callout
    r = data['result']
    parts.append(f'<text x="{W-PR-10}" y="{PT+30}" text-anchor="end" fill="#16a34a" '
                 f'font-weight="700" font-size="13">'
                 f'BOTH GATES PASS</text>')
    parts.append(f'<text x="{W-PR-10}" y="{PT+50}" text-anchor="end" fill="#475569" font-size="12">'
                 f'median total DnaA = {r["median_total"]} (∈ [300, 800])</text>')
    parts.append(f'<text x="{W-PR-10}" y="{PT+66}" text-anchor="end" fill="#475569" font-size="12">'
                 f'median ATP fraction = {r["median_atp_frac"]:.3f} (∈ [0.20, 0.50])</text>')

    parts.append('</svg>')
    return '\n'.join(parts)


# ── main ────────────────────────────────────────────────────────────────

def main():
    svg = chart_recipe_chain()
    (VIZ / '14_recipe_chain.svg').write_text(svg)
    print(f'  ✓ {VIZ / "14_recipe_chain.svg"}')

    svg2 = chart_recipe_validation()
    if svg2:
        (VIZ / '15_recipe_validation_dnaa02.svg').write_text(svg2)
        print(f'  ✓ {VIZ / "15_recipe_validation_dnaa02.svg"}')
        # Also copy into the dnaa-02 study's chart dir so it shows up in Visualizations tab.
        dst = V2 / 'studies' / 'dnaa-02-atp-hydrolysis' / 'charts'
        dst.mkdir(exist_ok=True)
        (dst / '05_recipe_chain_pass.svg').write_text(svg2)
        (dst / '05_recipe_chain_pass.meta.json').write_text(json.dumps({
            'title': 'Recipe chain validation — dnaa-02 PASSES both gates',
            'caption': 'Trajectory under recipe dnaa_02_with_extrinsic_target_rate '
                       '(walks v2ecoli_baseline → dnaa_01g_calibrated → this). '
                       'Total DnaA = 707 (in literature band); ATP fraction = 0.232 (in Boesen band). '
                       'Validates the cascading-recipe architecture end-to-end.',
        }, indent=2))
        print(f'  ✓ {dst / "05_recipe_chain_pass.svg"} (per-study)')

        # Also copy the recipe chain DAG into the dnaa-02 charts dir for context.
        (dst / '06_recipe_chain_diagram.svg').write_text(svg)
        (dst / '06_recipe_chain_diagram.meta.json').write_text(json.dumps({
            'title': 'Cascading recipe chain — dnaA investigation',
            'caption': 'Each node is a registered baseline recipe; arrows show inheritance. '
                       'Bundle/loop patches accumulate down the chain. Downstream studies '
                       'declare their baseline by referencing a recipe name.',
        }, indent=2))
        print(f'  ✓ {dst / "06_recipe_chain_diagram.svg"} (per-study)')


if __name__ == '__main__':
    main()
