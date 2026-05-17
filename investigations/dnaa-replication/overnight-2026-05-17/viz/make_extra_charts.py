"""Additional SVG charts:
 - 04_dnaa_te_percentile.svg     — dnaA's TE position within proteome (Insight #1)
 - 06_dnaa_atp_fraction.svg      — DnaA-ATP fraction over time (Insight #3)
 - 07_findings_overview.svg      — All findings from the night, cross-study
 - 08_investigation_dag.svg      — Investigation status DAG
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

OUT_DIR = Path(__file__).resolve().parent.parent  # the overnight dir

# ---------- 04: TE percentile ----------
def make_te_percentile():
    from v2ecoli.core import load_cache_bundle
    import numpy as np

    bundle = load_cache_bundle('out/cache')
    pi = bundle['configs']['ecoli-polypeptide-initiation']
    te = pi['translation_efficiencies']
    DNAA_IDX = 3861

    W, H = 800, 380
    PL, PR, PT, PB = 60, 30, 50, 60
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    log_min = np.log10(te[te > 0].min())
    log_max = np.log10(te.max())
    bins = 60
    hist, edges = np.histogram(np.log10(te[te > 0]), bins=bins, range=(log_min, log_max))
    bin_w = plot_w / bins

    def x_bin(i):  return PL + i * bin_w
    h_max = hist.max() * 1.1
    def y(h): return PT + plot_h - (h / h_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
        f'Proteome translation_efficiency distribution — dnaA at 8.6th percentile</text>',
    ]
    # Bars
    for i, h in enumerate(hist):
        if h <= 0:
            continue
        parts.append(f'<rect x="{x_bin(i)}" y="{y(h)}" width="{bin_w*0.92}" '
                     f'height="{y(0)-y(h)}" fill="#94a3b8" fill-opacity="0.8"/>')

    # Highlight dnaA's position
    dnaa_log = np.log10(te[DNAA_IDX])
    dnaa_x = PL + (dnaa_log - log_min) / (log_max - log_min) * plot_w
    parts.append(f'<line x1="{dnaa_x}" y1="{PT}" x2="{dnaa_x}" y2="{PT+plot_h}" '
                 f'stroke="#dc2626" stroke-width="2.5"/>')
    parts.append(f'<text x="{dnaa_x+6}" y="{PT+18}" fill="#dc2626" font-weight="600" '
                 f'font-size="12">dnaA (PD03831)</text>')
    parts.append(f'<text x="{dnaa_x+6}" y="{PT+34}" fill="#dc2626" font-size="11">'
                 f'TE = {te[DNAA_IDX]:.2e}</text>')
    parts.append(f'<text x="{dnaa_x+6}" y="{PT+48}" fill="#dc2626" font-size="11">'
                 f'8.6th percentile</text>')

    # Median marker
    median_log = np.log10(np.median(te[te > 0]))
    median_x = PL + (median_log - log_min) / (log_max - log_min) * plot_w
    parts.append(f'<line x1="{median_x}" y1="{PT}" x2="{median_x}" y2="{PT+plot_h}" '
                 f'stroke="#22c55e" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{median_x+6}" y="{PT+plot_h-10}" fill="#16a34a" font-size="11">'
                 f'median TE</text>')

    # X axis ticks
    for tick_exp in range(int(log_min), int(log_max)+1):
        tx = PL + (tick_exp - log_min) / (log_max - log_min) * plot_w
        parts.append(f'<line x1="{tx}" y1="{PT+plot_h}" x2="{tx}" y2="{PT+plot_h+4}" stroke="#64748b"/>')
        parts.append(f'<text x="{tx}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">10^{tick_exp}</text>')
    parts.append(f'<text x="{W/2}" y="{PT+plot_h+44}" text-anchor="middle" fill="#334155">'
                 f'translation_efficiency (log scale)</text>')
    parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">'
                 f'count of monomers</text>')
    parts.append('</svg>')
    (OUT_DIR / 'viz' / '04_dnaa_te_percentile.svg').write_text('\n'.join(parts))


# ---------- 06: DnaA-ATP fraction over time ----------
def make_atp_fraction_chart():
    p = OUT_DIR / 'dnaa_states_timeseries.json'
    if not p.exists():
        print('  skipping 06: no timeseries json')
        return
    data = json.loads(p.read_text())
    samples = data['samples']
    ts = [s['t'] for s in samples]
    fracs = [s['atp_frac'] for s in samples]

    W, H = 800, 380
    PL, PR, PT, PB = 60, 30, 50, 50
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    def x(tv): return PL + (tv / max(ts)) * plot_w
    def y(v): return PT + plot_h - v * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
        f'DnaA-ATP fraction over time (baseline 1× TE)</text>',
    ]

    # Boesen 2024 band [0.2, 0.5]
    parts.append(f'<rect x="{PL}" y="{y(0.5)}" width="{plot_w}" '
                 f'height="{y(0.2)-y(0.5)}" fill="#86efac" fill-opacity="0.3"/>')
    parts.append(f'<text x="{PL+10}" y="{y(0.5)+15}" fill="#16a34a" font-size="11">'
                 f'Boesen 2024 physiological band [0.2, 0.5]</text>')

    # Y ticks
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yt = y(tick)
        parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick:.2f}</text>')
    # X ticks
    for tick in [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
        if tick > max(ts): break
        xt = x(tick)
        parts.append(f'<text x="{xt}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}s</text>')

    # Trajectory
    path = 'M ' + ' L '.join(f'{x(t):.1f},{y(v):.1f}' for (t, v) in zip(ts, fracs))
    parts.append(f'<path d="{path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>')

    # Annotation
    parts.append(f'<text x="{W-PR-10}" y="{PT+30}" text-anchor="end" fill="#dc2626" font-size="12">'
                 f'final ATP fraction = {fracs[-1]:.3f}</text>')
    parts.append(f'<text x="{W-PR-10}" y="{PT+46}" text-anchor="end" fill="#dc2626" font-size="11">'
                 f'2× over upper bound of physiological band</text>')
    parts.append(f'<text x="{W/2}" y="{PT+plot_h+38}" text-anchor="middle" fill="#334155">'
                 f'simulated time (seconds)</text>')
    parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">DnaA-ATP / total</text>')
    parts.append('</svg>')
    (OUT_DIR / 'viz' / '06_dnaa_atp_fraction.svg').write_text('\n'.join(parts))


# ---------- 07: Findings overview ----------
def make_findings_overview():
    import yaml, glob
    # Gather all findings from all dnaa-* studies + the overnight ones
    rows = []
    for p in sorted(glob.glob('studies/dnaa-*/study.yaml')):
        d = yaml.safe_load(open(p))
        sname = p.split('/')[1]
        for f in (d.get('findings') or []):
            rows.append({
                'study': sname,
                'id': f.get('id', '?'),
                'kind': f.get('kind', '?'),
                'status': f.get('status', '?'),
                'statement': (f.get('statement', '') or '').replace('\n', ' ')[:160],
            })
    W = 1100
    rowh = 28
    H = 50 + rowh * len(rows) + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="20" y="28" font-weight="600" font-size="15" fill="#0f172a">'
        f'Investigation findings overview — {len(rows)} total</text>',
    ]
    # Header row
    y0 = 50
    parts.append(f'<text x="20" y="{y0}" font-weight="600" fill="#334155">Study</text>')
    parts.append(f'<text x="350" y="{y0}" font-weight="600" fill="#334155">ID</text>')
    parts.append(f'<text x="400" y="{y0}" font-weight="600" fill="#334155">Kind</text>')
    parts.append(f'<text x="490" y="{y0}" font-weight="600" fill="#334155">Status</text>')
    parts.append(f'<text x="580" y="{y0}" font-weight="600" fill="#334155">Statement</text>')
    parts.append(f'<line x1="20" y1="{y0+4}" x2="{W-20}" y2="{y0+4}" stroke="#e5e7eb"/>')

    status_color = {
        'confirms':    '#16a34a',
        'contradicts': '#dc2626',
        'novel':       '#2563eb',
        'partial':     '#ea580c',
    }
    kind_color = {
        'biological':    '#7c3aed',
        'computational': '#0891b2',
        'methodological':'#65a30d',
    }

    for i, r in enumerate(rows):
        ry = y0 + 20 + i * rowh
        # Alternating row stripe
        if i % 2 == 0:
            parts.append(f'<rect x="20" y="{ry-12}" width="{W-40}" height="{rowh-2}" fill="#f8fafc"/>')
        parts.append(f'<text x="20" y="{ry+4}" fill="#64748b" font-size="11" font-family="ui-monospace,monospace">'
                     f'{r["study"][:48]}</text>')
        parts.append(f'<text x="350" y="{ry+4}" fill="#334155" font-weight="600">{r["id"]}</text>')
        parts.append(f'<text x="400" y="{ry+4}" fill="{kind_color.get(r["kind"], "#64748b")}">'
                     f'{r["kind"]}</text>')
        parts.append(f'<text x="490" y="{ry+4}" fill="{status_color.get(r["status"], "#64748b")}">'
                     f'{r["status"]}</text>')
        parts.append(f'<text x="580" y="{ry+4}" fill="#0f172a">'
                     f'{r["statement"][:80]}</text>')
    parts.append('</svg>')
    (OUT_DIR / 'viz' / '07_findings_overview.svg').write_text('\n'.join(parts))


# ---------- 08: Investigation DAG ----------
def make_investigation_dag():
    import yaml, glob
    # Walk all dnaa-* studies and build the DAG
    nodes = {}  # name -> {phase, status, findings_count}
    edges = []  # (parent, child)
    for p in sorted(glob.glob('studies/dnaa-*/study.yaml')):
        d = yaml.safe_load(open(p))
        n = d.get('name')
        pg = d.get('pipeline_gate', {}) or {}
        nodes[n] = {
            'phase':  d.get('phase', 'Design'),
            'status': d.get('status', 'planned'),
            'findings': len(d.get('findings') or []),
            'runs': len(d.get('runs') or []),
            'gate':   pg.get('gate_status', 'unknown'),
        }
        for ps in (d.get('parent_studies') or []):
            parent = ps.get('study') if isinstance(ps, dict) else ps
            edges.append((parent, n))

    # Simple layered layout by phase + parent chain. Group nodes by their "level".
    # Level 0: root (dnaa-01)
    # Level 1: dnaa-01's children
    # etc.
    level = {n: 0 for n in nodes}
    for _ in range(10):
        changed = False
        for p, c in edges:
            if p in level and c in level:
                if level[c] <= level[p]:
                    level[c] = level[p] + 1
                    changed = True
        if not changed: break

    # Place: x by level, y by name within level
    by_level = {}
    for n, lv in level.items():
        by_level.setdefault(lv, []).append(n)
    for lv in by_level:
        by_level[lv].sort()

    max_lv = max(by_level.keys())
    NODE_W, NODE_H = 240, 80
    GAP_X, GAP_Y = 320, 30
    LEFT, TOP = 30, 60

    pos = {}
    for lv, names in by_level.items():
        for i, n in enumerate(names):
            pos[n] = (LEFT + lv * GAP_X, TOP + i * (NODE_H + GAP_Y))

    W = LEFT + (max_lv + 1) * GAP_X
    H = max(TOP + len(by_level.get(lv, [])) * (NODE_H + GAP_Y) + 60 for lv in by_level)

    phase_color = {
        'Design':   '#94a3b8',
        'Build':    '#0891b2',
        'Simulate': '#2563eb',
        'Evaluate': '#7c3aed',
        'Decide':   '#16a34a',
    }
    status_color = {
        'planned':  '#cbd5e1',
        'running':  '#3b82f6',
        'ran':      '#10b981',
        'complete': '#16a34a',
    }
    # Gate-status: dominant visual cue (border color + corner badge)
    gate_color = {
        'open':        '#16a34a',  # green — clear to move past
        'conditional': '#f59e0b',  # amber — clear with caveats
        'hold':        '#fb923c',  # orange — work done but question outstanding
        'ready':       '#3b82f6',  # blue — pick up next
        'blocked':     '#dc2626',  # red — wait
        'unknown':     '#94a3b8',
    }
    gate_label = {
        'open':        '✓ OPEN',
        'conditional': '⚠ CONDITIONAL',
        'hold':        '⏸ HOLD',
        'ready':       '▶ READY',
        'blocked':     '✗ BLOCKED',
        'unknown':     '? UNKNOWN',
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="28" text-anchor="middle" font-weight="600" font-size="15" fill="#0f172a">'
        f'dnaA / replication-initiation investigation — DAG status</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/></marker></defs>',
    ]

    # Edges
    for p, c in edges:
        if p not in pos or c not in pos:
            continue
        px, py = pos[p]
        cx, cy = pos[c]
        x1 = px + NODE_W
        y1 = py + NODE_H/2
        x2 = cx
        y2 = cy + NODE_H/2
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#94a3b8" '
                     f'stroke-width="1.2" fill="none" marker-end="url(#arrow)"/>')

    # Nodes — gate_status drives the dominant visual (thick border + corner badge)
    for n, (x, y) in pos.items():
        meta = nodes[n]
        fill = '#fff'
        gate = meta.get('gate', 'unknown')
        gate_c = gate_color.get(gate, '#94a3b8')
        # Tint the fill subtly by gate so the user can scan from across the room
        if gate == 'open':
            fill = '#f0fdf4'
        elif gate == 'conditional':
            fill = '#fffbeb'
        elif gate == 'hold':
            fill = '#fff7ed'
        elif gate == 'ready':
            fill = '#eff6ff'
        elif gate == 'blocked':
            fill = '#fef2f2'
        parts.append(f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
                     f'fill="{fill}" stroke="{gate_c}" stroke-width="3" rx="6"/>')
        # Gate badge — top-right corner, large + bold
        badge_x = x + NODE_W - 95
        parts.append(f'<rect x="{badge_x}" y="{y+5}" width="90" height="20" rx="3" '
                     f'fill="{gate_c}" fill-opacity="0.95"/>')
        parts.append(f'<text x="{badge_x+45}" y="{y+19}" text-anchor="middle" '
                     f'fill="white" font-weight="700" font-size="11">'
                     f'{gate_label.get(gate, gate.upper())}</text>')
        # Name (truncate)
        name_short = n.replace('-', ' ').replace('dnaa ', '').replace('expression dynamics', 'expr')
        if len(name_short) > 30:
            name_short = name_short[:27] + '…'
        parts.append(f'<text x="{x+10}" y="{y+20}" font-weight="600" fill="#0f172a">{name_short}</text>')
        # Phase + status pills
        parts.append(f'<rect x="{x+10}" y="{y+30}" width="55" height="18" rx="9" '
                     f'fill="{phase_color.get(meta["phase"], "#94a3b8")}" fill-opacity="0.18" '
                     f'stroke="{phase_color.get(meta["phase"], "#94a3b8")}"/>')
        parts.append(f'<text x="{x+37}" y="{y+43}" text-anchor="middle" fill="{phase_color.get(meta["phase"], "#94a3b8")}" '
                     f'font-size="10" font-weight="600">{meta["phase"]}</text>')
        parts.append(f'<rect x="{x+72}" y="{y+30}" width="55" height="18" rx="9" '
                     f'fill="{status_color.get(meta["status"], "#cbd5e1")}" fill-opacity="0.2" '
                     f'stroke="{status_color.get(meta["status"], "#cbd5e1")}"/>')
        parts.append(f'<text x="{x+99}" y="{y+43}" text-anchor="middle" fill="{status_color.get(meta["status"], "#64748b")}" '
                     f'font-size="10" font-weight="600">{meta["status"]}</text>')
        # Counts
        parts.append(f'<text x="{x+10}" y="{y+66}" fill="#64748b" font-size="11">'
                     f'{meta["findings"]} findings · {meta["runs"]} runs</text>')

    # Legend at bottom
    legend_y = H - 30
    parts.append(f'<text x="30" y="{legend_y}" fill="#475569" font-size="11" font-weight="600">Gate status: </text>')
    lx = 120
    for state in ['open', 'conditional', 'hold', 'ready', 'blocked']:
        parts.append(f'<rect x="{lx}" y="{legend_y-12}" width="14" height="14" '
                     f'fill="{gate_color[state]}" rx="2"/>')
        parts.append(f'<text x="{lx+18}" y="{legend_y}" fill="#475569" font-size="11">{state}</text>')
        lx += 100

    parts.append('</svg>')
    (OUT_DIR / 'viz' / '08_investigation_dag.svg').write_text('\n'.join(parts))


if __name__ == '__main__':
    print('Generating extra charts ...')
    make_te_percentile()
    print('  ✓ 04_dnaa_te_percentile.svg')
    make_atp_fraction_chart()
    print('  ✓ 06_dnaa_atp_fraction.svg')
    make_findings_overview()
    print('  ✓ 07_findings_overview.svg')
    make_investigation_dag()
    print('  ✓ 08_investigation_dag.svg')
