"""Additional per-study charts for the more recent studies (dnaa-03..06).

These supplement per_study_charts.py with study-specific diagrams that
illustrate the biology more concretely.

Adds:
  dnaa-03: genome-coordinate scatter of DnaA boxes; cooperativity prediction chart
  dnaa-04: code-diff diagram of the swap point; cell-state-at-t=0 schematic
  dnaa-05: pathway-contribution stacked bar; flux-window operating region
  dnaa-06: SeqA sequestration timing diagram; proteome inventory bar
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))


def write_chart(study_dir: Path, slug: str, svg: str, title: str, caption: str):
    cd = study_dir / 'charts'
    cd.mkdir(exist_ok=True)
    (cd / f'{slug}.svg').write_text(svg)
    (cd / f'{slug}.meta.json').write_text(json.dumps({'title': title, 'caption': caption}, indent=2))
    print(f'  ✓ {study_dir.name}/charts/{slug}.svg')


def svg_header(W, H, title, subtitle=''):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" '
        f'font-size="15" fill="#0f172a">{title}</text>',
    ]
    if subtitle:
        parts.append(f'<text x="{W/2}" y="40" text-anchor="middle" '
                     f'fill="#64748b" font-size="11">{subtitle}</text>')
    return parts


# ─── dnaa-03: genome coordinate scatter ─────────────────────────────────

def dnaa_03_genome_map():
    """Read box-binding probe data + draw 456 boxes around the 4.6Mb genome."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    from process_bigraph import Composite
    import numpy as np

    print('  building composite to read box coords...')
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir=str(V2 / 'out' / 'cache'))
    comp = Composite(doc, core=core)
    u = comp.state['agents']['0']['unique']
    boxes = u['DnaA_box']
    coords = boxes['coordinates']  # int, ~ [-2.3M, +2.3M]
    domain_idx = boxes['domain_index']
    n = len(coords)

    # Convert to genome-circular polar coordinates
    GENOME_LEN = 4_641_652   # E. coli MG1655
    # coords are signed bp from origin; convert to angle
    angles = (np.asarray(coords) / GENOME_LEN) * 2 * math.pi  # in radians

    W, H = 720, 720
    CX, CY = W // 2, H // 2 + 20
    R_OUTER = 280
    R_INNER = 250

    parts = svg_header(W, H,
        'DnaA-box positions around the 4.6 Mb E. coli chromosome (n=456)',
        f'Circular layout; ring outside = chromosomal positions, color by domain. oriC at top (12 o\'clock).')

    # Backbone circle
    parts.append(f'<circle cx="{CX}" cy="{CY}" r="{R_OUTER-15}" '
                 f'fill="none" stroke="#cbd5e1" stroke-width="2"/>')
    # oriC marker
    parts.append(f'<text x="{CX}" y="{CY-R_OUTER-2}" text-anchor="middle" '
                 f'fill="#dc2626" font-weight="700" font-size="13">oriC ▲</text>')
    parts.append(f'<text x="{CX}" y="{CY+R_OUTER+18}" text-anchor="middle" '
                 f'fill="#64748b" font-size="13">ter ▼</text>')

    # Domain colors
    domain_colors = {0: '#2563eb', 1: '#16a34a', 2: '#dc2626'}

    # Boxes as small lines outside the circle
    for ang, di in zip(angles, domain_idx):
        c = domain_colors.get(int(di), '#94a3b8')
        # Position on the ring
        x1 = CX + R_INNER * math.sin(ang)
        y1 = CY - R_INNER * math.cos(ang)
        x2 = CX + (R_OUTER) * math.sin(ang)
        y2 = CY - (R_OUTER) * math.cos(ang)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{c}" stroke-width="1.2" opacity="0.7"/>')

    # Domain legend + count
    domain_counts = {0: 0, 1: 0, 2: 0}
    for di in domain_idx:
        domain_counts[int(di)] = domain_counts.get(int(di), 0) + 1
    lx = 40
    ly = H - 50
    parts.append(f'<text x="{lx}" y="{ly}" fill="#475569" font-weight="600">Domain:</text>')
    lx += 80
    for di in (0, 1, 2):
        c = domain_colors.get(di, '#94a3b8')
        parts.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{c}"/>')
        parts.append(f'<text x="{lx+22}" y="{ly}" fill="#475569">'
                     f'domain {di}: {domain_counts.get(di, 0)} boxes</text>')
        lx += 160

    # Inset numbers
    parts.append(f'<text x="{CX}" y="{CY-10}" text-anchor="middle" '
                 f'fill="#0f172a" font-weight="700" font-size="36">{n}</text>')
    parts.append(f'<text x="{CX}" y="{CY+18}" text-anchor="middle" '
                 f'fill="#64748b" font-size="13">total DnaA boxes</text>')
    parts.append(f'<text x="{CX}" y="{CY+40}" text-anchor="middle" '
                 f'fill="#64748b" font-size="11">'
                 f'(spec said 322 = 307 chrom + 11 oriC + 4 dnaAp)</text>')

    return '\n'.join(parts) + '\n</svg>'


def dnaa_03_cooperativity():
    """Plot Hill curves at n=2, 5, 10, 20 for an oriC-like site (Kd=200)."""
    W, H = 720, 380
    PL, PR, PT, PB = 60, 220, 50, 60
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    parts = svg_header(W, H,
        'Hill-cooperativity prediction for oriC binding',
        'Probability that an oriC site is bound, vs free DnaA-ATP. Higher Hill n = sharper threshold.')

    # X axis: 0 to 1000 DnaA-ATP molecules
    X_MAX = 1000
    KD = 200  # oriC effective Kd (DnaA-ATP-equivalent)

    def x(v): return PL + (v / X_MAX) * plot_w
    def y(v): return PT + plot_h - v * plot_h

    # Pass band: oriC should stay < 0.2 below ~400 (proteome-aware threshold)
    parts.append(f'<rect x="{PL}" y="{y(1.0)}" width="{plot_w}" '
                 f'height="{y(0.2)-y(1.0)}" fill="#fef9c3" fill-opacity="0.4"/>')
    parts.append(f'<text x="{PL+10}" y="{y(0.95)}" fill="#a16207" font-size="11">'
                 f'desirable: oriC <0.20 until chromosomal saturates</text>')

    # Y axis
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yt = y(tick)
        parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick:.2f}</text>')
    # X axis
    for tick in [0, 200, 400, 600, 800, 1000]:
        xt = x(tick)
        parts.append(f'<text x="{xt}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}</text>')
    parts.append(f'<text x="{(PL+PL+plot_w)/2}" y="{PT+plot_h+38}" text-anchor="middle" fill="#334155">DnaA-ATP count</text>')
    parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">oriC binding probability</text>')

    # Curves
    colors = {2: '#3b82f6', 5: '#f59e0b', 10: '#16a34a', 20: '#dc2626'}
    for n_hill in (2, 5, 10, 20):
        pts = []
        for i in range(101):
            x_val = (i / 100) * X_MAX
            if x_val == 0:
                p = 0
            else:
                p = (x_val ** n_hill) / (KD ** n_hill + x_val ** n_hill)
            pts.append((x_val, p))
        path = 'M ' + ' L '.join(f'{x(xv):.1f},{y(pv):.2f}' for (xv, pv) in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="{colors[n_hill]}" stroke-width="2.2"/>')

    # Legend on the right
    lx = PL + plot_w + 20
    ly = PT + 20
    parts.append(f'<text x="{lx}" y="{ly}" fill="#475569" font-weight="600">Hill n:</text>')
    ly += 24
    for n_hill, c in colors.items():
        parts.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{c}"/>')
        if n_hill == 2:
            note = '(probe result — FAIL)'
        elif n_hill in (5, 10):
            note = '(target — predicted PASS)'
        else:
            note = '(over-sharp)'
        parts.append(f'<text x="{lx+22}" y="{ly}" fill="#475569">n={n_hill} {note}</text>')
        ly += 22
    parts.append(f'<text x="{lx}" y="{ly+10}" fill="#64748b" font-size="11">Kd = {KD} DnaA-ATP</text>')

    return '\n'.join(parts) + '\n</svg>'


# ─── dnaa-04: cell-state diagram + swap-point ─────────────────────────────

def dnaa_04_cell_state():
    """Schematic of the cell state at t=0: 2 oriCs + 2 active replisomes mid-genome."""
    W, H = 720, 480
    CX, CY = W // 2, H // 2 + 20
    R = 200

    parts = svg_header(W, H,
        'Cell state at t=0 in v2ecoli baseline — already mid-replication',
        '2 active oriCs + 2 active replisomes at ±1.33Mb (29% replicated). '
        '10-min sims won\'t capture a new initiation event.')

    # Chromosome circle
    parts.append(f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="none" '
                 f'stroke="#cbd5e1" stroke-width="3"/>')

    # oriC at top (2 of them, side by side)
    parts.append(f'<circle cx="{CX-10}" cy="{CY-R}" r="8" fill="#dc2626"/>')
    parts.append(f'<circle cx="{CX+10}" cy="{CY-R}" r="8" fill="#dc2626"/>')
    parts.append(f'<text x="{CX}" y="{CY-R-16}" text-anchor="middle" '
                 f'fill="#dc2626" font-weight="700">2 oriCs</text>')

    # Replisome forks at ±29% (in angle, ~109° each side of top)
    # Genome circle: 0 = top, +1.33Mb on one side, -1.33Mb on other
    # 1.33/4.64 ≈ 29% of half-genome
    ang = (1331293 / 4_641_652) * 2 * math.pi  # angle from origin
    fx1 = CX + R * math.sin(ang); fy1 = CY - R * math.cos(ang)
    fx2 = CX + R * math.sin(-ang); fy2 = CY - R * math.cos(-ang)
    parts.append(f'<polygon points="{fx1-8},{fy1-4} {fx1+8},{fy1-4} {fx1},{fy1+8}" '
                 f'fill="#2563eb"/>')
    parts.append(f'<polygon points="{fx2-8},{fy2-4} {fx2+8},{fy2-4} {fx2},{fy2+8}" '
                 f'fill="#2563eb"/>')
    parts.append(f'<text x="{fx1+24}" y="{fy1+5}" fill="#2563eb" font-weight="600">replisome ▲</text>')
    parts.append(f'<text x="{fx2-104}" y="{fy2+5}" fill="#2563eb" font-weight="600">replisome ▲</text>')

    # ter at bottom
    parts.append(f'<text x="{CX}" y="{CY+R+22}" text-anchor="middle" fill="#64748b">ter</text>')

    # Annotation box
    parts.append(f'<rect x="{CX-180}" y="{CY+R+45}" width="360" height="90" '
                 f'fill="#f8fafc" stroke="#e5e7eb" rx="4"/>')
    parts.append(f'<text x="{CX-170}" y="{CY+R+65}" fill="#0f172a" font-weight="600">'
                 f'Implication for dnaa-04 validation:</text>')
    parts.append(f'<text x="{CX-170}" y="{CY+R+85}" fill="#475569" font-size="12">'
                 f'• Cell mass 1266 → 1479 fg in 10 min (modest growth)</text>')
    parts.append(f'<text x="{CX-170}" y="{CY+R+102}" fill="#475569" font-size="12">'
                 f'• criticalMassPerOriC: not emitted (early-exit @ n_oriC > 0)</text>')
    parts.append(f'<text x="{CX-170}" y="{CY+R+119}" fill="#475569" font-size="12">'
                 f'• Need ≥60 min sims to capture termination + new initiation</text>')

    return '\n'.join(parts) + '\n</svg>'


def dnaa_04_swap_point():
    """Code-diff visualization of the swap-point change at chromosome_replication.py:244."""
    W, H = 800, 420
    parts = svg_header(W, H,
        'dnaa-04 implementation: replace mass-threshold with DnaA-occupancy trigger',
        'Single-line conditional change at chromosome_replication.py:244. '
        'All surrounding logic (replisome request, fork creation, oriC consumption) unchanged.')

    # Two code blocks side by side
    BLOCK_W = 350
    BLOCK_H = 280
    TOP_Y = 70

    def code_block(x, y, w, h, title, header_color, lines):
        out = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#0f172a" rx="6"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="32" fill="{header_color}" rx="6"/>',
            f'<rect x="{x}" y="{y+16}" width="{w}" height="16" fill="{header_color}"/>',
            f'<text x="{x+w/2}" y="{y+21}" text-anchor="middle" fill="white" font-weight="700">{title}</text>',
        ]
        ly = y + 50
        for code, color in lines:
            esc = code.replace('<', '&lt;').replace('>', '&gt;')
            out.append(f'<text x="{x+12}" y="{ly}" fill="{color}" '
                       f'font-family="ui-monospace,monospace" font-size="12">{esc}</text>')
            ly += 18
        return out

    parts += code_block(40, TOP_Y, BLOCK_W, BLOCK_H, 'BEFORE (current, line 244)', '#dc2626', [
        ('# Calculate mass per origin', '#94a3b8'),
        ('massPerOrigin = cellMass / n_oriC', '#e0e7ff'),
        ('self.criticalMassPerOriC = (', '#e0e7ff'),
        ('    massPerOrigin /', '#e0e7ff'),
        ('    self.criticalInitiationMass', '#e0e7ff'),
        (').to("dimensionless")', '#e0e7ff'),
        ('', '#e0e7ff'),
        ('# Trigger: mass-threshold heuristic', '#94a3b8'),
        ('if self.criticalMassPerOriC >= 1.0:', '#fbbf24'),
        ('    requests["bulk"].append(...)', '#e0e7ff'),
    ])

    parts += code_block(W - BLOCK_W - 40, TOP_Y, BLOCK_W, BLOCK_H, 'AFTER (proposed)', '#16a34a', [
        ('# Read oriC binding state from', '#94a3b8'),
        ('# dnaa-03 DnaA_bound + dnaa-02 ATP pool', '#94a3b8'),
        ('n_bound_at_oriC = sum_oriC_bound(', '#e0e7ff'),
        ('    states["DnaA_boxes"],', '#e0e7ff'),
        ('    states["oriCs"]', '#e0e7ff'),
        (')', '#e0e7ff'),
        ('', '#e0e7ff'),
        ('# Trigger: DnaA-occupancy cooperative filament', '#94a3b8'),
        ('if n_bound_at_oriC >= 11:  # 11 oriC sites', '#fbbf24'),
        ('    requests["bulk"].append(...)', '#e0e7ff'),
    ])

    # Bottom note
    parts.append(f'<text x="{W/2}" y="{TOP_Y + BLOCK_H + 40}" text-anchor="middle" '
                 f'fill="#475569" font-weight="600">The mechanism is biological, not heuristic — '
                 f'replication starts when enough DnaA-ATP is loaded at oriC.</text>')
    parts.append(f'<text x="{W/2}" y="{TOP_Y + BLOCK_H + 60}" text-anchor="middle" '
                 f'fill="#64748b" font-size="11">'
                 f'Implementation hooks: chromosome_initiation.DnaABinder.update() (currently empty stub)</text>')

    return '\n'.join(parts) + '\n</svg>'


# ─── dnaa-05: pathway contributions ─────────────────────────────────────

def dnaa_05_pathway_stack():
    """Stacked-bar showing intrinsic vs each extrinsic pathway contribution."""
    W, H = 720, 460
    PL, PR, PT, PB = 80, 30, 70, 80
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    parts = svg_header(W, H,
        'DnaA-ATP → DnaA-ADP flux: required vs literature estimates',
        'Each bar shows one source of hydrolysis flux. Y-axis: 1/min per cell.')

    # Data: pathway, literature_min, literature_max, color
    pathways = [
        ('Intrinsic (Boesen 2024)',        0.046, 0.046, '#94a3b8'),
        ('RIDA (replisome-coupled)',       0.50,  1.40,  '#3b82f6'),
        ('DDAH (datA, pulsatile avg)',     0.10,  0.50,  '#f59e0b'),
        ('Total extrinsic (lit. sum)',     0.65,  1.95,  '#16a34a'),
        ('Required (v2ecoli probe F-04)',  4.60,  4.60,  '#dc2626'),
    ]
    n = len(pathways)
    bar_w = plot_w / n * 0.6
    Y_MAX = 5.5
    def x(i): return PL + (i + 0.5) * plot_w / n
    def y(v): return PT + plot_h - (v / Y_MAX) * plot_h

    # Y grid
    for tick in [0, 1, 2, 3, 4, 5]:
        yt = y(tick)
        parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick}</text>')
    parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
                 f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">hydrolysis rate (1/min)</text>')

    # Operating window — shade [1, 50] (we clamp at top of plot)
    parts.append(f'<rect x="{PL}" y="{y(5.0)}" width="{plot_w}" height="{y(1.0)-y(5.0)}" '
                 f'fill="#bbf7d0" fill-opacity="0.3"/>')
    parts.append(f'<text x="{PL+plot_w-10}" y="{y(5.0)+14}" text-anchor="end" '
                 f'fill="#16a34a" font-size="11">probe-validated operating window [1, 50]/min</text>')

    # Bars
    for i, (label, mn, mx, color) in enumerate(pathways):
        cx = x(i)
        bx = cx - bar_w/2
        # Range bar from mn to mx
        if mx > mn:
            parts.append(f'<rect x="{bx}" y="{y(mx)}" width="{bar_w}" '
                         f'height="{y(mn)-y(mx)}" fill="{color}" opacity="0.6"/>')
            parts.append(f'<line x1="{cx}" y1="{y(mn)}" x2="{cx}" y2="{y(mx)}" '
                         f'stroke="#0f172a" stroke-width="1"/>')
        else:
            parts.append(f'<rect x="{bx}" y="{y(mx)}" width="{bar_w}" '
                         f'height="6" fill="{color}"/>')
        # Value label
        parts.append(f'<text x="{cx}" y="{y(mx)-6}" text-anchor="middle" '
                     f'fill="#0f172a" font-weight="600" font-size="11">'
                     f'{mn:.2f}{("–"+f"{mx:.2f}") if mx>mn else ""}</text>')
        # X-axis label (word-wrap)
        words = label.split()
        ly = PT + plot_h + 16
        line = ''
        line_parts = []
        for w in words:
            if len(line) + len(w) + 1 > 22:
                line_parts.append(line.strip()); line = w + ' '
            else:
                line += w + ' '
        if line: line_parts.append(line.strip())
        for j, ln in enumerate(line_parts[:3]):
            parts.append(f'<text x="{cx}" y="{ly + j*13}" text-anchor="middle" '
                         f'fill="#475569" font-size="10">{ln}</text>')

    # Annotation: gap
    parts.append(f'<text x="{W/2}" y="{H-20}" text-anchor="middle" fill="#dc2626" '
                 f'font-weight="600" font-size="12">'
                 f'Gap: literature extrinsic sum (~1.3) is ~3.5× short of v2ecoli\'s 4.6 target → '
                 f'audit equilibrium reverse-rate first.</text>')

    return '\n'.join(parts) + '\n</svg>'


# ─── dnaa-06: SeqA timing diagram ───────────────────────────────────────

def dnaa_06_sequestration_timeline():
    """Timeline showing oriC methylation states + SeqA window."""
    W, H = 800, 380
    PL, PR, PT, PB = 70, 30, 80, 60
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    parts = svg_header(W, H,
        'SeqA sequestration timeline — oriC methylation cycle',
        'After initiation, oriC is hemimethylated for ~10 min. SeqA binds GATC sites in hemimethylated form, blocking re-initiation.')

    # Timeline runs left-to-right, 0-30 min
    T_MIN, T_MAX = 0, 30
    def x(t): return PL + ((t - T_MIN) / (T_MAX - T_MIN)) * plot_w

    # Three lanes: oriC methylation state, DnaA-ATP loading, SeqA bound
    lane_h = (plot_h - 40) / 3
    LY = [PT + 10, PT + 10 + lane_h + 15, PT + 10 + 2*lane_h + 30]
    LH = lane_h - 10

    # Lane 1: oriC methylation state
    # 0-0 = fully methylated, 0-10 = hemimethylated (RED), 10-30 = fully methylated again
    parts.append(f'<text x="{PL-10}" y="{LY[0]+LH/2+4}" text-anchor="end" '
                 f'fill="#475569" font-weight="600">oriC methylation</text>')
    parts.append(f'<rect x="{x(0)}" y="{LY[0]}" width="{x(0.5)-x(0)}" '
                 f'height="{LH}" fill="#16a34a" opacity="0.7"/>')  # fully meth
    parts.append(f'<rect x="{x(0.5)}" y="{LY[0]}" width="{x(10)-x(0.5)}" '
                 f'height="{LH}" fill="#dc2626" opacity="0.7"/>')   # hemimeth
    parts.append(f'<rect x="{x(10)}" y="{LY[0]}" width="{x(30)-x(10)}" '
                 f'height="{LH}" fill="#16a34a" opacity="0.7"/>')   # remeth
    # Labels in lanes
    parts.append(f'<text x="{(x(0.5)+x(10))/2}" y="{LY[0]+LH/2+4}" text-anchor="middle" '
                 f'fill="white" font-weight="600">hemimethylated (~10 min after replication)</text>')
    parts.append(f'<text x="{(x(10)+x(30))/2}" y="{LY[0]+LH/2+4}" text-anchor="middle" '
                 f'fill="white" font-weight="600">fully methylated</text>')

    # Lane 2: SeqA bound
    parts.append(f'<text x="{PL-10}" y="{LY[1]+LH/2+4}" text-anchor="end" '
                 f'fill="#475569" font-weight="600">SeqA bound</text>')
    parts.append(f'<rect x="{x(0)}" y="{LY[1]}" width="{x(0.5)-x(0)}" '
                 f'height="{LH}" fill="#cbd5e1" opacity="0.7"/>')
    parts.append(f'<rect x="{x(0.5)}" y="{LY[1]}" width="{x(10)-x(0.5)}" '
                 f'height="{LH}" fill="#2563eb" opacity="0.7"/>')
    parts.append(f'<rect x="{x(10)}" y="{LY[1]}" width="{x(30)-x(10)}" '
                 f'height="{LH}" fill="#cbd5e1" opacity="0.7"/>')
    parts.append(f'<text x="{(x(0.5)+x(10))/2}" y="{LY[1]+LH/2+4}" text-anchor="middle" '
                 f'fill="white" font-weight="600">YES (blocks re-initiation)</text>')
    parts.append(f'<text x="{(x(10)+x(30))/2}" y="{LY[1]+LH/2+4}" text-anchor="middle" '
                 f'fill="#475569" font-weight="600">no</text>')

    # Lane 3: re-initiation possible?
    parts.append(f'<text x="{PL-10}" y="{LY[2]+LH/2+4}" text-anchor="end" '
                 f'fill="#475569" font-weight="600">re-initiation possible?</text>')
    parts.append(f'<rect x="{x(0)}" y="{LY[2]}" width="{x(0.5)-x(0)}" '
                 f'height="{LH}" fill="#16a34a" opacity="0.5"/>')
    parts.append(f'<rect x="{x(0.5)}" y="{LY[2]}" width="{x(10)-x(0.5)}" '
                 f'height="{LH}" fill="#dc2626" opacity="0.7"/>')
    parts.append(f'<rect x="{x(10)}" y="{LY[2]}" width="{x(30)-x(10)}" '
                 f'height="{LH}" fill="#16a34a" opacity="0.5"/>')
    parts.append(f'<text x="{(x(0.5)+x(10))/2}" y="{LY[2]+LH/2+4}" text-anchor="middle" '
                 f'fill="white" font-weight="600">NO (blocked)</text>')
    parts.append(f'<text x="{(x(10)+x(30))/2}" y="{LY[2]+LH/2+4}" text-anchor="middle" '
                 f'fill="#0f172a" font-weight="600">YES (after Dam re-methylates)</text>')

    # Initiation marker (top)
    parts.append(f'<line x1="{x(0)}" y1="{PT-10}" x2="{x(0)}" y2="{PT+plot_h+8}" '
                 f'stroke="#0f172a" stroke-width="2" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{x(0)}" y="{PT-15}" text-anchor="middle" fill="#0f172a" '
                 f'font-weight="700" font-size="12">initiation fires (t=0)</text>')

    # X-axis ticks
    for tick in [0, 5, 10, 15, 20, 25, 30]:
        parts.append(f'<text x="{x(tick)}" y="{PT+plot_h+22}" text-anchor="middle" '
                     f'fill="#64748b">{tick} min</text>')

    # Bottom note
    parts.append(f'<text x="{W/2}" y="{H-20}" text-anchor="middle" fill="#475569" '
                 f'font-weight="600" font-size="12">'
                 f'Implementation: add `sequestered_until: float` to oriC; SeqA-Step '
                 f'sets it on initiation; DnaABinder gate respects it.</text>')

    return '\n'.join(parts) + '\n</svg>'


def dnaa_06_proteome_inventory():
    """Bar chart showing which proteins exist in proteome for SeqA sequestration."""
    W, H = 720, 380
    PL, PR, PT, PB = 80, 30, 60, 80
    plot_w = W - PL - PR
    plot_h = H - PT - PB

    parts = svg_header(W, H,
        'Proteome inventory for SeqA sequestration mechanism',
        'All 4 proteins needed for the sequestration mechanism already exist in v2ecoli\'s proteome — just need a Process that uses them.')

    # Data
    proteins = [
        ('SeqA',  'EG12197', 181, '#16a34a', True,  'negative modulator of initiation'),
        ('Dam',   'EG10204', 278, '#16a34a', True,  'DNA adenine methyltransferase'),
        ('HspQ',  'G6500',   104, '#16a34a', True,  'hemimethylated-DNA binding'),
        ('DamX',  'EG11183', 428, '#16a34a', True,  'cell-division-coupled Dam regulator'),
        ('SeqASequestration Step', '—', 0, '#dc2626', False, 'NOT IMPLEMENTED'),
        ('oriC.sequestered_until', '—', 0, '#dc2626', False, 'attribute MISSING'),
    ]
    n = len(proteins)
    bar_h = (plot_h - 20) / n - 6
    Y_VALUE_MAX = 500  # protein length

    def x(v): return PL + (v / Y_VALUE_MAX) * plot_w
    def y(i): return PT + i * (bar_h + 6) + 10

    for i, (name, eg, length, color, present, note) in enumerate(proteins):
        # Label
        parts.append(f'<text x="{PL-10}" y="{y(i)+bar_h/2+4}" text-anchor="end" '
                     f'fill="#0f172a" font-weight="600">{name}</text>')
        if length > 0:
            parts.append(f'<rect x="{PL}" y="{y(i)}" width="{x(length)-PL}" '
                         f'height="{bar_h}" fill="{color}" opacity="0.7"/>')
            parts.append(f'<text x="{x(length)+8}" y="{y(i)+bar_h/2+4}" '
                         f'fill="#475569" font-size="11">'
                         f'{eg} · {length} aa · ✓ in proteome</text>')
        else:
            parts.append(f'<rect x="{PL}" y="{y(i)}" width="200" '
                         f'height="{bar_h}" fill="{color}" opacity="0.3" stroke="{color}"/>')
            parts.append(f'<text x="{PL+8}" y="{y(i)+bar_h/2+4}" fill="{color}" '
                         f'font-weight="600">{note}</text>')

    # Bottom note
    parts.append(f'<text x="{W/2}" y="{H-25}" text-anchor="middle" fill="#475569" '
                 f'font-weight="600" font-size="12">'
                 f'Implementation effort: ~1 day. Smallest piece of the dnaA upstream chain.</text>')

    return '\n'.join(parts) + '\n</svg>'


def main():
    print('Generating extra per-study charts for dnaa-03..06...')

    sd = V2 / 'studies' / 'dnaa-03-box-binding'
    print(f'\n=== {sd.name} ===')
    write_chart(sd, '02_genome_map', dnaa_03_genome_map(),
        'DnaA-box positions around the genome',
        '456 boxes spread across the 4.6Mb circular chromosome; colored by replication domain. '
        'Spec said 322 (Roth1998 curated); v2ecoli\'s motif-scan finds more putative sites.')
    write_chart(sd, '03_cooperativity_curves', dnaa_03_cooperativity(),
        'Hill cooperativity prediction for oriC binding',
        'Probability oriC site is bound vs DnaA-ATP count, at Hill exponents n=2/5/10/20. '
        'n=2 (probe result) gives gradual fill — wrong. n=5-10 (target) gives sharp threshold = textbook two-step pattern.')

    sd = V2 / 'studies' / 'dnaa-04-initiation-mechanism'
    print(f'\n=== {sd.name} ===')
    write_chart(sd, '01_cell_state_t0', dnaa_04_cell_state(),
        'Cell state at t=0: mid-replication',
        'In v2ecoli baseline, the initial state already has 2 active oriCs + 2 replisomes mid-elongation. '
        '10-min sims won\'t capture a new initiation event — need ≥60 min runs.')
    write_chart(sd, '02_swap_point_diff', dnaa_04_swap_point(),
        'Swap point: mass-threshold → DnaA-occupancy',
        'The conditional at chromosome_replication.py:244 is the single line that changes. '
        'Surrounding logic (replisome request, fork creation) unchanged.')

    sd = V2 / 'studies' / 'dnaa-05-rida-ddah-dars'
    print(f'\n=== {sd.name} ===')
    write_chart(sd, '02_pathway_contributions', dnaa_05_pathway_stack(),
        'Pathway contributions vs required total flux',
        'Each bar is a hydrolysis-flux source. The v2ecoli probe-derived target (4.6/min, F-04) is ~3.5× higher than '
        'the literature sum — suggests v2ecoli\'s equilibrium reverse-rate is mis-calibrated and needs audit first.')

    sd = V2 / 'studies' / 'dnaa-06-seqa-sequestration'
    print(f'\n=== {sd.name} ===')
    write_chart(sd, '01_sequestration_timeline', dnaa_06_sequestration_timeline(),
        'SeqA sequestration timing — 10-min post-initiation window',
        'After initiation, oriC sits hemimethylated for ~10 min (Lu 1994, Kang 2003). SeqA binds the GATC sites '
        'in the hemimethylated state and prevents re-initiation. Once Dam re-methylates both strands, SeqA releases '
        'and re-initiation becomes possible.')
    write_chart(sd, '02_proteome_inventory', dnaa_06_proteome_inventory(),
        'Proteome inventory — all 4 sequestration proteins already present',
        'SeqA, Dam, HspQ, DamX are all in v2ecoli\'s proteome. The implementation gap is just the Process + '
        'one oriC attribute (sequestered_until).')

    print('\nDone. Run dashboard refresh to see updated charts.')


if __name__ == '__main__':
    main()
