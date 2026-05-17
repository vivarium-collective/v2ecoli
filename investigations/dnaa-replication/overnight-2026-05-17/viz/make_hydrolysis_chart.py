"""Render 12_hydrolysis_rate_sensitivity.svg — bar chart of ATP fraction vs
hydrolysis rate, with Boesen [0.20, 0.50] band highlighted.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
data = json.loads((OUT / 'hydrolysis_rate_sensitivity.json').read_text())

W, H = 760, 420
PL, PR, PT, PB = 70, 30, 60, 80
plot_w = W - PL - PR
plot_h = H - PT - PB

mults = [d['rate_multiplier'] for d in data]
fracs = [d['median_atp_frac'] for d in data]
totals = [d['median_total'] for d in data]

bar_w = plot_w / len(mults) * 0.55

def x(i): return PL + (i + 0.5) * plot_w / len(mults)
def y(v): return PT + plot_h - v * plot_h

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
    f'DnaA-ATP fraction vs hydrolysis rate (sensitivity probe, dnaa-02)</text>',
    f'<text x="{W/2}" y="40" text-anchor="middle" fill="#64748b" font-size="11">'
    f'Calibration: TE=20×, fc=0.7. Rate-multipliers relative to Boesen 0.046/min intrinsic.</text>',
]

# Pass band
parts.append(f'<rect x="{PL}" y="{y(0.5)}" width="{plot_w}" height="{y(0.2)-y(0.5)}" '
             f'fill="#86efac" fill-opacity="0.3"/>')
parts.append(f'<text x="{PL+10}" y="{y(0.5)+14}" fill="#16a34a" font-size="11">'
             f'Boesen 2024 physiological band [0.20, 0.50]</text>')

# Y axis
for tick in [0, 0.25, 0.5, 0.75, 1.0]:
    yt = y(tick)
    parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick:.2f}</text>')
parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
             f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">DnaA-ATP / total</text>')

# Bars
for i, (m, f, tot) in enumerate(zip(mults, fracs, totals)):
    cx = x(i)
    bar_x = cx - bar_w/2
    bar_top = y(f)
    bar_bottom = y(0)
    passes = 0.20 <= f <= 0.50
    color = '#22c55e' if passes else '#94a3b8'
    parts.append(f'<rect x="{bar_x}" y="{bar_top}" width="{bar_w}" '
                 f'height="{bar_bottom-bar_top}" fill="{color}"/>')
    # Label
    parts.append(f'<text x="{cx}" y="{bar_top-4}" text-anchor="middle" '
                 f'fill="#0f172a" font-weight="600">{f:.3f}</text>')
    # X label
    parts.append(f'<text x="{cx}" y="{PT+plot_h+18}" text-anchor="middle" '
                 f'fill="#334155">×{m}</text>')
    parts.append(f'<text x="{cx}" y="{PT+plot_h+34}" text-anchor="middle" '
                 f'fill="#64748b" font-size="11">k={data[i]["k_per_min"]:.2f}/min</text>')
    parts.append(f'<text x="{cx}" y="{PT+plot_h+50}" text-anchor="middle" '
                 f'fill="#64748b" font-size="11">total={tot}</text>')
    star = ' ★ PASS' if passes else ''
    if star:
        parts.append(f'<text x="{cx}" y="{PT+plot_h+66}" text-anchor="middle" '
                     f'fill="#16a34a" font-weight="600" font-size="12">{star}</text>')

# Caption
parts.append(f'<text x="{W/2}" y="{H-10}" text-anchor="middle" fill="#475569" font-size="11">'
             f'Intrinsic-only (×1) fails; 100× rate (≈ RIDA+DDAH+DARS combined) passes; 1000× over-corrects to 0%.</text>')
parts.append('</svg>')

(OUT / 'viz' / '12_hydrolysis_rate_sensitivity.svg').write_text('\n'.join(parts))
print(f'Wrote {OUT}/viz/12_hydrolysis_rate_sensitivity.svg')
