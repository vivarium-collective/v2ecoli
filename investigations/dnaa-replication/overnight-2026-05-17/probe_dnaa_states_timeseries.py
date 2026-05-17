"""Probe: DnaA-ATP / DnaA-ADP / apo trajectory over time.

For dnaa-02. Runs a baseline simulation and records (apo, ATP-bound, ADP-bound)
bulk counts at every step. Writes a JSON time-series + an SVG line chart.

Why: F-01 + the static probe showed DnaA distribution at one point. To see the
EQUILIBRATION DYNAMICS — does it converge to 0.99 ATP-bound in 30s? 5 min?
— we need the trajectory.

Output:
  investigations/dnaa-replication/overnight-2026-05-17/viz/05_dnaa_states_timeseries.svg
  investigations/dnaa-replication/overnight-2026-05-17/dnaa_states_timeseries.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
from v2ecoli.core import build_core
from v2ecoli.composites.baseline import baseline
from process_bigraph import Composite
import numpy as np

DURATION_S = 10 * 60    # 10 min
SAMPLE_EVERY_S = 10     # sample bulk every 10s of simulated time
SEED = 0
OUT_DIR = Path(__file__).resolve().parent

core = build_core()
print('[probe] building composite ...', flush=True)
doc = baseline(core=core, seed=SEED, cache_dir='out/cache')

agent0 = doc['state']['agents']['0']
bulk0 = agent0['bulk']
ids = bulk0['id']
APO = int(np.where(ids == 'PD03831[c]')[0][0])
ATP = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
ADP = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])

comp = Composite(doc, core=core)

samples = []  # (sim_t, apo, atp, adp)

def sample():
    """Read live bulk state."""
    bulk = comp.state['agents']['0']['bulk']
    return (
        int(bulk['count'][APO]),
        int(bulk['count'][ATP]),
        int(bulk['count'][ADP]),
    )

# initial
a, t, p = sample()
samples.append((0.0, a, t, p))
print(f'[probe] t=0:    apo={a:>4}  ATP={t:>4}  ADP={p:>4}  total={a+t+p}')

print(f'[probe] simulating {DURATION_S}s in {SAMPLE_EVERY_S}s chunks ...', flush=True)
t0 = time.time()
sim_t = 0.0
while sim_t < DURATION_S:
    comp.update({}, SAMPLE_EVERY_S)
    sim_t += SAMPLE_EVERY_S
    a, t, p = sample()
    samples.append((sim_t, a, t, p))
    if int(sim_t) % 60 == 0:
        print(f'[probe] t={sim_t:>4.0f}s  apo={a:>4}  ATP={t:>4}  ADP={p:>4}  '
              f'total={a+t+p}  atp_frac={t/(a+t+p) if (a+t+p) > 0 else 0:.3f}  '
              f'  elapsed_wall={time.time()-t0:.1f}s', flush=True)
print(f'[probe] DONE in {time.time()-t0:.1f}s wall')

# Persist JSON
out_json = {
    'description': 'DnaA-ATP / DnaA-ADP / apo trajectory for baseline composite (1× TE, seed 0)',
    'sample_every_s': SAMPLE_EVERY_S,
    'duration_s': DURATION_S,
    'seed': SEED,
    'samples': [{'t': t, 'apo': a, 'atp': p, 'adp': d, 'total': a+p+d,
                 'atp_frac': p/(a+p+d) if (a+p+d) > 0 else 0.0}
                for (t, a, p, d) in samples],
}
(OUT_DIR / 'dnaa_states_timeseries.json').write_text(json.dumps(out_json, indent=2))
print(f'[probe] wrote dnaa_states_timeseries.json')

# Build SVG
W, H = 900, 480
PL, PR, PT, PB = 70, 30, 50, 60
plot_w = W - PL - PR
plot_h = H - PT - PB
totals = [a+p+d for (_,a,p,d) in samples]
y_max = max(totals) * 1.15
ts = [t for (t,*_) in samples]
def x(tv): return PL + (tv / DURATION_S) * plot_w
def y(v): return PT + plot_h - (v / y_max) * plot_h

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">'
         f'<rect width="{W}" height="{H}" fill="white"/>'
         f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
         f'DnaA equilibration: apo / DnaA-ATP / DnaA-ADP over time (baseline, 1× TE)</text>']

# Y axis ticks
for tick in [0, 50, 100, 150, 200]:
    if tick > y_max:
        break
    yt = y(tick)
    parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick}</text>')

# X axis ticks
for tick in [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
    if tick > DURATION_S:
        break
    xt = x(tick)
    parts.append(f'<text x="{xt}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}s</text>')

# Series
series = [
    ('apo (PD03831)', [(t, a) for (t,a,p,d) in samples], '#3b82f6'),
    ('DnaA-ATP (MONOMER0-160)', [(t, p) for (t,a,p,d) in samples], '#10b981'),
    ('DnaA-ADP (MONOMER0-4565)', [(t, d) for (t,a,p,d) in samples], '#f59e0b'),
    ('total', [(t, a+p+d) for (t,a,p,d) in samples], '#1e293b'),
]
for label, pts, color in series:
    path = 'M ' + ' L '.join(f'{x(t):.1f},{y(v):.1f}' for (t,v) in pts)
    sw = '2' if label != 'total' else '2.5'
    da = ' stroke-dasharray="4,3"' if label == 'total' else ''
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{da}/>')

# Legend
ly = 50
for label, _, color in series:
    parts.append(f'<rect x="{PL}" y="{ly-9}" width="14" height="14" fill="{color}"/>')
    parts.append(f'<text x="{PL+22}" y="{ly+3}" fill="#334155">{label}</text>')
    ly += 18

# Boesen 2024 reference band [0.2, 0.5] for ATP fraction — overlaid as text annotation
final_atp_frac = (samples[-1][2] / max(samples[-1][1]+samples[-1][2]+samples[-1][3], 1))
parts.append(f'<text x="{W-PR-10}" y="{PT+30}" text-anchor="end" fill="#dc2626" font-size="12">'
             f'final ATP fraction = {final_atp_frac:.3f}</text>')
parts.append(f'<text x="{W-PR-10}" y="{PT+46}" text-anchor="end" fill="#dc2626" font-size="11">'
             f'Boesen 2024 target [0.20, 0.50] — v2ecoli is {final_atp_frac/0.35:.1f}× over</text>')

parts.append('</svg>')
viz_dir = OUT_DIR / 'viz'
viz_dir.mkdir(exist_ok=True)
(viz_dir / '05_dnaa_states_timeseries.svg').write_text('\n'.join(parts))
print(f'[probe] wrote viz/05_dnaa_states_timeseries.svg')
