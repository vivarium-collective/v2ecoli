"""dnaa-03 simulation probe: does a simple Hill-binding model on the 456 boxes
reproduce the textbook two-step oriC-occupancy pattern (oriC unbound while
chromosomal sites titrate, then sharp fill)?

Method: at each sample interval, read bulk[MONOMER0-160] (DnaA-ATP count).
For each box, draw bound/unbound by Hill probability with Kd assigned by
class:
  - 11 high-affinity oriC sites (Kd = 1 nM ≈ ~5 DnaA-ATP at cell volume)
  - 295 low-affinity chromosomal sites (Kd = 100 nM ≈ ~500 DnaA-ATP)
  - ~150 medium-affinity sites (Kd = 20 nM)

(The classification is heuristic — we don't have per-box affinities. The goal
is to see if such a model PRODUCES the textbook pattern given correct
DnaA-ATP totals.)

Output:
  box_binding_timeseries.json
  viz/13_box_occupancy_two_step.svg
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.composites.baseline import baseline
from process_bigraph import Composite
import numpy as np
from wholecell.utils.random import stochasticRound

# Use the validated calibration so DnaA-ATP rises through physiological range
DURATION_S = 10 * 60
SAMPLE_EVERY_S = 10
SEED = 0
TE_MULTIPLIER = 20.0
AUTOREP_MULTIPLIER = 0.7

# Hill-binding params
# Kd values in molecules-per-cell units (more useful than nM here)
# Roughly: at cell vol 1e-15 L, 1 nM ≈ 0.6 molecules; 100 nM ≈ 60. But the
# practically meaningful threshold is the count at which 50% of boxes bind.
# Hansen 2018 + Speck 1999 + Saggioro 2013 suggest oriC ~5x affinity vs
# chromosomal. We use: high Kd=5, medium Kd=50, low Kd=300.
N_BOXES_HIGH = 11    # oriC + some dnaAp = 11+4 = 15, round to 11 strict oriC
N_BOXES_MED = 50     # dnaAp + nearby
N_BOXES_LOW = 395    # rest (456 - 11 - 50)
KD_HIGH = 5.0
KD_MED = 50.0
KD_LOW = 300.0
HILL_N = 2.0         # cooperative binding

OUT = Path(__file__).resolve().parent

# ── Build composite + apply calibration ──────────────────────────────────
core = build_core()
bundle = load_cache_bundle(str(V2 / 'out' / 'cache'))
tf_cfg = bundle['configs']['ecoli-tf-binding']
tf_cfg['emit_n_bound_TF_per_TU'] = True
pi_cfg = bundle['configs']['ecoli-polypeptide-initiation']
pi_cfg['translation_efficiencies'][3861] *= TE_MULTIPLIER
dp = tf_cfg['delta_prob']
deltaV = np.asarray(dp['deltaV'])
mask = (np.asarray(dp['deltaJ']) == 12)
deltaV[mask] = deltaV[mask] * AUTOREP_MULTIPLIER
dp['deltaV'] = deltaV

print(f'[probe-binding] config: TE={TE_MULTIPLIER}, fc={AUTOREP_MULTIPLIER}, '
      f'high/med/low Kd={KD_HIGH}/{KD_MED}/{KD_LOW}, Hill n={HILL_N}', flush=True)

print('[probe-binding] building composite...', flush=True)
doc = baseline(core=core, seed=SEED, cache_dir=str(V2 / 'out' / 'cache'))
comp = Composite(doc, core=core)
bulk0 = comp.state['agents']['0']['bulk']
ids = bulk0['id']
ATP_IDX = int(np.where(ids == 'MONOMER0-160[c]')[0][0])

# ── Assign affinity class per box (synthetic) ───────────────────────────
u = comp.state['agents']['0']['unique']
boxes = u['DnaA_box']
n_boxes_active = int(boxes['_entryState'].astype(bool).sum())
print(f'[probe-binding] n_boxes_active = {n_boxes_active}')

rng = np.random.RandomState(SEED)
# We can't truly label which boxes are oriC vs not without coordinates+oriC
# location, so partition by index. Random assignment is biologically meaningless
# but lets us test the Hill mechanics.
classes = np.array(['low'] * n_boxes_active, dtype=object)
shuffled = rng.permutation(n_boxes_active)
classes[shuffled[:N_BOXES_HIGH]] = 'high'
classes[shuffled[N_BOXES_HIGH:N_BOXES_HIGH+N_BOXES_MED]] = 'med'
class_to_kd = {'high': KD_HIGH, 'med': KD_MED, 'low': KD_LOW}
kds = np.array([class_to_kd[c] for c in classes])

# ── Run loop ────────────────────────────────────────────────────────────
samples = []

def occupancy_by_class(bound_array, classes):
    out = {}
    for c in ('high', 'med', 'low'):
        m = (classes == c)
        n = m.sum()
        if n == 0: continue
        out[c] = {'n_total': int(n), 'n_bound': int(bound_array[m].sum()),
                  'frac': float(bound_array[m].sum() / n)}
    return out

print(f'[probe-binding] simulating {DURATION_S}s in {SAMPLE_EVERY_S}s chunks...', flush=True)
t0 = time.time()
sim_t = 0.0
bound = np.zeros(n_boxes_active, dtype=bool)  # external state

while sim_t < DURATION_S:
    comp.update({}, SAMPLE_EVERY_S)
    sim_t += SAMPLE_EVERY_S
    # Re-derive bound state from current DnaA-ATP via Hill
    bulk = comp.state['agents']['0']['bulk']
    n_atp = float(bulk['count'][ATP_IDX])
    # Hill probability = n_atp^n / (Kd^n + n_atp^n)
    p_bind = np.power(n_atp, HILL_N) / (np.power(kds, HILL_N) + np.power(n_atp, HILL_N))
    bound = (rng.random(n_boxes_active) < p_bind)
    occ = occupancy_by_class(bound, classes)
    samples.append({'t': sim_t, 'n_atp': int(n_atp), 'occupancy': occ})
    if int(sim_t) % 60 == 0:
        h = occ.get('high', {}).get('frac', 0)
        m = occ.get('med', {}).get('frac', 0)
        l = occ.get('low', {}).get('frac', 0)
        print(f"[probe-binding] t={sim_t:>4.0f}s  ATP={n_atp:>4.0f}  "
              f"oriC(high)={h:.2f}  med={m:.2f}  chrom(low)={l:.2f}  "
              f"elapsed_wall={time.time()-t0:.0f}s", flush=True)

print(f'[probe-binding] DONE in {time.time()-t0:.1f}s wall')

# ── Analyze: does the two-step pattern appear? ──────────────────────────
# Textbook: chromosomal saturates BEFORE oriC; oriC stays unbound until
# chromosomal >90% occupied.
ts = np.array([s['t'] for s in samples])
atps = np.array([s['n_atp'] for s in samples])
high_occ = np.array([s['occupancy'].get('high', {}).get('frac', 0) for s in samples])
med_occ = np.array([s['occupancy'].get('med', {}).get('frac', 0) for s in samples])
low_occ = np.array([s['occupancy'].get('low', {}).get('frac', 0) for s in samples])

# Find the timepoint where low_occ first crosses 0.5 (chromosomal half-occupied)
half_cross = ts[np.argmax(low_occ >= 0.5)] if (low_occ >= 0.5).any() else None
# Find where high_occ crosses 0.5 (oriC half-occupied)
high_cross = ts[np.argmax(high_occ >= 0.5)] if (high_occ >= 0.5).any() else None

print(f'\n=== Two-step occupancy analysis ===')
print(f'Final occupancy: high={high_occ[-1]:.3f}, med={med_occ[-1]:.3f}, low={low_occ[-1]:.3f}')
print(f'DnaA-ATP range: {atps.min()}..{atps.max()}')
if half_cross is not None and high_cross is not None:
    delay = high_cross - half_cross
    print(f'Chromosomal half-cross: t={half_cross}s  |  oriC half-cross: t={high_cross}s  |  delay: {delay}s')
    print(f'Two-step pattern: {"YES (oriC fills AFTER chromosomal)" if delay > 0 else "NO (oriC fills first or simultaneously)"}')

out_json = {
    'description': 'External binding-Hill-model probe (dnaa-03)',
    'params': {'TE_multiplier': TE_MULTIPLIER, 'autorep_multiplier': AUTOREP_MULTIPLIER,
               'Kd_high': KD_HIGH, 'Kd_med': KD_MED, 'Kd_low': KD_LOW, 'Hill_n': HILL_N,
               'n_boxes_high': N_BOXES_HIGH, 'n_boxes_med': N_BOXES_MED, 'n_boxes_low': N_BOXES_LOW},
    'samples': samples,
}
(OUT / 'box_binding_timeseries.json').write_text(json.dumps(out_json, indent=2))
print(f'\n[probe-binding] wrote box_binding_timeseries.json')

# ── Build SVG ───────────────────────────────────────────────────────────
W, H = 900, 460
PL, PR, PT, PB = 70, 70, 60, 60
plot_w = W - PL - PR
plot_h = H - PT - PB

def x(tv): return PL + (tv / DURATION_S) * plot_w
def yL(v): return PT + plot_h - v * plot_h     # occupancy 0..1 on left
def yR(v): return PT + plot_h - (v / max(atps.max(), 800)) * plot_h  # ATP count on right

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
    f'DnaA-box occupancy by affinity class — Hill binding probe (dnaa-03)</text>',
    f'<text x="{W/2}" y="40" text-anchor="middle" fill="#64748b" font-size="11">'
    f'Calibrated baseline (TE=20×, fc=0.7); Hill model n=2; Kd high/med/low = 5/50/300 DnaA-ATP</text>',
]
# Y-left axis
for tick in [0, 0.25, 0.5, 0.75, 1.0]:
    yt = yL(tick)
    parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick:.2f}</text>')
parts.append(f'<text x="20" y="{PT+plot_h/2}" text-anchor="middle" '
             f'transform="rotate(-90 20 {PT+plot_h/2})" fill="#334155">fraction of boxes bound</text>')
parts.append(f'<text x="{W-15}" y="{PT+plot_h/2}" text-anchor="middle" '
             f'transform="rotate(90 {W-15} {PT+plot_h/2})" fill="#94a3b8">DnaA-ATP count (gray)</text>')
# X
for tick in [0, 120, 240, 360, 480, 600]:
    if tick > DURATION_S: break
    xt = x(tick)
    parts.append(f'<text x="{xt}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}s</text>')

# ATP background line (right axis)
atp_path = 'M ' + ' L '.join(f'{x(t):.1f},{yR(a):.1f}' for t, a in zip(ts, atps))
parts.append(f'<path d="{atp_path}" fill="none" stroke="#cbd5e1" stroke-width="1.5"/>')
for v in [200, 400, 600, 800]:
    if v <= max(atps.max(), 800):
        parts.append(f'<text x="{PL+plot_w+8}" y="{yR(v)+4}" text-anchor="start" '
                     f'fill="#94a3b8" font-size="11">{v}</text>')

# Occupancy lines
series = [
    ('high (oriC-like, Kd=5)',   high_occ, '#dc2626'),
    ('med  (intermediate, Kd=50)', med_occ, '#f59e0b'),
    ('low  (chromosomal, Kd=300)', low_occ, '#3b82f6'),
]
for label, ys, color in series:
    path = 'M ' + ' L '.join(f'{x(t):.1f},{yL(v):.1f}' for t, v in zip(ts, ys))
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>')

# Pass-band shading: oriC should stay <0.2 while chromosomal <0.9 (textbook)
parts.append(f'<line x1="{PL}" y1="{yL(0.2)}" x2="{PL+plot_w}" y2="{yL(0.2)}" '
             f'stroke="#16a34a" stroke-dasharray="3,3"/>')
parts.append(f'<text x="{PL+5}" y="{yL(0.2)-5}" fill="#16a34a" font-size="11">'
             f'oriC pass band ≤ 0.20 until chromosomal saturates</text>')

# Legend
ly = 50
for label, _, color in series:
    parts.append(f'<rect x="{PL}" y="{ly-9}" width="14" height="14" fill="{color}"/>')
    parts.append(f'<text x="{PL+22}" y="{ly+3}" fill="#334155">{label}</text>')
    ly += 18

# Final results
parts.append(f'<text x="{W-PR-10}" y="{PT+30}" text-anchor="end" fill="#475569" font-size="12">'
             f'final: high={high_occ[-1]:.2f}  med={med_occ[-1]:.2f}  low={low_occ[-1]:.2f}</text>')
parts.append('</svg>')

VIZ = OUT / 'viz'
(VIZ / '13_box_occupancy_two_step.svg').write_text('\n'.join(parts))
print(f'[probe-binding] wrote viz/13_box_occupancy_two_step.svg')
