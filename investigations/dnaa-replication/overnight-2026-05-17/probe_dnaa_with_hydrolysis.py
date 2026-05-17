"""dnaa-02 experimental probe: does adding intrinsic hydrolysis (Boesen 2024
rate 0.046/min) bring DnaA-ATP fraction into the [0.2, 0.5] physiological band?

Method: same as probe_dnaa_states_timeseries.py, but BETWEEN each
comp.update(), we apply the IntrinsicHydrolysis effect EXTERNALLY by
directly mutating bulk[MONOMER0-160] → bulk[MONOMER0-4565] at the
intrinsic rate.

This is the simulated equivalent of wiring the Step into baseline,
without touching baseline.py. If ATP fraction lands in [0.2, 0.5], we've
validated the dnaa-02 hypothesis and can confidently promote the Step.

ALSO applies the (TE=20×, fc=0.7) calibration found in the overnight
sweep, so the result is the FULL "everything fixed" v2ecoli baseline.

Output:
  dnaa_with_hydrolysis_timeseries.json
  viz/11_dnaa_with_hydrolysis.svg
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

# ── Parameters ───────────────────────────────────────────────────────────
DURATION_S = 10 * 60       # 10 min
SAMPLE_EVERY_S = 10        # sample + apply hydrolysis every 10s of sim time
SEED = 0
K_HYDROLYSIS_PER_MIN = 0.046    # Boesen 2024 in-vitro intrinsic rate
TE_MULTIPLIER = 20.0
AUTOREP_MULTIPLIER = 0.7        # (TE=20×, fc=0.7) winning calibration

OUT_DIR = Path(__file__).resolve().parent
VIZ = OUT_DIR / 'viz'
VIZ.mkdir(exist_ok=True)

print(f'[probe-hydrolysis] config:', flush=True)
print(f'  TE_MULTIPLIER       = {TE_MULTIPLIER}')
print(f'  AUTOREP_MULTIPLIER  = {AUTOREP_MULTIPLIER}')
print(f'  K_HYDROLYSIS_PER_MIN= {K_HYDROLYSIS_PER_MIN}')
print(f'  DURATION_S          = {DURATION_S}  (sample every {SAMPLE_EVERY_S}s)')

# ── Apply calibration to bundle ───────────────────────────────────────────
core = build_core()
bundle = load_cache_bundle(str(V2 / 'out' / 'cache'))

# Heavy TF listener (for autorepression evaluation if we want)
tf_cfg = bundle['configs']['ecoli-tf-binding']
tf_cfg['emit_n_bound_TF_per_TU'] = True

# TE multiplier on dnaA monomer (idx 3861)
pi_cfg = bundle['configs']['ecoli-polypeptide-initiation']
pi_cfg['translation_efficiencies'][3861] *= TE_MULTIPLIER

# Autorep fold-change multiplier on delta_prob[deltaJ==12]
dp = tf_cfg['delta_prob']
deltaV = np.asarray(dp['deltaV'])
mask = (np.asarray(dp['deltaJ']) == 12)
deltaV[mask] = deltaV[mask] * AUTOREP_MULTIPLIER
dp['deltaV'] = deltaV
print(f'[probe-hydrolysis] applied calibration: TE×{TE_MULTIPLIER}, fc×{AUTOREP_MULTIPLIER}')

# ── Build composite ──────────────────────────────────────────────────────
print('[probe-hydrolysis] building composite...', flush=True)
doc = baseline(core=core, seed=SEED, cache_dir=str(V2 / 'out' / 'cache'))
comp = Composite(doc, core=core)

# Resolve DnaA bulk indices
bulk0 = comp.state['agents']['0']['bulk']
ids = bulk0['id']
APO_IDX = int(np.where(ids == 'PD03831[c]')[0][0])
ATP_IDX = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
ADP_IDX = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])
print(f'[probe-hydrolysis] DnaA bulk indices: apo={APO_IDX} ATP={ATP_IDX} ADP={ADP_IDX}')

# ── Run loop with manual hydrolysis between updates ───────────────────────
rng = np.random.RandomState(seed=SEED)
k_per_s = K_HYDROLYSIS_PER_MIN / 60.0
samples = []
hydrolysis_events_total = 0

def sample(t):
    bulk = comp.state['agents']['0']['bulk']
    return {
        't': t,
        'apo': int(bulk['count'][APO_IDX]),
        'atp': int(bulk['count'][ATP_IDX]),
        'adp': int(bulk['count'][ADP_IDX]),
    }

# Initial
s0 = sample(0.0)
samples.append({**s0, 'n_hydrolyzed_step': 0, 'total': s0['apo']+s0['atp']+s0['adp']})
print(f"[probe-hydrolysis] t=  0s  apo={s0['apo']:>4}  ATP={s0['atp']:>4}  ADP={s0['adp']:>4}  total={s0['apo']+s0['atp']+s0['adp']}")

print(f'[probe-hydrolysis] simulating {DURATION_S}s in {SAMPLE_EVERY_S}s chunks (with manual hydrolysis)...', flush=True)
t0 = time.time()
sim_t = 0.0
while sim_t < DURATION_S:
    comp.update({}, SAMPLE_EVERY_S)
    sim_t += SAMPLE_EVERY_S

    # === MANUAL INTRINSIC HYDROLYSIS (the proposed Step's logic, applied externally) ===
    bulk = comp.state['agents']['0']['bulk']
    n_atp = int(bulk['count'][ATP_IDX])
    expected_hyd = k_per_s * n_atp * SAMPLE_EVERY_S
    n_hyd = int(stochasticRound(rng, np.asarray([expected_hyd]))[0])
    n_hyd = max(0, min(n_hyd, n_atp))
    if n_hyd > 0:
        bulk['count'][ATP_IDX] -= n_hyd
        bulk['count'][ADP_IDX] += n_hyd
        hydrolysis_events_total += n_hyd
    # ==================================================================================

    s = sample(sim_t)
    total = s['apo'] + s['atp'] + s['adp']
    samples.append({**s, 'n_hydrolyzed_step': n_hyd, 'total': total})
    if int(sim_t) % 60 == 0:
        atp_frac = s['atp'] / total if total > 0 else 0
        print(f"[probe-hydrolysis] t={sim_t:>4.0f}s  apo={s['apo']:>4}  ATP={s['atp']:>4}  ADP={s['adp']:>4}  total={total}  atp_frac={atp_frac:.3f}  this-step hyd={n_hyd}  elapsed_wall={time.time()-t0:.1f}s", flush=True)

print(f'[probe-hydrolysis] DONE in {time.time()-t0:.1f}s wall. Total hydrolysis events: {hydrolysis_events_total}')

# ── Persist + chart ──────────────────────────────────────────────────────
out_json = {
    'description': 'DnaA-state trajectory under (TE=20×, fc=0.7) + EXTERNAL intrinsic hydrolysis (Boesen 0.046/min)',
    'params': {
        'TE_multiplier': TE_MULTIPLIER,
        'autorep_multiplier': AUTOREP_MULTIPLIER,
        'k_hydrolysis_per_min': K_HYDROLYSIS_PER_MIN,
        'duration_s': DURATION_S,
        'seed': SEED,
    },
    'samples': samples,
    'total_hydrolysis_events': hydrolysis_events_total,
}
(OUT_DIR / 'dnaa_with_hydrolysis_timeseries.json').write_text(json.dumps(out_json, indent=2))
print(f'[probe-hydrolysis] wrote dnaa_with_hydrolysis_timeseries.json')

# Second-half ATP fraction (the test target)
n = len(samples)
sh = samples[n//2:]
atp_fracs = [s['atp']/s['total'] if s['total']>0 else 0 for s in sh]
median_atp_frac = sorted(atp_fracs)[len(atp_fracs)//2]
median_total = sorted(s['total'] for s in sh)[len(sh)//2]
print()
print(f"=== Result (second-half median over t∈[{sh[0]['t']:.0f}s, {sh[-1]['t']:.0f}s]) ===")
print(f"  median total DnaA: {median_total}")
print(f"  median ATP fraction: {median_atp_frac:.3f}")
print(f"  Boesen 2024 target [0.20, 0.50]: {'PASS' if 0.20 <= median_atp_frac <= 0.50 else 'FAIL'}")
print(f"  Schmidt 2016 target [300, 800]:  {'PASS' if 300 <= median_total <= 800 else 'FAIL'}")

# Build SVG
W, H = 900, 480
PL, PR, PT, PB = 70, 30, 50, 60
plot_w = W - PL - PR
plot_h = H - PT - PB
ts = [s['t'] for s in samples]
totals = [s['total'] for s in samples]
y_max = max(totals) * 1.15
def x(tv): return PL + (tv / DURATION_S) * plot_w
def y(v): return PT + plot_h - (v / y_max) * plot_h

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,sans-serif" font-size="13">'
         f'<rect width="{W}" height="{H}" fill="white"/>'
         f'<text x="{W/2}" y="22" text-anchor="middle" font-weight="600" font-size="15">'
         f'DnaA states with (TE=20×, fc=0.7) + intrinsic hydrolysis (Boesen 0.046/min)</text>']
# Y ticks
for tick in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
    if tick > y_max: break
    yt = y(tick)
    parts.append(f'<line x1="{PL}" y1="{yt}" x2="{PL+plot_w}" y2="{yt}" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{PL-8}" y="{yt+4}" text-anchor="end" fill="#64748b">{tick}</text>')
# X ticks
for tick in [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]:
    if tick > DURATION_S: break
    xt = x(tick)
    parts.append(f'<text x="{xt}" y="{PT+plot_h+18}" text-anchor="middle" fill="#64748b">{tick}s</text>')
# Series
series = [
    ('apo (PD03831)', [(s['t'], s['apo']) for s in samples], '#3b82f6'),
    ('DnaA-ATP (MONOMER0-160)', [(s['t'], s['atp']) for s in samples], '#10b981'),
    ('DnaA-ADP (MONOMER0-4565)', [(s['t'], s['adp']) for s in samples], '#f59e0b'),
    ('total', [(s['t'], s['total']) for s in samples], '#1e293b'),
]
for label, pts, color in series:
    path = 'M ' + ' L '.join(f'{x(t):.1f},{y(v):.1f}' for (t,v) in pts)
    sw = '2.5' if label == 'total' else '2'
    da = ' stroke-dasharray="4,3"' if label == 'total' else ''
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{da}/>')
# Legend
ly = 50
for label, _, color in series:
    parts.append(f'<rect x="{PL}" y="{ly-9}" width="14" height="14" fill="{color}"/>')
    parts.append(f'<text x="{PL+22}" y="{ly+3}" fill="#334155">{label}</text>')
    ly += 18

# Result annotation
result_color = '#16a34a' if 0.20 <= median_atp_frac <= 0.50 else '#dc2626'
parts.append(f'<text x="{W-PR-10}" y="{PT+30}" text-anchor="end" fill="{result_color}" font-size="12" font-weight="600">'
             f'median ATP fraction = {median_atp_frac:.3f}</text>')
parts.append(f'<text x="{W-PR-10}" y="{PT+48}" text-anchor="end" fill="{result_color}" font-size="11">'
             f'Boesen [0.20, 0.50]: {"PASS" if 0.20 <= median_atp_frac <= 0.50 else "FAIL"}</text>')
parts.append(f'<text x="{W-PR-10}" y="{PT+64}" text-anchor="end" fill="#475569" font-size="11">'
             f'median total DnaA = {median_total}</text>')

parts.append('</svg>')
(VIZ / '11_dnaa_with_hydrolysis.svg').write_text('\n'.join(parts))
print(f'[probe-hydrolysis] wrote viz/11_dnaa_with_hydrolysis.svg')
