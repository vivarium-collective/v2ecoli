"""Sensitivity probe: does hydrolysis EVER bring ATP fraction into [0.2, 0.5]
at higher rates? Tests 10×, 100×, 1000× the Boesen intrinsic rate to see if
the mechanism is fundamentally capable.

Hypothesis: equilibrium reverse rate is fast enough that intrinsic
hydrolysis (0.046/min) is overwhelmed. With 100-1000× faster hydrolysis
(approximating what extrinsic RIDA / DDAH / DARS together might
deliver), we should see ATP fraction drop into the physiological range.

Output: per-rate (median total, median ATP fraction).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.composites.baseline import baseline
from process_bigraph import Composite
import numpy as np
from wholecell.utils.random import stochasticRound

DURATION_S = 300         # shorter — 5 min
SAMPLE_EVERY_S = 10
SEED = 0
TE_MULTIPLIER = 20.0
AUTOREP_MULTIPLIER = 0.7

# Rates to test (multiplier × 0.046/min)
RATE_MULTIPLIERS = [1, 10, 100, 1000]

results = []
for rm in RATE_MULTIPLIERS:
    k_per_min = 0.046 * rm
    print(f'\n=== probe rate-multiplier × {rm}× (k = {k_per_min:.3f}/min) ===', flush=True)

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

    doc = baseline(core=core, seed=SEED, cache_dir=str(V2 / 'out' / 'cache'))
    comp = Composite(doc, core=core)
    bulk0 = comp.state['agents']['0']['bulk']
    ids = bulk0['id']
    APO = int(np.where(ids == 'PD03831[c]')[0][0])
    ATP = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
    ADP = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])

    rng = np.random.RandomState(SEED + rm)
    k_per_s = k_per_min / 60.0
    samples = []
    t = 0.0
    sim_start = time.time()
    while t < DURATION_S:
        comp.update({}, SAMPLE_EVERY_S)
        t += SAMPLE_EVERY_S
        bulk = comp.state['agents']['0']['bulk']
        n_atp = int(bulk['count'][ATP])
        expected = k_per_s * n_atp * SAMPLE_EVERY_S
        n_hyd = int(stochasticRound(rng, np.asarray([expected]))[0])
        n_hyd = max(0, min(n_hyd, n_atp))
        if n_hyd > 0:
            bulk['count'][ATP] -= n_hyd
            bulk['count'][ADP] += n_hyd
        samples.append({
            't': t,
            'apo': int(bulk['count'][APO]),
            'atp': int(bulk['count'][ATP]),
            'adp': int(bulk['count'][ADP]),
            'n_hyd': n_hyd,
        })
    sh = samples[len(samples)//2:]
    totals = [s['apo']+s['atp']+s['adp'] for s in sh]
    fracs = [s['atp']/(s['apo']+s['atp']+s['adp']) if (s['apo']+s['atp']+s['adp']) > 0 else 0 for s in sh]
    med_total = sorted(totals)[len(totals)//2]
    med_frac = sorted(fracs)[len(fracs)//2]
    boesen_pass = 0.20 <= med_frac <= 0.50
    print(f'  median total: {med_total}, median ATP frac: {med_frac:.3f}, Boesen [.20,.50]: {"PASS" if boesen_pass else "FAIL"}, wall: {time.time()-sim_start:.0f}s')
    results.append({'rate_multiplier': rm, 'k_per_min': k_per_min,
                    'median_total': med_total, 'median_atp_frac': med_frac,
                    'boesen_pass': boesen_pass})

print('\n=== Sensitivity summary ===')
print(f"{'×rate':>6} {'k/min':>7} {'tot':>5} {'ATP frac':>9} {'PASS?':>6}")
print('-' * 40)
for r in results:
    print(f"{r['rate_multiplier']:>6} {r['k_per_min']:>7.3f} {r['median_total']:>5} {r['median_atp_frac']:>9.3f} {'PASS' if r['boesen_pass'] else 'FAIL':>6}")

import json
out = Path('/Users/eranagmon/code/v2ecoli/investigations/dnaa-replication/overnight-2026-05-17/hydrolysis_rate_sensitivity.json')
out.write_text(json.dumps(results, indent=2))
print(f'\nWrote {out}')
