"""Generic probe that runs ANY baseline-recipe + records DnaA-state trajectory.

Demonstrates the recipe-chain concept end-to-end. Reads bulk[apo/ATP/ADP]
between updates; applies all accumulated loop-patches.

Usage:
    python probe_via_recipe.py <recipe_name> [duration_min] [seed]

Example:
    python probe_via_recipe.py dnaa_02_with_intrinsic_hydrolysis 10 0
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from v2ecoli.composites.baseline_recipes import get_recipe
from process_bigraph import Composite
import numpy as np


def main(recipe_name: str, duration_min: float, seed: int):
    SAMPLE_EVERY_S = 10
    duration_s = duration_min * 60

    print(f'[recipe-probe] recipe={recipe_name}, duration_min={duration_min}, seed={seed}', flush=True)
    recipe = get_recipe(recipe_name)
    print(f'[recipe-probe] lineage: {" → ".join(recipe.lineage())}')
    print(f'[recipe-probe] bundle_patches: {recipe.all_bundle_patches()}')
    print(f'[recipe-probe] loop_patches:   {recipe.all_loop_patches()}')

    print('[recipe-probe] building composite...', flush=True)
    from v2ecoli.core import build_core
    core = build_core()
    doc = recipe.build_doc(core=core, seed=seed, cache_dir=str(V2 / 'out' / 'cache'))
    comp = Composite(doc, core=core)

    bulk0 = comp.state['agents']['0']['bulk']
    ids = bulk0['id']
    APO = int(np.where(ids == 'PD03831[c]')[0][0])
    ATP = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
    ADP = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])

    loop_patches = recipe.make_loop_patch_objects(seed=seed)
    for lp in loop_patches:
        lp.init(comp)
    print(f'[recipe-probe] initialized {len(loop_patches)} loop patches')

    samples = []
    def sample(t):
        b = comp.state['agents']['0']['bulk']
        return {'t': t, 'apo': int(b['count'][APO]),
                'atp': int(b['count'][ATP]),
                'adp': int(b['count'][ADP])}

    s0 = sample(0.0)
    samples.append({**s0, 'total': s0['apo']+s0['atp']+s0['adp'], 'loop_info': {}})
    print(f"[recipe-probe] t=0   apo={s0['apo']} ATP={s0['atp']} ADP={s0['adp']} total={samples[-1]['total']}")

    t0 = time.time()
    sim_t = 0.0
    while sim_t < duration_s:
        comp.update({}, SAMPLE_EVERY_S)
        sim_t += SAMPLE_EVERY_S
        loop_info = {}
        for lp in loop_patches:
            loop_info[lp.kind] = lp.apply(comp, SAMPLE_EVERY_S)
        s = sample(sim_t)
        total = s['apo']+s['atp']+s['adp']
        samples.append({**s, 'total': total, 'loop_info': loop_info})
        if int(sim_t) % 60 == 0:
            af = s['atp']/total if total>0 else 0
            extras = ' '.join(f'{k}={v.get("n_hydrolyzed_step", "?")}' for k,v in loop_info.items())
            print(f"[recipe-probe] t={sim_t:>4.0f}s apo={s['apo']:>4} ATP={s['atp']:>4} ADP={s['adp']:>4} total={total:>4} atp_frac={af:.3f} {extras} wall={time.time()-t0:.0f}s", flush=True)

    print(f'[recipe-probe] DONE in {time.time()-t0:.1f}s wall')

    # Stats: second-half medians
    sh = samples[len(samples)//2:]
    totals = [s['total'] for s in sh]
    fracs = [s['atp']/s['total'] if s['total']>0 else 0 for s in sh]
    med_total = sorted(totals)[len(totals)//2]
    med_atp_frac = sorted(fracs)[len(fracs)//2]
    print()
    print(f'=== Result (second-half median, t∈[{sh[0]["t"]:.0f}s, {sh[-1]["t"]:.0f}s]) ===')
    print(f'  median total DnaA: {med_total}')
    print(f'  median ATP fraction: {med_atp_frac:.3f}')
    print(f'  Boesen 2024 target [0.20, 0.50]: {"PASS" if 0.20 <= med_atp_frac <= 0.50 else "FAIL"}')
    print(f'  Schmidt 2016 target [300, 800]:  {"PASS" if 300 <= med_total <= 800 else "FAIL"}')

    out_json = {
        'recipe': recipe_name,
        'lineage': recipe.lineage(),
        'bundle_patches': recipe.all_bundle_patches(),
        'loop_patches': recipe.all_loop_patches(),
        'samples': samples,
        'result': {
            'median_total': med_total,
            'median_atp_frac': med_atp_frac,
            'count_pass': 300 <= med_total <= 800,
            'atp_frac_pass': 0.20 <= med_atp_frac <= 0.50,
        },
    }
    out_path = Path(__file__).resolve().parent / f'recipe_probe_{recipe_name}.json'
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f'\n[recipe-probe] wrote {out_path.name}')


if __name__ == '__main__':
    recipe = sys.argv[1] if len(sys.argv) > 1 else 'dnaa_02_with_intrinsic_hydrolysis'
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(recipe, dur, seed)
