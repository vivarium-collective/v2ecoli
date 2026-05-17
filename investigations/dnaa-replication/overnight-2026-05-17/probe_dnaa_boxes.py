"""dnaa-03 probe: inventory of DnaA-box infrastructure at runtime.

Discovers:
  - How many DnaA_boxes are initialized?
  - How many are at the oriC (high-affinity sites)?
  - Initial DnaA_bound state — all false? Or some pre-bound?
  - Are any boxes annotated with affinity class (Kd)?
  - Does any process update DnaA_bound during a sim?

Compare to dnaa-03's assumption of 322 sites (307 chromosomal + 11 oriC + 4 dnaAp).

Output: JSON inventory + console summary.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from v2ecoli.core import build_core
from v2ecoli.composites.baseline import baseline
from process_bigraph import Composite
import numpy as np

OUT = Path(__file__).resolve().parent

# ── Build composite ─────────────────────────────────────────────────────
core = build_core()
print('[probe-boxes] building composite ...', flush=True)
doc = baseline(core=core, seed=0, cache_dir=str(V2 / 'out' / 'cache'))
comp = Composite(doc, core=core)

# ── Reach into the initial state for DnaA_boxes ─────────────────────────
agent0 = comp.state['agents']['0']
print('[probe-boxes] agent keys:', list(agent0.keys())[:25])
unique = agent0.get('unique')
print('[probe-boxes] unique keys:', list(unique.keys())[:30] if isinstance(unique, dict) else type(unique))
boxes = None
oriCs = None
if isinstance(unique, dict):
    for k in ('DnaA_box', 'DnaA_boxes'):
        if k in unique:
            boxes = unique[k]; break
    for k in ('oriC', 'oriCs'):
        if k in unique:
            oriCs = unique[k]; break

# ── Inventory ───────────────────────────────────────────────────────────
inventory = {
    'n_boxes_initial': 0,
    'n_oriCs_initial': 0,
    'box_attrs': [],
    'oriC_attrs': [],
    'initial_bound': 0,
    'unique_domain_indices': [],
}

if boxes is None:
    print('[probe-boxes] WARN: no DnaA_box(es) key under unique.')
else:
    print(f'[probe-boxes] DnaA_boxes type: {type(boxes).__name__}')
    if hasattr(boxes, 'dtype') and boxes.dtype.names:
        inventory['n_boxes_initial'] = int(len(boxes))
        inventory['box_attrs'] = list(boxes.dtype.names)
        # _entryState indicates active rows in unique array
        if '_entryState' in boxes.dtype.names:
            active = boxes['_entryState'].astype(bool)
            inventory['n_boxes_active'] = int(active.sum())
            print(f'[probe-boxes] n_boxes total slots: {len(boxes)}  active: {active.sum()}')
        else:
            print(f'[probe-boxes] n_boxes: {len(boxes)}  (no _entryState column)')
        if 'DnaA_bound' in boxes.dtype.names:
            bound = boxes['DnaA_bound'].astype(bool)
            inventory['initial_bound'] = int(bound.sum())
        if 'domain_index' in boxes.dtype.names:
            di = boxes['domain_index']
            inventory['unique_domain_indices'] = sorted(set(int(x) for x in di.tolist()))
        if 'coordinates' in boxes.dtype.names:
            coords = boxes['coordinates']
            inventory['coord_min'] = int(coords.min())
            inventory['coord_max'] = int(coords.max())

if oriCs is None:
    print('[probe-boxes] WARN: no oriC(s) key under unique.')
else:
    if hasattr(oriCs, 'dtype') and oriCs.dtype.names:
        inventory['n_oriCs_initial'] = int(len(oriCs))
        inventory['oriC_attrs'] = list(oriCs.dtype.names)
        if '_entryState' in oriCs.dtype.names:
            inventory['n_oriCs_active'] = int(oriCs['_entryState'].astype(bool).sum())

# ── Run a short sim, then re-check DnaA_bound ─────────────────────────
print('\n[probe-boxes] running 60s sim and re-checking DnaA_bound...', flush=True)
t0 = time.time()
comp.update({}, 60)
print(f'[probe-boxes] sim done in {time.time()-t0:.1f}s')

u = comp.state['agents']['0']['unique']
boxes_after = u.get('DnaA_box') if 'DnaA_box' in u else u.get('DnaA_boxes') if 'DnaA_boxes' in u else None
if boxes_after is not None and hasattr(boxes_after, 'dtype') and 'DnaA_bound' in boxes_after.dtype.names:
    bound_after = boxes_after['DnaA_bound'].astype(bool).sum()
    inventory['bound_after_60s'] = int(bound_after)
    delta = int(bound_after) - inventory['initial_bound']
    print(f'[probe-boxes] DnaA_bound: initial={inventory["initial_bound"]}, after-60s={bound_after}, delta={delta:+d}')

# ── Persist ─────────────────────────────────────────────────────────────
(OUT / 'dnaa_boxes_inventory.json').write_text(json.dumps(inventory, indent=2, default=str))
print(f'\n[probe-boxes] wrote dnaa_boxes_inventory.json')
print('Summary:')
for k, v in inventory.items():
    if isinstance(v, list) and len(v) > 10:
        print(f'  {k}: [{len(v)} entries] e.g. {v[:5]}...')
    else:
        print(f'  {k}: {v}')
