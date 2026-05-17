"""Probe: is monomer_counts[3861] (PD03831 / apo-DnaA) only PART of total DnaA?

If MONOMER0-160 (DnaA-ATP) and MONOMER0-4565 (DnaA-ADP) hold real counts at
runtime, then F-01's "115 DnaA/cell" finding may be an undercount — only the
apo pool. Total DnaA = apo + ATP-bound + ADP-bound.

This script runs a short baseline simulation and captures the bulk vector
at the end to count all three DnaA forms directly.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, '.')
from v2ecoli.core import build_core
from v2ecoli.composites.baseline import baseline
from process_bigraph import Composite
import numpy as np

DURATION_S = 5 * 60  # 5 minutes — fast enough, long enough to equilibrate

core = build_core()
print('[probe] building composite...')
doc = baseline(core=core, seed=0, cache_dir='out/cache')

# Find DnaA-related bulk indices BEFORE sim
agent0 = doc['state']['agents']['0']
bulk0 = agent0['bulk']
ids = bulk0['id']
DNAA_APO = int(np.where(ids == 'PD03831[c]')[0][0])      # 11565
DNAA_ATP = int(np.where(ids == 'MONOMER0-160[c]')[0][0]) # 10822
DNAA_ADP = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])# 11114
print(f'[probe] bulk indices: apo={DNAA_APO} ATP={DNAA_ATP} ADP={DNAA_ADP}')
print(f'[probe] initial: apo={bulk0["count"][DNAA_APO]}, '
      f'ATP={bulk0["count"][DNAA_ATP]}, ADP={bulk0["count"][DNAA_ADP]}')

comp = Composite(doc, core=core)
print(f'[probe] simulating {DURATION_S}s ...')
t = time.time()
comp.update({}, DURATION_S)
print(f'[probe] done in {time.time()-t:.1f}s')

# Read the live state out of the Composite after sim
post_state = comp.state
agent = post_state['agents']['0']
bulk = agent['bulk']
apo = int(bulk['count'][DNAA_APO])
atp = int(bulk['count'][DNAA_ATP])
adp = int(bulk['count'][DNAA_ADP])
total = apo + atp + adp
print()
print(f'=== Final DnaA pool (after {DURATION_S}s simulation) ===')
print(f'  apo  (PD03831):     {apo}')
print(f'  ATP  (MONOMER0-160): {atp}')
print(f'  ADP  (MONOMER0-4565): {adp}')
print(f'  TOTAL:              {total}')
print()
if total > 0:
    print(f'  ATP fraction: {atp/total:.3f}')
    print(f'  Apo fraction: {apo/total:.3f}')
    print()
    print(f'>>> KEY: F-01 reported "DnaA = 115" but that was monomer_counts[3861] = apo only.')
    print(f'>>> TRUE total DnaA at this point: {total} (apo {apo} + ATP {atp} + ADP {adp})')
    print(f'>>> Literature range [300, 800]: {"PASSES" if 300 <= total <= 800 else "FAILS"}')
