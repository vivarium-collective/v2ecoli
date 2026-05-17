"""End-to-end verification of the two new live Visualization Steps.

Runs a calibrated sim (recipe dnaa_02_with_extrinsic_target_rate) for 5 min.
Drives DnaAStateVisualization + DnaABoxOccupancyVisualization manually
through their accumulate/render lifecycle so we can verify they produce
valid HTML reports from live emitted state, without requiring the
composite-level visualization wiring to be in place.

Output:
  investigations/dnaa-replication/overnight-2026-05-17/viz_step_dnaa_state.html
  investigations/dnaa-replication/overnight-2026-05-17/viz_step_dnaa_box_occupancy.html

Also copies to:
  studies/dnaa-02-atp-hydrolysis/viz/manual/  ← simulating where a real
                                                  wired-in viz would write
"""
from __future__ import annotations
import sys, time
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from v2ecoli.composites.baseline_recipes import get_recipe
from v2ecoli.visualizations.dnaa_state import DnaAStateVisualization
from v2ecoli.visualizations.dnaa_box_occupancy import DnaABoxOccupancyVisualization
from v2ecoli.core import build_core
from process_bigraph import Composite

OUT = Path(__file__).resolve().parent

DURATION_S = 5 * 60   # 5-min run
SAMPLE_EVERY_S = 10

print('[verify-viz] building composite via recipe dnaa_02_with_extrinsic_target_rate', flush=True)
core = build_core()
recipe = get_recipe('dnaa_02_with_extrinsic_target_rate')
doc = recipe.build_doc(core=core, seed=0, cache_dir=str(V2 / 'out' / 'cache'))
comp = Composite(doc, core=core)

# Instantiate the viz Steps standalone
state_viz = DnaAStateVisualization({'title': 'DnaA state — recipe verify'}, core=core)
box_viz = DnaABoxOccupancyVisualization({'title': 'DnaA-box occupancy — recipe verify'}, core=core)

loop_patches = recipe.make_loop_patch_objects(seed=0)
for lp in loop_patches:
    lp.init(comp)

def get_state_view():
    """Build the input-port view that each viz Step needs."""
    agent = comp.state['agents']['0']
    return {
        'bulk':       agent['bulk'],
        'listeners':  agent['listeners'],
        'global_time': float(comp.state.get('global_time', 0.0)),
        'DnaA_boxes': agent['unique'].get('DnaA_box'),
    }

# Initial sample
sv = get_state_view()
state_viz.accumulate(sv); box_viz.accumulate(sv)
print(f'[verify-viz] t=0 accumulated', flush=True)

t0 = time.time()
sim_t = 0.0
while sim_t < DURATION_S:
    comp.update({}, SAMPLE_EVERY_S)
    sim_t += SAMPLE_EVERY_S
    for lp in loop_patches:
        lp.apply(comp, SAMPLE_EVERY_S)
    sv = get_state_view()
    state_viz.accumulate(sv); box_viz.accumulate(sv)
    if int(sim_t) % 60 == 0:
        print(f'[verify-viz] t={sim_t:.0f}s accumulated  wall={time.time()-t0:.0f}s', flush=True)

print(f'[verify-viz] DONE simulating in {time.time()-t0:.1f}s. Rendering...')

# Render both
html1 = state_viz.render()
html2 = box_viz.render()

p1 = OUT / 'viz_step_dnaa_state.html'
p2 = OUT / 'viz_step_dnaa_box_occupancy.html'
p1.write_text(html1)
p2.write_text(html2)
print(f'[verify-viz] wrote {p1.name} ({len(html1)} chars)')
print(f'[verify-viz] wrote {p2.name} ({len(html2)} chars)')

# Also copy into a per-study viz/ dir under dnaa-02 so the dashboard's
# investigation viz discovery can show them (the existing /api/viz route
# globs studies/<name>/viz/<run_id>/*.html — we use 'manual' as the run_id).
for study in ['dnaa-02-atp-hydrolysis', 'dnaa-03-box-binding']:
    d = V2 / 'studies' / study / 'viz' / 'manual'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'dnaa_state.html').write_text(html1)
    (d / 'dnaa_box_occupancy.html').write_text(html2)
    print(f'[verify-viz] also wrote to {study}/viz/manual/')
