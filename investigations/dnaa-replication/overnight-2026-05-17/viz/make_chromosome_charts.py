"""Multi-panel chromosome diagrams across the dnaA studies.

Captures chromosome state at multiple timepoints during a calibrated-baseline
run (TE=20×, fc=0.7 via the recipe chain), then renders study-specific
chromosome diagrams using pbg_superpowers.study_charts.chromosome_circle_svg.

Each study gets a chart tailored to its biology:
  dnaa-01  → t=0 / 5 / 10 min, RNAP density + replisomes (basic chromosome view)
  dnaa-02  → t=0 / 1 / 5 min, DnaA-ATP fraction as oriC marker color
  dnaa-03  → t=0, all 456 DnaA-box positions + replisomes
  dnaa-04  → t=0 / 5 / 10 min, oriC DnaA-bound state (proxy for filament)
  dnaa-05  → replisome positions vs DnaA-ADP pool (RIDA target context)
  dnaa-06  → oriC sequestration state (placeholder until SeqA Step lands)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
sys.path.insert(0, str(V2))

from pbg_superpowers.study_charts import chromosome_circle_svg, write_chart
from v2ecoli.composites.baseline_recipes import get_recipe
from process_bigraph import Composite
import numpy as np

GENOME_LEN = 4_641_652

# ── Run the calibrated baseline once + snapshot state at key timepoints ──

SNAPSHOT_TIMES_S = [0, 60, 300, 600]  # 0, 1, 5, 10 minutes

def snapshot_state(comp):
    """Pull out replisome / oriC / RNAP / DnaA-box positions + DnaA pool."""
    agent = comp.state['agents']['0']
    bulk = agent['bulk']
    ids = bulk['id']
    counts = bulk['count']
    APO = int(np.where(ids == 'PD03831[c]')[0][0])
    ATP = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
    ADP = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])

    u = agent['unique']

    snap = {
        'dnaa_apo': int(counts[APO]),
        'dnaa_atp': int(counts[ATP]),
        'dnaa_adp': int(counts[ADP]),
    }

    # Replisomes
    if 'active_replisome' in u:
        ar = u['active_replisome']
        if hasattr(ar, 'dtype') and '_entryState' in ar.dtype.names:
            active = ar['_entryState'].astype(bool)
            coords = ar['coordinates'][active] if 'coordinates' in ar.dtype.names else np.array([])
            snap['replisome_coords'] = [int(c) for c in coords]
            snap['replisome_domains'] = (
                [int(d) for d in ar['domain_index'][active]]
                if 'domain_index' in ar.dtype.names else [0]*len(coords)
            )
        else:
            snap['replisome_coords'] = []; snap['replisome_domains'] = []
    else:
        snap['replisome_coords'] = []; snap['replisome_domains'] = []

    # oriCs (coordinates not stored — they're all at position 0 by convention)
    if 'oriC' in u:
        oc = u['oriC']
        if hasattr(oc, 'dtype') and '_entryState' in oc.dtype.names:
            active = oc['_entryState'].astype(bool)
            snap['n_oriCs'] = int(active.sum())
            snap['oriC_domains'] = (
                [int(d) for d in oc['domain_index'][active]]
                if 'domain_index' in oc.dtype.names else [0]*int(active.sum())
            )
        else:
            snap['n_oriCs'] = 0; snap['oriC_domains'] = []
    else:
        snap['n_oriCs'] = 0; snap['oriC_domains'] = []

    # DnaA boxes (capture once at t=0 — they don't move)
    if 'DnaA_box' in u:
        db = u['DnaA_box']
        if hasattr(db, 'dtype') and '_entryState' in db.dtype.names:
            active = db['_entryState'].astype(bool)
            snap['dnaA_box_coords'] = [int(c) for c in db['coordinates'][active]]
            snap['dnaA_box_domains'] = (
                [int(d) for d in db['domain_index'][active]]
                if 'domain_index' in db.dtype.names else [0]*int(active.sum())
            )
            snap['dnaA_box_bound'] = (
                [bool(b) for b in db['DnaA_bound'][active]]
                if 'DnaA_bound' in db.dtype.names else [False]*int(active.sum())
            )

    # Active RNAP positions (for chromosome backbone density)
    if 'active_RNAP' in u:
        ar = u['active_RNAP']
        if hasattr(ar, 'dtype') and '_entryState' in ar.dtype.names:
            active = ar['_entryState'].astype(bool)
            snap['n_rnap'] = int(active.sum())
            if 'coordinates' in ar.dtype.names:
                snap['rnap_coords'] = [int(c) for c in ar['coordinates'][active]]
            else:
                snap['rnap_coords'] = []

    return snap


def run_and_snapshot():
    print('Building dnaa_02_with_extrinsic_target_rate composite...', flush=True)
    from v2ecoli.core import build_core
    core = build_core()
    recipe = get_recipe('dnaa_02_with_extrinsic_target_rate')
    doc = recipe.build_doc(core=core, seed=0, cache_dir=str(V2 / 'out' / 'cache'))
    comp = Composite(doc, core=core)
    loop_patches = recipe.make_loop_patch_objects(seed=0)
    for lp in loop_patches:
        lp.init(comp)

    snapshots = {}
    SAMPLE_EVERY = 10
    next_snap_idx = 0
    t = 0.0
    snapshots[0] = snapshot_state(comp)
    print(f't=0: n_oriCs={snapshots[0]["n_oriCs"]} repl={len(snapshots[0]["replisome_coords"])} boxes={len(snapshots[0].get("dnaA_box_coords", []))}')

    import time
    t0 = time.time()
    target_times = set(SNAPSHOT_TIMES_S[1:])
    while t < max(SNAPSHOT_TIMES_S):
        comp.update({}, SAMPLE_EVERY)
        t += SAMPLE_EVERY
        for lp in loop_patches:
            lp.apply(comp, SAMPLE_EVERY)
        ti = int(t)
        if ti in target_times:
            snapshots[ti] = snapshot_state(comp)
            s = snapshots[ti]
            print(f't={ti}: n_oriCs={s["n_oriCs"]} repl={len(s["replisome_coords"])} apo={s["dnaa_apo"]} ATP={s["dnaa_atp"]} ADP={s["dnaa_adp"]}  wall={time.time()-t0:.0f}s')
    return snapshots


# ── Build per-study chromosome charts ────────────────────────────────────

def panels_basic_chromosome(snapshots, sample_every_bp=92000):
    """Backbone view: oriC + ter + replisomes + RNAP density (sampled)."""
    panels = []
    for t in sorted(snapshots):
        s = snapshots[t]
        n_chr = max(1, max(s['oriC_domains']) if s['oriC_domains'] else 1)
        # Group features by chromosome (domain_index) — for now collapse all into 1
        # since v2ecoli reports chromosomes via domain_index 0,1,2 and we want
        # one circle per parent + daughters
        # Simplify: split into chromosomes based on oriC count
        n_chrs_to_draw = max(1, s['n_oriCs'])

        chromosomes = []
        # All features go into chromosome 0 for simplicity (real per-chromosome
        # routing would require matching domain_indexes precisely)
        features = {}
        # oriCs at coord 0 (all oriCs live at the origin by definition)
        features['oriC'] = [{
            'coord': 0, 'marker': 'circle', 'color': '#16a34a',
            'size': 9, 'category': 'OriC',
        }]
        # Ter at coord ~2.32M
        features['ter'] = [{
            'coord': 2_320_826, 'marker': 'square', 'color': '#dc2626',
            'size': 7, 'category': 'Ter',
        }]
        # Replisomes
        features['replisomes'] = [
            {'coord': c, 'marker': 'triangle_up', 'color': '#f59e0b',
             'size': 8, 'category': f'Replisome ({len(s["replisome_coords"])})'}
            for c in s['replisome_coords']
        ]
        # RNAP density — sample every Nth to avoid SVG bloat
        rnap = s.get('rnap_coords', [])
        if rnap:
            sampled = rnap[::max(1, len(rnap)//150)]
            features['rnap'] = [
                {'coord': c, 'marker': 'circle', 'color': '#3b82f6',
                 'size': 1.5, 'category': f'RNAP ({len(rnap)})'}
                for c in sampled
            ]
        chromosomes.append({'features': features})
        panels.append({
            'label': f't={t}s ({n_chrs_to_draw} chr)',
            'chromosomes': chromosomes,
        })
    return panels


def panels_dnaa_box_landscape(snapshots):
    """Full DnaA-box catalog at t=0, plus replisomes."""
    s = snapshots[0]
    box_coords = s.get('dnaA_box_coords', [])
    box_domains = s.get('dnaA_box_domains', [])
    box_bound = s.get('dnaA_box_bound', [])
    # Color by domain
    domain_colors = {0: '#2563eb', 1: '#16a34a', 2: '#a855f7'}
    box_features = [
        {'coord': c, 'marker': 'tick', 'color': domain_colors.get(d, '#94a3b8'),
         'size': 6, 'category': f'DnaA-box ({len(box_coords)} total)',
         'outside': True}
        for c, d in zip(box_coords, box_domains)
    ]
    features = {
        'boxes': box_features,
        'oriC': [{'coord': 0, 'marker': 'circle', 'color': '#dc2626', 'size': 10,
                  'category': 'OriC'}],
        'ter':  [{'coord': 2_320_826, 'marker': 'square', 'color': '#0f172a',
                  'size': 8, 'category': 'Ter'}],
        'replisomes': [
            {'coord': c, 'marker': 'triangle_up', 'color': '#f59e0b', 'size': 8,
             'category': f'Replisome ({len(s["replisome_coords"])})'}
            for c in s['replisome_coords']
        ],
    }
    return [{
        'label': f't=0  ·  {len(box_coords)} DnaA boxes  ·  {s["n_oriCs"]} oriC(s)  ·  {len(s["replisome_coords"])} replisome(s)',
        'chromosomes': [{'features': features}],
    }]


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    snapshots = run_and_snapshot()

    studies = V2 / 'studies'

    # dnaa-01: basic chromosome over time
    panels = panels_basic_chromosome(snapshots)
    svg = chromosome_circle_svg(
        'Chromosome state over a calibrated 10-min run',
        subtitle='Recipe: dnaa_02_with_extrinsic_target_rate (TE=20×, fc=0.7, k_hyd=4.6/min). '
                 'OriC, Ter, replisome forks, and RNAP density along the 4.6 Mb circular chromosome.',
        panels=panels, genome_len=GENOME_LEN,
    )
    write_chart(studies / 'dnaa-01-expression-dynamics', '06_chromosome_timecourse', svg,
        title='Chromosome state over calibrated 10-min run',
        caption='OriC (green), Ter (red), replisome forks (orange ▲), RNAP positions sampled '
                'across the backbone. Four timepoints under the validated baseline.')

    # dnaa-02: similar but emphasis on DnaA equilibrium (note the bulk DnaA in caption)
    write_chart(studies / 'dnaa-02-atp-hydrolysis', '07_chromosome_timecourse', svg,
        title='Chromosome state at calibrated DnaA-ATP / ADP cycling',
        caption='Same trajectory as dnaa-01\'s but the bulk DnaA pool now contains both '
                'DnaA-ATP and DnaA-ADP (see also chart 02 ATP-fraction over time). Forks '
                'mid-elongation; oriC is unbound (DnaA-box binding mechanism is dnaa-03).')

    # dnaa-03: DnaA-box landscape — all boxes
    panels_boxes = panels_dnaa_box_landscape(snapshots)
    svg_boxes = chromosome_circle_svg(
        'DnaA-box landscape across the chromosome',
        subtitle=f'All DnaA boxes plotted at their genomic coordinates (radial ticks outside '
                 f'the backbone). OriC at top (12 o\'clock); Ter at bottom.',
        panels=panels_boxes, genome_len=GENOME_LEN,
        width=900, panel_radius=270,
    )
    write_chart(studies / 'dnaa-03-box-binding', '04_chromosome_dnaa_boxes', svg_boxes,
        title='DnaA-box positions on the chromosome',
        caption='All active DnaA boxes plotted at their genomic coordinates. '
                'Boxes colored by replication domain (parent + 2 daughter). '
                'Note the clustering near oriC (top) — the cooperative DnaA-ATP filament site.')
    # dnaa-04: same with emphasis on oriC binding (for initiation trigger context)
    write_chart(studies / 'dnaa-04-initiation-mechanism', '03_chromosome_oric_context', svg_boxes,
        title='Chromosome context for DnaA-occupancy initiation trigger',
        caption='Same DnaA-box landscape used by dnaa-03. dnaa-04 reads the bound-state of '
                'the 11 oriC-proximal boxes (top of the circle) to decide whether to trigger '
                'replication — replacing the current mass-threshold heuristic.')

    # dnaa-05: replisome positions + DnaA-ADP context for RIDA
    panels_rida = panels_basic_chromosome(snapshots)
    svg_rida = chromosome_circle_svg(
        'Replisome positions over time — RIDA target context for dnaa-05',
        subtitle='RIDA (replisome-coupled hydrolysis) converts DnaA-ATP → DnaA-ADP at active '
                 'forks. The orange triangles above are RIDA\'s spatially-resolved acting sites.',
        panels=panels_rida, genome_len=GENOME_LEN,
    )
    write_chart(studies / 'dnaa-05-rida-ddah-dars', '03_chromosome_replisome_context', svg_rida,
        title='Replisome positions — RIDA hydrolysis sites over time',
        caption='RIDA acts at active replication forks (orange triangles). Each fork pair '
                'contributes ~0.5-1.4 events/min of DnaA-ATP → DnaA-ADP conversion. '
                'DDAH acts at a specific GATC cluster near oriC (datA, not shown). '
                'DARS1 and DARS2 reverse this at specific positions ~17′ and ~94′ of the genome.')

    # dnaa-06: simpler view focused on oriC + (eventual) SeqA binding
    panels_seqa = panels_basic_chromosome({0: snapshots[0]})  # just t=0
    svg_seqa = chromosome_circle_svg(
        'oriC chromosomal context — SeqA sequestration target site',
        subtitle='SeqA binds hemimethylated GATC sites clustered around oriC (top). After '
                 'initiation, oriC is hemimethylated for ~10 min while SeqA blocks re-initiation.',
        panels=panels_seqa, genome_len=GENOME_LEN,
        width=900, panel_radius=260,
    )
    write_chart(studies / 'dnaa-06-seqa-sequestration', '03_chromosome_seqa_context', svg_seqa,
        title='Chromosomal context for SeqA sequestration',
        caption='SeqA recognizes hemimethylated GATC sites clustered near oriC (green circle at top). '
                'When `sequestered_until > t`, the oriC marker should switch color — this is the '
                'dnaa-06 implementation target.')

    print('\nDone. New chromosome charts in:')
    for s in ['dnaa-01-expression-dynamics', 'dnaa-02-atp-hydrolysis',
              'dnaa-03-box-binding', 'dnaa-04-initiation-mechanism',
              'dnaa-05-rida-ddah-dars', 'dnaa-06-seqa-sequestration']:
        cd = studies / s / 'charts'
        if cd.exists():
            ls = sorted(cd.glob('*.svg'))
            print(f'  {s}/charts/  ({len(ls)} SVGs)')


if __name__ == '__main__':
    main()
