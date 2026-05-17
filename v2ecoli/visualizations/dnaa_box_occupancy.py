"""DnaABoxOccupancyVisualization — live visualization of DnaA-box binding
state across the chromosome, rendered ONCE at end of sim from emitted history.

Companion to DnaAStateVisualization. Focuses on the *binding-state*
dimension that dnaa-03's eventual cooperative-binding Step will populate.

Today (no binding mechanism wired): DnaA_bound stays at 0 throughout, so
this Step renders a placeholder annotation noting that all 456 boxes
remain unbound. Once dnaa-03's binding Step lands and starts toggling
DnaA_bound, this same Step will render the dynamic occupancy.

Renders:
  1. Bound-fraction trajectory: oriC-proximal vs chromosomal sites
  2. End-state chromosome circle with bound boxes highlighted
  3. Per-class occupancy table

Inputs (typed ports): listeners + unique.DnaA_box read via topology.
"""
from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from pbg_superpowers.study_charts import (
    line_chart_svg, chromosome_circle_svg,
)


GENOME_LEN = 4_641_652
# Boxes within this many bp of oriC are tagged as "oriC-proximal" (high-affinity
# in the cooperative-binding model). 5kb covers the dnaA-promoter cluster + the
# 11 canonical oriC sites.
ORIC_PROXIMITY_BP = 5_000


class DnaABoxOccupancyVisualization(Visualization):
    """Render DnaA-box binding-state evolution + end-state chromosome view.

    Inputs:
      - DnaA_boxes:  unique array of {coordinates, domain_index, DnaA_bound}
      - global_time: float per-tick sim time

    Today: DnaA_bound is never toggled (dnaa-03 mechanism not yet wired);
    this Step emits a placeholder with "all-unbound" annotation, so its
    output is correct-by-construction once the mechanism lands.
    """

    config_schema = {
        **Visualization.config_schema,
        'sample_every': {'_type': 'integer', '_default': 10},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            'DnaA_boxes': {
                '_type': 'unique_array[coordinates:integer|domain_index:integer|DnaA_bound:boolean|_entryState:integer]',
                '_default': [],
            },
            'global_time': {'_type': 'float', '_default': 0.0},
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._times: list[float] = []
        self._n_total: list[int] = []
        self._n_bound: list[int] = []
        self._n_bound_oric: list[int] = []
        self._n_bound_chrom: list[int] = []
        # End-state: hold the last full snapshot for the chromosome circle
        self._last_box_coords: list[int] = []
        self._last_box_domains: list[int] = []
        self._last_box_bound: list[bool] = []

    def accumulate(self, state: dict) -> None:
        boxes = state.get('DnaA_boxes')
        if boxes is None or not hasattr(boxes, 'dtype'):
            return
        if '_entryState' not in boxes.dtype.names:
            return
        import numpy as np
        active = boxes['_entryState'].astype(bool)
        if active.sum() == 0:
            return
        coords = boxes['coordinates'][active]
        domains = boxes['domain_index'][active] if 'domain_index' in boxes.dtype.names else np.zeros(active.sum())
        bound = boxes['DnaA_bound'][active].astype(bool) if 'DnaA_bound' in boxes.dtype.names else np.zeros(active.sum(), dtype=bool)

        t = float(state.get('global_time', 0.0))
        n_total = int(active.sum())
        n_bound = int(bound.sum())
        # Classify by proximity to oriC (coord=0)
        oric_proximal_mask = np.abs(coords) <= ORIC_PROXIMITY_BP
        n_bound_oric = int((bound & oric_proximal_mask).sum())
        n_bound_chrom = n_bound - n_bound_oric

        self._times.append(t)
        self._n_total.append(n_total)
        self._n_bound.append(n_bound)
        self._n_bound_oric.append(n_bound_oric)
        self._n_bound_chrom.append(n_bound_chrom)
        # Update last-snapshot
        self._last_box_coords = [int(c) for c in coords]
        self._last_box_domains = [int(d) for d in domains]
        self._last_box_bound = [bool(b) for b in bound]

    def render(self) -> str:
        if not self._times:
            return self._empty_html()

        n_total_final = self._n_total[-1] if self._n_total else 0
        n_bound_final = self._n_bound[-1] if self._n_bound else 0
        n_oric_proximal = sum(1 for c in self._last_box_coords if abs(c) <= ORIC_PROXIMITY_BP)

        # If NO box ever got bound, render a placeholder explaining why.
        ever_bound = max(self._n_bound) if self._n_bound else 0
        placeholder_note = ''
        if ever_bound == 0:
            placeholder_note = (
                f'<div style="padding:14px;background:#fef9c3;border-left:4px solid #facc15;'
                f'margin:16px 0;border-radius:4px;font-family:-apple-system,sans-serif">'
                f'<strong>ℹ Placeholder render.</strong> No box ever transitioned to bound during '
                f'this {self._times[-1]:.0f}s sim. v2ecoli\'s DnaA_bound state is never written by '
                f'any process today — dnaa-03 is responsible for implementing the cooperative '
                f'binding mechanism. Once that Step lands and starts toggling DnaA_bound, this '
                f'same visualization will render the dynamic occupancy.</div>'
            )

        # Chart 1: bound-fraction trajectory (oriC-proximal vs chromosomal)
        chart1 = line_chart_svg(
            'DnaA-box bound count over time — by class',
            subtitle=f'oriC-proximal: boxes within ±{ORIC_PROXIMITY_BP} bp of origin '
                     f'({n_oric_proximal} of {n_total_final} total).',
            x_label='simulated seconds',
            y_label='# boxes bound',
            series=[
                {'label': 'oriC-proximal bound', 'xs': self._times, 'ys': self._n_bound_oric, 'color': '#dc2626'},
                {'label': 'chromosomal bound',   'xs': self._times, 'ys': self._n_bound_chrom, 'color': '#3b82f6'},
                {'label': 'total bound',         'xs': self._times, 'ys': self._n_bound, 'color': '#0f172a', 'dashed': True},
            ],
        )

        # Chart 2: end-state chromosome with bound vs unbound visible
        bound_features = []
        unbound_features = []
        domain_palette = {0: '#2563eb', 1: '#16a34a', 2: '#a855f7'}
        for c, d, b in zip(self._last_box_coords, self._last_box_domains, self._last_box_bound):
            if b:
                bound_features.append({
                    'coord': c, 'marker': 'circle', 'color': '#dc2626',
                    'size': 5, 'category': f'bound ({n_bound_final})',
                })
            else:
                unbound_features.append({
                    'coord': c, 'marker': 'tick',
                    'color': domain_palette.get(d, '#94a3b8'),
                    'size': 5, 'category': f'unbound ({n_total_final - n_bound_final})',
                    'outside': True,
                })

        chrom_features = {
            'unbound_boxes': unbound_features,
            'bound_boxes':   bound_features,
            'oriC': [{'coord': 0, 'marker': 'circle', 'color': '#16a34a',
                      'size': 11, 'category': 'OriC'}],
            'ter':  [{'coord': 2_320_826, 'marker': 'square', 'color': '#0f172a',
                      'size': 8, 'category': 'Ter'}],
        }
        chart2 = chromosome_circle_svg(
            'End-state DnaA-box landscape',
            subtitle=f'Bound boxes shown as red dots ON the backbone; '
                     f'unbound shown as colored ticks OUTSIDE (color by replication domain).',
            panels=[{
                'label': f't={self._times[-1]:.0f}s  ·  {n_bound_final}/{n_total_final} bound',
                'chromosomes': [{'features': chrom_features}],
            }],
            genome_len=GENOME_LEN, width=900, panel_radius=270,
        )

        # Summary table
        bound_frac = (n_bound_final / n_total_final) if n_total_final > 0 else 0
        n_oric_bound = self._n_bound_oric[-1] if self._n_bound_oric else 0
        oric_frac = (n_oric_bound / n_oric_proximal) if n_oric_proximal > 0 else 0
        summary = (
            f'<div style="display:flex;gap:24px;margin:16px 0;padding:14px;'
            f'background:#f8fafc;border-radius:6px;font-family:-apple-system,sans-serif">'
            f'<div><strong>Total bound fraction</strong>: '
            f'<span style="font-size:1.5em">{bound_frac:.3f}</span></div>'
            f'<div><strong>oriC-proximal bound</strong>: '
            f'<span style="font-size:1.5em">{n_oric_bound}/{n_oric_proximal}</span></div>'
            f'<div><strong>Samples</strong>: '
            f'<span style="font-size:1.5em">{len(self._times)}</span></div>'
            f'</div>'
        )

        title = self.config.get('title') or 'DnaA-box occupancy visualization'
        return (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f'<body style="font-family:-apple-system,sans-serif;margin:24px;color:#0f172a">'
            f'<h1 style="margin:0 0 4px 0">{title}</h1>'
            f'<p style="color:#64748b;margin:0 0 16px 0">'
            f'Live visualization rendered at sim end from emitted DnaA_box state '
            f'(sampled every {self.config.get("sample_every", 10)} ticks).</p>'
            f'{placeholder_note}'
            f'{summary}'
            f'<h3>1. Bound-count trajectory</h3>'
            f'{chart1}'
            f'<h3>2. End-state chromosome landscape</h3>'
            f'{chart2}'
            f'</body></html>'
        )

    def _empty_html(self) -> str:
        return (
            '<!DOCTYPE html><html><body><p>No DnaA-box data accumulated. Check that the '
            'DnaA_boxes port is wired and sample_every is reasonable.</p></body></html>'
        )
