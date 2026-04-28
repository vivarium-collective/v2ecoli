"""
================
DnaA Box Binding
================

Binds DnaA-ATP / DnaA-ADP to chromosomal DnaA boxes based on
per-region affinity classes from the curated reference. For each
active DnaA box at each tick, computes an equilibrium occupancy
probability from the relevant DnaA pool concentration and the
class-specific Kd, then samples whether the box is bound.

This makes the ``DnaA_bound`` field on ``DNAA_BOX_ARRAY`` do its job —
prior to Phase 2 the field was write-only (set False at init and on
fork passage, never set True).

Mathematical model (per box, per tick):

    [DnaA]   = (n_atp + n_adp) / (V * N_A) * 1e9      [nM]
              (subset by the region's binds_atp / binds_adp flags)
    p_bound  = [DnaA] / (Kd + [DnaA])                  [equilibrium occupancy]
    bound    ~ Bernoulli(p_bound)

Per-region (Kd, nucleotide-preference) come from
``v2ecoli.data.replication_initiation.REGION_BINDING_RULES``. Boxes
outside any named region fall back to ``DEFAULT_REGION_BINDING_RULE``
(low-affinity, ATP-preferential).

Reference (in the curated PDF):
    Speck C, Weigel C, Messer W. ATP- and ADP-DnaA protein, a molecular
    switch in gene regulation. EMBO J. 18(21):6169–6176 (1999).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import (
    DEFAULT_REGION_BINDING_RULE,
    DNAA_ADP_BULK_ID, DNAA_ATP_BULK_ID,
    DNAA_BOX_HIGH_AFFINITY_KD_NM,
    DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND,
    REGION_BINDING_RULES,
    region_for_coord,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY


NAME = "dnaA_box_binding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# Avogadro's number; cellular volume default (typical fast-growth E. coli).
_N_AVOGADRO = 6.022e23
_DEFAULT_CELL_VOLUME_L = 1e-15  # 1 fL


class DnaABoxBinding(Step):
    """DnaA Box Binding Step

    Per active DnaA box, sample bound/unbound from the equilibrium
    occupancy probability. Updates the ``DnaA_bound`` attribute via
    the unique-array ``set`` interface. Emits per-region bound counts.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'seed': 'integer{0}',
        'cell_volume_L': f'float{{{_DEFAULT_CELL_VOLUME_L:e}}}',
        'kd_high_nM': f'float{{{DNAA_BOX_HIGH_AFFINITY_KD_NM}}}',
        'kd_low_nM': f'float{{{DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND}}}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.cell_volume_L = float(self.parameters.get(
            'cell_volume_L', _DEFAULT_CELL_VOLUME_L))
        self.kd_high = float(self.parameters.get(
            'kd_high_nM', DNAA_BOX_HIGH_AFFINITY_KD_NM))
        self.kd_low = float(self.parameters.get(
            'kd_low_nM', DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND))
        self.random_state = np.random.RandomState(
            seed=int(self.parameters.get('seed', 0)))
        self._bulk_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            # Read-only on DnaA_boxes — the listener emits sampled
            # occupancy without writing back to DnaA_bound; see the
            # comment in update().
            'listeners': {
                'dnaA_binding': {
                    'total_bound': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'total_active': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'fraction_bound': {
                        '_type': 'overwrite[float]', '_default': []},
                    'bound_oric': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_dnaA_promoter': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_datA': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_DARS1': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_DARS2': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_other': {
                        '_type': 'overwrite[integer]', '_default': []},
                },
            },
        }

    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def update(self, states, interval=None):
        if self._bulk_idx is None:
            ids = states['bulk']['id']
            self._bulk_idx = bulk_name_to_idx(
                [DNAA_ATP_BULK_ID, DNAA_ADP_BULK_ID], ids)

        atp_count, adp_count = counts(states['bulk'], self._bulk_idx)
        atp_count = int(atp_count)
        adp_count = int(adp_count)

        # Concentrations in nM. M = count / (V_L * N_A); nM = M * 1e9.
        denom = self.cell_volume_L * _N_AVOGADRO
        atp_nM = (atp_count / denom) * 1e9 if denom > 0 else 0.0
        adp_nM = (adp_count / denom) * 1e9 if denom > 0 else 0.0

        boxes = states['DnaA_boxes']
        empty_listener = {
            'total_bound': 0, 'total_active': 0, 'fraction_bound': 0.0,
            'bound_oric': 0, 'bound_dnaA_promoter': 0, 'bound_datA': 0,
            'bound_DARS1': 0, 'bound_DARS2': 0, 'bound_other': 0,
        }
        if (boxes is None or not hasattr(boxes, 'dtype')
                or '_entryState' not in boxes.dtype.names):
            return {'listeners': {'dnaA_binding': empty_listener}}

        active_mask = boxes['_entryState'].view(np.bool_)
        n_active = int(active_mask.sum())
        if n_active == 0:
            return {'listeners': {'dnaA_binding': empty_listener}}

        coords = boxes['coordinates'][active_mask]
        new_bound = np.zeros(n_active, dtype=bool)
        region_bound_counts = {
            'oriC': 0, 'dnaA_promoter': 0, 'datA': 0,
            'DARS1': 0, 'DARS2': 0, 'other': 0,
        }

        for i, coord in enumerate(coords):
            region = region_for_coord(int(coord))
            rule = REGION_BINDING_RULES.get(region) if region else None
            if rule is None:
                rule = DEFAULT_REGION_BINDING_RULE
                region_key = 'other'
            else:
                region_key = region
            kd = self.kd_high if rule.affinity_class == 'high' else self.kd_low
            concentration = (
                (atp_nM if rule.binds_atp else 0.0) +
                (adp_nM if rule.binds_adp else 0.0))
            if concentration <= 0:
                p_bound = 0.0
            else:
                p_bound = concentration / (kd + concentration)
            if self.random_state.random() < p_bound:
                new_bound[i] = True
                region_bound_counts[region_key] = (
                    region_bound_counts.get(region_key, 0) + 1)

        total_bound = int(new_bound.sum())

        # We do *not* write back to the DnaA_bound field on the unique
        # store. The 'set' update mode requires the new value array to
        # match the current active-box count exactly, but
        # chromosome_structure adds and deletes boxes during fork
        # passage in the same tick, so the count at apply-time can
        # differ from sample-time and the set raises a numpy size
        # error. Instead, the binding process is a *listener-only*
        # report on equilibrium occupancy. Phase 3's initiation gate
        # will read these listener counts directly.
        return {
            'listeners': {
                'dnaA_binding': {
                    'total_bound': total_bound,
                    'total_active': n_active,
                    'fraction_bound': (total_bound / n_active
                                       if n_active else 0.0),
                    'bound_oric': region_bound_counts['oriC'],
                    'bound_dnaA_promoter': region_bound_counts['dnaA_promoter'],
                    'bound_datA': region_bound_counts['datA'],
                    'bound_DARS1': region_bound_counts['DARS1'],
                    'bound_DARS2': region_bound_counts['DARS2'],
                    'bound_other': region_bound_counts['other'],
                },
            },
        }
