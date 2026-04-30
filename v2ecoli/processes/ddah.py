"""
====
DDAH
====

datA-Dependent DnaA-ATP Hydrolysis — backup pathway to RIDA.

Mechanism per the curated PDF: the chromosomal datA locus (94.7 min,
~363 bp) binds IHF, which induces a DNA architecture (likely loop
formation) that catalyzes hydrolysis of DnaA-ATP bound at the locus.
Unlike RIDA, DDAH is **not coupled to active replication** — it
fires constitutively whenever IHF is bound to datA. That makes it a
"steady-leak" complement to RIDA's "replication-pulse" hydrolysis.

This first-cut implementation models DDAH as a constitutive
first-order hydrolysis of DnaA-ATP at a small rate constant. The
biology being skipped:

  * datA region coordinates are not yet loaded into
    `motif_coordinates` — Phase 0 found 0 strict-consensus boxes in
    the datA window. Enriching the box set is a follow-up.
  * IHF binding at datA is not wired. A future refinement would
    scale the DDAH rate by the IHF heterodimer count and the IBS-1
    occupancy at datA.

Mathematical model:

    rate = k_ddah * dnaA_atp_count    [molecules / s]

    n_to_hydrolyze ~ Poisson(rate * dt), capped at dnaA_atp_count

    DnaA-ATP -= n_to_hydrolyze
    DnaA-ADP += n_to_hydrolyze

Default k_ddah is chosen ~10× smaller than RIDA's per-replisome rate
so DDAH provides a visible-but-secondary contribution to DnaA-ATP
hydrolysis.

References (in the curated PDF):
    Katayama et al. 2017 (Front. Microbiol. 8:2496).
    Hansen & Atlung 2018 (Front. Microbiol. 9:319).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import (
    DNAA_ADP_BULK_ID, DNAA_ATP_BULK_ID,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ddah"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# Default constitutive hydrolysis rate (per second per DnaA-ATP molecule).
# Smaller than RIDA's per-replisome rate so DDAH stays a backup.
DEFAULT_K_DDAH_PER_S: float = 0.0005


class DDAH(Step):
    """DDAH Step — constitutive DnaA-ATP hydrolysis at the datA locus.

    First-order in DnaA-ATP. Pairs with RIDA (Phase 5): RIDA
    dominates while replisomes are active, DDAH provides a steady
    background drain.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'rate_per_s': f'float{{{DEFAULT_K_DDAH_PER_S}}}',
        'seed': 'integer{0}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.k = float(self.parameters.get(
            'rate_per_s', DEFAULT_K_DDAH_PER_S))
        self.random_state = np.random.RandomState(
            seed=int(self.parameters.get('seed', 0)))
        self._bulk_idx = None
        # Running cumulative DnaA-ATP -> DnaA-ADP flux across all
        # ticks. The per-tick flux is small (Poisson on a few hundred
        # molecules at a small rate constant) and snapshots are
        # taken much less frequently than ticks, so a per-tick
        # listener can read 0 in most snapshots even when DDAH is
        # firing several times per snapshot interval. The cumulative
        # field lets the report compute per-period flux as a
        # difference between consecutive snapshots.
        self._cumulative_flux: int = 0

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'ddah': {
                    'flux_atp_to_adp': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'cumulative_flux_atp_to_adp': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'rate_constant': {
                        '_type': 'overwrite[float]', '_default': []},
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

        atp_count, _adp_count = counts(states['bulk'], self._bulk_idx)
        atp_count = int(atp_count)
        dt = float(states.get('timestep', 1.0))

        n_hydrolyzed = 0
        if atp_count > 0 and dt > 0 and self.k > 0:
            expected = self.k * atp_count * dt
            n_hydrolyzed = int(
                min(self.random_state.poisson(expected), atp_count))

        # Bulk indices: [DnaA-ATP, DnaA-ADP] -> delta = [-h, +h]
        delta = np.array([-n_hydrolyzed, n_hydrolyzed], dtype=np.int64)

        self._cumulative_flux += n_hydrolyzed

        return {
            'bulk': [(self._bulk_idx, delta)],
            'listeners': {
                'ddah': {
                    'flux_atp_to_adp': n_hydrolyzed,
                    'cumulative_flux_atp_to_adp': self._cumulative_flux,
                    'rate_constant': float(self.k),
                },
            },
        }
