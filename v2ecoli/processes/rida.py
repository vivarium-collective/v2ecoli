"""
====
RIDA
====

Regulatory Inactivation of DnaA — converts DnaA-ATP to DnaA-ADP at a
rate proportional to active replisome count, modeling the in vivo
mechanism where the DNA-loaded β-clamp + Hda complex catalytically
hydrolyzes DnaA-ATP during ongoing replication.

Why a dedicated process and not the FBA reaction?
    The corresponding stoichiometric reaction is already in the model
    (``RXN0-7444`` in ``flat/metabolic_reactions.tsv``, catalyzed by
    ``CPLX0-10342`` = Hda-β-clamp complex; see also Phase 5 in the
    replication-initiation report). It is registered with the FBA
    metabolism step but carries zero flux because nothing demands the
    products and the reaction has no kinetic constraint. Rather than
    re-engineer the FBA's objective or kinetic-constraint set to push
    flux through one regulatory reaction, we add a focused Step that
    directly transfers the molecules at a biologically motivated rate.

Mathematical model:

    rate = k * n_active_replisomes * dnaA_atp_count    [molecules / s]

    n_to_hydrolyze ~ Poisson(rate * dt), capped at dnaA_atp_count

    DnaA-ATP -= n_to_hydrolyze
    DnaA-ADP += n_to_hydrolyze

The single rate constant ``k`` (per replisome per second per DnaA-ATP
molecule) is the tuning knob. Initial value chosen so that with two
replisomes the half-life of DnaA-ATP in the absence of regeneration is
on the order of minutes — comparable to the cell cycle.

Reference:
    Katayama T, Kasho K, Kawakami H. The DnaA cycle in Escherichia coli:
    activation, function and inactivation of the initiator protein.
    Front. Microbiol. 8:2496 (2017). [in curated PDF reference list]
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import (
    DNAA_ADP_BULK_ID, DNAA_ATP_BULK_ID,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.schema_types import ACTIVE_REPLISOME_ARRAY


NAME = "rida"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# Default per-replisome, per-second, per-DnaA-ATP-molecule rate constant
# for the Hda-β-clamp-catalyzed hydrolysis. Treat as provisional —
# verify against measured DnaA-ATP cell-cycle dynamics before treating
# as a load-bearing parameter.
DEFAULT_K_PER_REPLISOME_PER_S: float = 0.005


class RIDA(Step):
    """RIDA Step

    Catalytically converts DnaA-ATP -> DnaA-ADP at a rate proportional
    to active replisome count. Operates as a single-pass step (no
    request/allocate cycle): the only molecules it touches are the two
    DnaA bulk pools, both produced/consumed by the equilibrium step
    each tick, so direct application is safe.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'rate_per_replisome_per_s': f'float{{{DEFAULT_K_PER_REPLISOME_PER_S}}}',
        'seed': 'integer{0}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.k = float(self.parameters.get(
            'rate_per_replisome_per_s', DEFAULT_K_PER_REPLISOME_PER_S))
        self.random_state = np.random.RandomState(
            seed=int(self.parameters.get('seed', 0)))
        self._bulk_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'active_replisomes': {
                '_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'rida': {
                    'flux_atp_to_adp': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'active_replisomes': {
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

        replisomes = states['active_replisomes']
        n_replisomes = 0
        if (replisomes is not None
                and hasattr(replisomes, 'dtype')
                and '_entryState' in replisomes.dtype.names):
            n_replisomes = int(
                replisomes['_entryState'].view(np.bool_).sum())

        dt = float(states.get('timestep', 1.0))

        n_hydrolyzed = 0
        if atp_count > 0 and n_replisomes > 0 and dt > 0 and self.k > 0:
            expected = self.k * n_replisomes * atp_count * dt
            n_hydrolyzed = int(
                min(self.random_state.poisson(expected), atp_count))

        delta = np.array([-n_hydrolyzed, n_hydrolyzed], dtype=np.int64)

        return {
            'bulk': [(self._bulk_idx, delta)],
            'listeners': {
                'rida': {
                    'flux_atp_to_adp': n_hydrolyzed,
                    'active_replisomes': n_replisomes,
                    'rate_constant': float(self.k),
                },
            },
        }
