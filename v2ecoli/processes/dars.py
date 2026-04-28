"""
====
DARS
====

DnaA-Reactivating Sequences — converts DnaA-ADP back into apo-DnaA
(releasing ADP), the missing closing-of-the-cycle step for the DnaA
nucleotide-state model. The freed apo-DnaA then re-binds ATP through
the active equilibrium reaction ``MONOMER0-160_RXN``, regenerating
DnaA-ATP.

Two chromosomal loci, DARS1 and DARS2, drive this reactivation in vivo;
DARS2 is dominant. They are non-coding sites — there is no enzyme being
synthesized, just specific DNA architectures that catalyze nucleotide
release. In a fully-detailed model their activity would be gated by
IHF and Fis binding cycles (not yet wired in v2ecoli); for this phase
the rate is a simple first-order term in DnaA-ADP. IHF/Fis modulation
is a follow-up.

Mathematical model:

    rate = k_dars * dnaA_adp_count    [molecules / s]

    n_to_release ~ Poisson(rate * dt), capped at dnaA_adp_count

    DnaA-ADP -= n_to_release
    apo-DnaA += n_to_release

The default rate constant is chosen so that, paired with the Phase 5
RIDA rate, the steady-state DnaA-ATP fraction sits in the literature
band (30–70%). Tune as needed.

Reference (in the curated PDF):
    Fujimitsu K, Senriuchi T, Katayama T. Specific genomic sequences of
    E. coli promote replicational initiation by directly reactivating
    ADP-DnaA. Genes Dev. 23(10):1221–1233 (2009).
    Kasho K, Tanaka H, Sakai R, Katayama T. Timely binding of IHF and
    Fis to DARS2 regulates ATP–DnaA production and replication
    initiation. Nucleic Acids Res. 42(21):13134–13149 (2014).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import (
    DNAA_ADP_BULK_ID, DNAA_APO_BULK_ID,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dars"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# Default first-order rate constant for DnaA-ADP -> apo-DnaA release.
# Paired with the Phase 5 RIDA default (0.005 / replisome / s), the
# steady-state DnaA-ATP fraction lands inside the literature band.
# Treat as provisional; re-verify against measured DnaA-ATP cell-cycle
# dynamics before relying on it.
DEFAULT_K_DARS_PER_S: float = 0.01


class DARS(Step):
    """DARS Step

    Releases ADP from DnaA-ADP at a first-order rate, regenerating
    apo-DnaA. The companion equilibrium reaction (MONOMER0-160_RXN)
    then re-loads apo-DnaA with ATP, completing the DnaA cycle.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'rate_per_s': f'float{{{DEFAULT_K_DARS_PER_S}}}',
        'seed': 'integer{0}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.k = float(self.parameters.get(
            'rate_per_s', DEFAULT_K_DARS_PER_S))
        self.random_state = np.random.RandomState(
            seed=int(self.parameters.get('seed', 0)))
        self._bulk_idx = None

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
                'dars': {
                    'flux_adp_to_apo': {
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
                [DNAA_ADP_BULK_ID, DNAA_APO_BULK_ID], ids)

        adp_count, _apo_count = counts(states['bulk'], self._bulk_idx)
        adp_count = int(adp_count)

        dt = float(states.get('timestep', 1.0))

        n_released = 0
        if adp_count > 0 and dt > 0 and self.k > 0:
            expected = self.k * adp_count * dt
            n_released = int(
                min(self.random_state.poisson(expected), adp_count))

        # Bulk indices: [DnaA-ADP, apo-DnaA] -> delta = [-released, +released]
        delta = np.array([-n_released, n_released], dtype=np.int64)

        return {
            'bulk': [(self._bulk_idx, delta)],
            'listeners': {
                'dars': {
                    'flux_adp_to_apo': n_released,
                    'rate_constant': float(self.k),
                },
            },
        }
