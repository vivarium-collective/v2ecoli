"""
=========================
DARS — DnaA-Reactivating Sequences DARS1 / DARS2 (dnaa-5)
=========================

DARS1 and DARS2 are chromosomal loci that REACTIVATE DnaA: they catalyze the
nucleotide exchange DnaA-ADP -> DnaA-ATP, regenerating the active form. This is
the recovery arm that opposes RIDA/DDAH inactivation and produces the
cell-cycle oscillation of DnaA-ATP (drop post-initiation via RIDA, recovery
mid-cycle via DARS). Each DARS activity scales with its locus copy number
(gene dosage), computed from the live replication-fork positions.

Modeled as a nucleotide exchange that consumes free ATP and releases free ADP
(the DnaA protein swaps its bound nucleotide):

    DnaA-ADP + ATP -> DnaA-ATP + ADP

    flux = (k_dars1 * copy(DARS1) + k_dars2 * copy(DARS2)) * dt   (capped by free DnaA-ADP)

Knockouts: dars1-knockout / dars2-knockout set the respective multiplier to 0
(env DARS1_RATE_MULTIPLIER / DARS2_RATE_MULTIPLIER).

Config knobs:
    DARS1_RATE_PER_MIN / DARS2_RATE_PER_MIN   per-copy rates (default 5 / 10 per min).
    dars1_multiplier / dars2_multiplier        1.0 = on, 0.0 = knockout.
"""

from __future__ import annotations

import os

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import ACTIVE_REPLISOME_ARRAY
from v2ecoli.library.locus_copy_number import locus_copy_number, DARS1_COORD, DARS2_COORD


NAME = "dars"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "dars"),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"
ATP_ID = "ATP[c]"
ADP_ID = "ADP[c]"

DARS1_RATE_PER_MIN = float(os.environ.get("DARS1_RATE_PER_MIN", "5.0"))
DARS2_RATE_PER_MIN = float(os.environ.get("DARS2_RATE_PER_MIN", "10.0"))


class Dars(Step):
    """DARS1+DARS2 copy-number-coupled DnaA-ADP -> DnaA-ATP reactivation."""

    description = (
        "DARS1/DARS2 — locus-copy-number-coupled DnaA reactivation. "
        "flux = (k1*copy(DARS1) + k2*copy(DARS2))*dt, applied as "
        "DnaA-ADP + ATP -> DnaA-ATP + ADP, capped by free DnaA-ADP."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'dars1_rate_per_copy_per_min': {'_type': 'float', '_default': DARS1_RATE_PER_MIN},
        'dars2_rate_per_copy_per_min': {'_type': 'float', '_default': DARS2_RATE_PER_MIN},
        'dars1_multiplier': {'_type': 'float', '_default': 1.0},
        'dars2_multiplier': {'_type': 'float', '_default': 1.0},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }

    def initialize(self, config):
        self.k1 = float(self.parameters.get("dars1_rate_per_copy_per_min", DARS1_RATE_PER_MIN))
        self.k2 = float(self.parameters.get("dars2_rate_per_copy_per_min", DARS2_RATE_PER_MIN))
        self.m1 = float(self.parameters.get("dars1_multiplier", 1.0))
        self.m2 = float(self.parameters.get("dars2_multiplier", 1.0))
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])
        self._atp_idx = self._adp_idx = self._fatp_idx = self._fadp_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'float[s]', '_default': 1.0},
            'next_update_time': {'_type': 'overwrite[float[s]]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'next_update_time': 'overwrite[float[s]]',
        }

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def _stochastic_round(self, x: float) -> int:
        if x <= 0:
            return 0
        f = int(np.floor(x))
        return f + 1 if self.random_state.random_sample() < (x - f) else f

    def update(self, states, interval=None):
        if self._atp_idx is None:
            ids = states["bulk"]["id"]
            self._atp_idx = bulk_name_to_idx(DNAA_ATP_ID, ids)   # DnaA-ATP
            self._adp_idx = bulk_name_to_idx(DNAA_ADP_ID, ids)   # DnaA-ADP
            self._fatp_idx = bulk_name_to_idx(ATP_ID, ids)       # free ATP
            self._fadp_idx = bulk_name_to_idx(ADP_ID, ids)       # free ADP

        next_time = states["global_time"] + states["timestep"]
        n_active = int(states["active_replisomes"]["_entryState"].sum())
        fork_coords = attrs(states["active_replisomes"], ["coordinates"])[0] if n_active else []
        copy1 = locus_copy_number(fork_coords, DARS1_COORD)
        copy2 = locus_copy_number(fork_coords, DARS2_COORD)

        dt_min = float(states["timestep"]) / 60.0
        rate = self.k1 * self.m1 * copy1 + self.k2 * self.m2 * copy2
        n = self._stochastic_round(rate * dt_min)
        free_dnaa_adp = int(counts(states["bulk"], self._adp_idx))
        if n > free_dnaa_adp:
            n = free_dnaa_adp

        update = {"next_update_time": next_time}
        if n > 0:
            # DnaA-ADP + ATP -> DnaA-ATP + ADP  (nucleotide exchange)
            update["bulk"] = [
                (self._adp_idx, -n), (self._atp_idx, n),
                (self._fatp_idx, -n), (self._fadp_idx, n),
            ]
        return update
