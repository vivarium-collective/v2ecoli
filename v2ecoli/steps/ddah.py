"""
=========================
DDAH — datA-dependent DnaA-ATP hydrolysis (dnaa-5)
=========================

DDAH (datA-dependent DnaA-ATP Hydrolysis) is the titration/hydrolysis of
DnaA-ATP at the datA locus, a high-capacity DnaA-binding region. Its activity
scales with the datA locus copy number (gene dosage) — after datA is replicated
its copy number doubles, increasing DnaA-ATP inactivation. Like RIDA it is an
EXTRINSIC ATP -> ADP conversion on top of the intrinsic hydrolysis:

    DnaA-ATP + WATER -> DnaA-ADP + Pi + PROTON

    flux = k_ddah * copy_number(datA) * dt        (capped by free DnaA-ATP)

copy_number(datA) is computed from the live replication-fork positions
(v2ecoli.library.locus_copy_number). Knockout: rate_multiplier = 0.

Config knobs:
    DDAH_RATE_PER_MIN        env override of the per-copy rate (default 12/min).
    rate_multiplier          1.0 = on, 0.0 = knockout.
"""

from __future__ import annotations

import os

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import ACTIVE_REPLISOME_ARRAY
from v2ecoli.library.locus_copy_number import locus_copy_number, DATA_COORD


NAME = "ddah"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "ddah"),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"
PI_ID = "Pi[c]"
PROTON_ID = "PROTON[c]"
WATER_ID = "WATER[c]"

DDAH_RATE_PER_MIN = float(os.environ.get("DDAH_RATE_PER_MIN", "12.0"))


class Ddah(Step):
    """datA-copy-number-coupled DnaA-ATP -> DnaA-ADP hydrolysis (DDAH)."""

    description = (
        "DDAH — datA-locus-coupled DnaA-ATP hydrolysis. "
        "flux = k_ddah * copy_number(datA) * dt, applied as "
        "DnaA-ATP + WATER -> DnaA-ADP + Pi + PROTON, capped by free DnaA-ATP."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'rate_per_copy_per_min': {'_type': 'float', '_default': DDAH_RATE_PER_MIN},
        'rate_multiplier': {'_type': 'float', '_default': 1.0},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }

    def initialize(self, config):
        self.rate_per_copy_per_min = float(
            self.parameters.get("rate_per_copy_per_min", DDAH_RATE_PER_MIN))
        self.rate_multiplier = float(self.parameters.get("rate_multiplier", 1.0))
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])
        self._atp_idx = self._adp_idx = self._pi_idx = self._proton_idx = self._water_idx = None

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
            self._atp_idx = bulk_name_to_idx(DNAA_ATP_ID, ids)
            self._adp_idx = bulk_name_to_idx(DNAA_ADP_ID, ids)
            self._pi_idx = bulk_name_to_idx(PI_ID, ids)
            self._proton_idx = bulk_name_to_idx(PROTON_ID, ids)
            self._water_idx = bulk_name_to_idx(WATER_ID, ids)

        next_time = states["global_time"] + states["timestep"]
        n_active = int(states["active_replisomes"]["_entryState"].sum())
        fork_coords = attrs(states["active_replisomes"], ["coordinates"])[0] if n_active else []
        copies = locus_copy_number(fork_coords, DATA_COORD)

        dt_min = float(states["timestep"]) / 60.0
        rate = self.rate_per_copy_per_min * self.rate_multiplier
        n = self._stochastic_round(rate * copies * dt_min)
        free_atp = int(counts(states["bulk"], self._atp_idx))
        if n > free_atp:
            n = free_atp

        update = {"next_update_time": next_time}
        if n > 0:
            update["bulk"] = [
                (self._atp_idx, -n), (self._adp_idx, n),
                (self._pi_idx, n), (self._proton_idx, n), (self._water_idx, -n),
            ]
        return update
