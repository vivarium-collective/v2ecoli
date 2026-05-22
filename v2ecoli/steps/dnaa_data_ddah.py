"""
=====================================
datA-Dependent DnaA-ATP Hydrolysis
=====================================

datA (DDAH): a high-affinity DnaA-box cluster ~13 min from oriC that, with
IHF, promotes hydrolysis of DnaA-ATP to DnaA-ADP — the second extrinsic
DnaA-ATP-lowering route alongside RIDA. At the v2ecoli bulk level this
transmutes ``MONOMER0-160[c]`` (DnaA-ATP) into ``MONOMER0-4565[c]``
(DnaA-ADP), exactly as intrinsic hydrolysis and RIDA do — no new bulk ids.

Difference from RIDA: RIDA is replication-fork-coupled (active only while
forks are present), whereas datA is a fixed chromosomal locus, so this Step
applies a first-order rate on the DnaA-ATP pool per datA locus (default 1),
independent of fork state.

Rate
----
wcm_stage1_parameters (Fu/Xiao/Jun 2023) heuristic: datA hydrolyses DnaA-ATP
at ~12 /min per locus. Treated as first-order on the DnaA-ATP-complex count.

Validates
---------
dnaa-06 behavior_tests (full extrinsic network holding/oscillating the
DnaA-ATP fraction in Boesen 2024's [0.2, 0.5] band).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-data-ddah"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
    "timestep":  ("timestep",),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaDataDdah(Step):
    """First-order datA-dependent DnaA-ATP -> DnaA-ADP hydrolysis Step.

    Reads the DnaA-ATP-complex count, applies a Poisson draw with mean
    ``rate_per_min * n_loci * dt_min * count``, and writes the transfer as a
    bulk delta MONOMER0-160 -> MONOMER0-4565. Stochastic by default; pass
    ``deterministic=True`` for a deterministic round.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "rate_per_min": {"_type": "float", "_default": 12.0},
        "n_loci":       {"_type": "integer", "_default": 1},
        "deterministic": {"_type": "boolean", "_default": False},
        "seed":         {"_type": "integer", "_default": 0},
        "time_step":    {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rate_per_min = float(self.parameters["rate_per_min"])
        self.n_loci = int(self.parameters["n_loci"])
        self.deterministic = bool(self.parameters["deterministic"])
        self.seed = int(self.parameters["seed"])
        self.random_state = np.random.RandomState(seed=self.seed)
        self._atp_idx: int | None = None
        self._adp_idx: int | None = None

    def inputs(self):
        return {
            "bulk":     {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "dnaA_cycle": {
                    "data_ddah_events": {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    def update(self, states, interval=None):
        if self._atp_idx is None:
            self._atp_idx = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        atp_count = int(counts(states["bulk"], self._atp_idx))
        dt_min = float(states.get("timestep", 1.0)) / 60.0
        mean_events = self.rate_per_min * self.n_loci * dt_min * atp_count

        if self.deterministic:
            events = int(round(mean_events))
        else:
            events = int(self.random_state.poisson(mean_events))
        events = min(events, atp_count)  # never drive the ATP pool negative

        if events <= 0:
            return {"listeners": {"dnaA_cycle": {"data_ddah_events": 0}}}

        idx = np.array([self._atp_idx, self._adp_idx], dtype=int)
        delta = np.array([-events, events], dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {"dnaA_cycle": {"data_ddah_events": events}},
        }
