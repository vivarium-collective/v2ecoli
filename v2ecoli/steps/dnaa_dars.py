"""
================================================
DARS1 / DARS2 — DnaA-ADP -> DnaA-ATP reactivation
================================================

DARS1 and DARS2 (DnaA-Reactivating Sequences) are chromosomal loci that
catalyse nucleotide exchange on DnaA-ADP, regenerating DnaA-ATP. They are the
RECOVERY arm of the DnaA cycle — the only route that raises the DnaA-ATP
fraction between initiations, balancing the lowering by RIDA + datA + intrinsic
hydrolysis. Without DARS the fraction can only ratchet down.

At the v2ecoli bulk level this transmutes ``MONOMER0-4565[c]`` (DnaA-ADP) back
into ``MONOMER0-160[c]`` (DnaA-ATP) — the exact reverse of the hydrolysis Steps,
on the same two existing bulk species (no new ids). The bound ADP is exchanged
for ATP within the complex; the nucleotide pools are folded into metabolism and
not tracked here (sub-mass-balance-noise, as for intrinsic hydrolysis).

Rate
----
wcm_stage1_parameters (Fu/Xiao/Jun 2023) heuristic: DARS1 ≈ 5 /min per locus,
DARS2 ≈ 10 /min per locus. The Step applies the summed first-order rate on the
DnaA-ADP-complex count (DARS2 is the stronger, growth-rate-regulated locus).

Validates
---------
dnaa-06 behavior_tests (full extrinsic network holding the DnaA-ATP fraction in
band AND reproducing its cell-cycle oscillation — DARS supplies the recovery).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-dars"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
    "timestep":  ("timestep",),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaDars(Step):
    """First-order DnaA-ADP -> DnaA-ATP reactivation (DARS1 + DARS2).

    Reads the DnaA-ADP-complex count, applies a Poisson draw with mean
    ``(dars1_rate_per_min + dars2_rate_per_min) * dt_min * count``, and writes
    the transfer as a bulk delta MONOMER0-4565 -> MONOMER0-160 (the reverse of
    the hydrolysis Steps). Stochastic by default; ``deterministic=True`` for a
    deterministic round.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "dars1_rate_per_min": {"_type": "float", "_default": 5.0},
        "dars2_rate_per_min": {"_type": "float", "_default": 10.0},
        "deterministic":      {"_type": "boolean", "_default": False},
        "seed":               {"_type": "integer", "_default": 0},
        "time_step":          {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rate_per_min = (float(self.parameters["dars1_rate_per_min"])
                             + float(self.parameters["dars2_rate_per_min"]))
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
                    "dars_reactivation_events": {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    def update(self, states, interval=None):
        if self._atp_idx is None:
            self._atp_idx = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        adp_count = int(counts(states["bulk"], self._adp_idx))
        dt_min = float(states.get("timestep", 1.0)) / 60.0
        mean_events = self.rate_per_min * dt_min * adp_count

        if self.deterministic:
            events = int(round(mean_events))
        else:
            events = int(self.random_state.poisson(mean_events))
        events = min(events, adp_count)  # never drive the ADP pool negative

        if events <= 0:
            return {"listeners": {"dnaA_cycle": {"dars_reactivation_events": 0}}}

        # Reverse of hydrolysis: +events DnaA-ATP, -events DnaA-ADP.
        idx = np.array([self._atp_idx, self._adp_idx], dtype=int)
        delta = np.array([events, -events], dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {"dnaA_cycle": {"dars_reactivation_events": events}},
        }
