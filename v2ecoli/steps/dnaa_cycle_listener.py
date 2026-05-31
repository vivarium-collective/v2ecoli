"""
=====================
DnaA Cycle Listener
=====================

Emits the DnaA nucleotide-state fractions each timestep:

    listeners.dnaA_cycle.apo_count          int   (PD03831[c])
    listeners.dnaA_cycle.atp_count          int   (MONOMER0-160[c])
    listeners.dnaA_cycle.adp_count          int   (MONOMER0-4565[c])
    listeners.dnaA_cycle.total              int   (sum of the three)
    listeners.dnaA_cycle.atp_fraction       float (atp / total)
    listeners.dnaA_cycle.adp_fraction       float (adp / total)
    listeners.dnaA_cycle.apo_fraction       float (apo / total)

These are the planning observables for dnaa-02 behavior_tests
(``apo-dnaa-fraction-is-small`` and
``dnaa-atp-fraction-in-physiological-range``). Total = 0 is reported
as fraction = NaN; pass-condition checks must handle that case.

The three species are existing v2ecoli bulk ids — see the audit note in
``v2ecoli/steps/dnaa_intrinsic_hydrolysis.py`` and the dnaa-02 study
``expert_decisions_needed[id=dnaa-02-EQ-02]``.
"""

from __future__ import annotations

import math

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-cycle-listener"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
}

DNAA_APO_ID = "PD03831[c]"
DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaCycleListener(Step):
    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self._idx_apo: int | None = None
        self._idx_atp: int | None = None
        self._idx_adp: int | None = None

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
        }

    def outputs(self):
        return {
            "listeners": {
                "dnaA_cycle": {
                    "apo_count":     {"_type": "overwrite[integer]", "_default": 0},
                    "atp_count":     {"_type": "overwrite[integer]", "_default": 0},
                    "adp_count":     {"_type": "overwrite[integer]", "_default": 0},
                    "total":         {"_type": "overwrite[integer]", "_default": 0},
                    "atp_fraction":  {"_type": "overwrite[float]",   "_default": 0.0},
                    "adp_fraction":  {"_type": "overwrite[float]",   "_default": 0.0},
                    "apo_fraction":  {"_type": "overwrite[float]",   "_default": 0.0},
                },
            },
        }

    def update(self, states, interval=None):
        if self._idx_apo is None:
            self._idx_apo = int(bulk_name_to_idx(DNAA_APO_ID, states["bulk"]["id"]))
            self._idx_atp = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._idx_adp = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        apo = int(counts(states["bulk"], self._idx_apo))
        atp = int(counts(states["bulk"], self._idx_atp))
        adp = int(counts(states["bulk"], self._idx_adp))
        total = apo + atp + adp

        if total > 0:
            atp_f = atp / total
            adp_f = adp / total
            apo_f = apo / total
        else:
            atp_f = adp_f = apo_f = math.nan

        return {
            "listeners": {
                "dnaA_cycle": {
                    "apo_count":     apo,
                    "atp_count":     atp,
                    "adp_count":     adp,
                    "total":         total,
                    "atp_fraction":  atp_f,
                    "adp_fraction":  adp_f,
                    "apo_fraction":  apo_f,
                },
            },
        }
