"""
=========================
DnaA Intrinsic Hydrolysis
=========================

First-order intrinsic ATP hydrolysis on DnaA:

    DnaA-ATP  --(k_intrinsic)-->  DnaA-ADP

At the v2ecoli bulk level this transmutes
``MONOMER0-160[c]`` (DnaA-ATP complex) into ``MONOMER0-4565[c]``
(DnaA-ADP complex). The bound ATP becomes bound ADP within the
complex; the released inorganic phosphate is folded into the
metabolism pool and not tracked here (its mass is ~100 Da vs
DnaA's 52 kDa — well below the per-tick mass-balance noise).

Rate
----
Sekimizu et al. 1987 (Cell 50.2): tightly bound DnaA-ATP hydrolyzes
~50% over 15 min in vitro. Treating as first-order:

    k_intrinsic = ln(2) / 15 min  ≈  0.046 / min

This rate alone is too slow to explain the in-vivo DnaA-ATP /
total-DnaA band of [0.2, 0.5] (Boesen 2024 PNAS), which is why
the full network requires the extrinsic RIDA / DDAH / DARS
machinery (dnaa-05 in the investigation, deferred). In this
study the rate is exposed as the composite parameter
``dnaA_intrinsic_hydrolysis_rate_per_min`` so a clamp Step
(req-4) can stand in for the missing extrinsic flux.

Validates
---------
- dnaa-02 behavior_tests.no-hydrolysis-accumulates-atp-form
  (rate=0 → DnaA-ATP fraction trends to 1.0)
- dnaa-02 behavior_tests.fast-hydrolysis-suppresses-atp-form
  (rate=10× default → DnaA-ATP fraction drops below 0.1)

Audit note (2026-05-17)
-----------------------
``MONOMER0-160[c]`` and ``MONOMER0-4565[c]`` are existing bulk
species in v2ecoli sim_data. The two equilibrium reactions
``MONOMER0-160_RXN`` (DnaA + ATP → DnaA-ATP) and
``MONOMER0-4565_RXN`` (DnaA + ADP → DnaA-ADP) already define
the binding pathways. This Step adds the missing intrinsic
hydrolysis link between the two complexes — no new bulk ids
needed. See ``studies/dnaa-02-atp-hydrolysis/study.yaml``
``expert_decisions_needed[id=dnaa-02-EQ-02]`` for the audit log.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-intrinsic-hydrolysis"
TOPOLOGY = {
    "bulk":     ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaIntrinsicHydrolysis(Step):
    """First-order DnaA-ATP -> DnaA-ADP hydrolysis Step.

    Reads the current DnaA-ATP-complex count, applies a Poisson draw
    with mean ``rate_per_min * dt_min * count``, and writes the resulting
    transfer as a bulk delta to MONOMER0-160 / MONOMER0-4565.

    The Step is stochastic-by-default (Poisson) so a single cell at low
    DnaA counts doesn't produce a deterministic rounding error each tick.
    For deterministic tests, pass ``deterministic=True`` in the parameters.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "rate_per_min": {"_type": "float", "_default": 0.046},
        "deterministic": {"_type": "boolean", "_default": False},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rate_per_min = float(self.parameters["rate_per_min"])
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
                    "intrinsic_hydrolysis_events": {
                        "_type": "overwrite[integer]",
                        "_default": 0,
                    },
                },
            },
        }

    def update(self, states, interval=None):
        if self._atp_idx is None:
            self._atp_idx = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        # Read current DnaA-ATP complex count
        atp_count = int(counts(states["bulk"], self._atp_idx))
        timestep_s = float(states.get("timestep", 1.0))
        dt_min = timestep_s / 60.0

        # Expected number of hydrolysis events this step. With first-order
        # kinetics and an integer count, Poisson(k * dt * count) gives the
        # right mean and variance.
        mean_events = self.rate_per_min * dt_min * atp_count

        if self.deterministic:
            events = int(round(mean_events))
        else:
            # Cap by the available ATP-complex count so we don't drive it negative.
            events = int(self.random_state.poisson(mean_events))
            events = min(events, atp_count)

        if events <= 0:
            return {
                "listeners": {
                    "dnaA_cycle": {"intrinsic_hydrolysis_events": 0}
                }
            }

        # bulk delta: -events on DnaA-ATP, +events on DnaA-ADP
        idx = np.array([self._atp_idx, self._adp_idx], dtype=int)
        delta = np.array([-events, events], dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {
                "dnaA_cycle": {"intrinsic_hydrolysis_events": events}
            },
        }
