"""
=========================
RIDA — Regulatory Inactivation of DnaA (dnaa-5)
=========================

RIDA is the replisome-coupled inactivation of DnaA: the Hda protein, loaded
onto the DNA-bound β-clamp at active replication forks, stimulates hydrolysis
of DnaA-ATP → DnaA-ADP. Its activity scales with the number of active
replisomes (more forks → more Hda·clamp → more inactivation), which is what
makes the DnaA-ATP pool DROP after initiation and recover later in the cycle.

This is an EXTRINSIC conversion that operates ON TOP OF the intrinsic
DnaA-ATP hydrolysis (owned by the equilibrium step, k≈0.025–0.046/min). Here
the rate is proportional to the active-replisome count:

    flux[DnaA-ATP → DnaA-ADP]  =  k_rida · n_active_replisomes · dt

following the same hydrolysis stoichiometry as DNAA-INTRINSIC-HYDROLYSIS-RXN:

    DnaA-ATP + WATER → DnaA-ADP + Pi + PROTON

The per-tick flux is small (k_rida·n_forks·dt ≈ O(1) molecule), so the
Pi/PROTON/WATER byproducts are written directly to bulk here (unlike the large
bound-pool hydrolysis in dnaa_box_binding, whose byproducts must route through
the equilibrium stoichMatrix to avoid perturbing FBA mass balance). The
conversion is also capped by the available free DnaA-ATP count.

Knockout: set ``rate_multiplier`` to 0.0 (the rida-knockout variant) to disable
RIDA without removing the process — its DnaA-ATP should then over-accumulate.

Config knobs:
    RIDA_RATE_PER_MIN   env override of the per-replisome rate (default 40/min,
                        from the stage-1 diagnostics target).
    rate_multiplier     config multiplier (1.0 = on, 0.0 = knockout).
"""

from __future__ import annotations

import os

import numpy as np

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
    "next_update_time": ("next_update_time", "rida"),
}

# Bulk IDs — DnaA nucleotide forms + hydrolysis byproducts (same as the
# intrinsic-hydrolysis stoichiometry: ATP + WATER → ADP + Pi + PROTON).
DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"
PI_ID = "Pi[c]"
PROTON_ID = "PROTON[c]"
WATER_ID = "WATER[c]"

# Per-replisome RIDA rate (1/min). 40/min/fork is the stage-1 diagnostics
# target for the DnaA-ATP post-initiation drop; env-overridable for calibration.
RIDA_RATE_PER_MIN = float(os.environ.get("RIDA_RATE_PER_MIN", "40.0"))


class Rida(Step):
    """Replisome-coupled DnaA-ATP → DnaA-ADP inactivation (RIDA)."""

    description = (
        "RIDA — replisome-coupled DnaA-ATP inactivation. "
        "flux = k_rida * n_active_replisomes * dt, applied as "
        "DnaA-ATP + WATER -> DnaA-ADP + Pi + PROTON, capped by free DnaA-ATP."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'rate_per_replisome_per_min': {'_type': 'float', '_default': RIDA_RATE_PER_MIN},
        'rate_multiplier': {'_type': 'float', '_default': 1.0},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }

    def initialize(self, config):
        self.rate_per_replisome_per_min = float(
            self.parameters.get("rate_per_replisome_per_min", RIDA_RATE_PER_MIN))
        self.rate_multiplier = float(self.parameters.get("rate_multiplier", 1.0))
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)
        self._atp_idx = None
        self._adp_idx = None
        self._pi_idx = None
        self._proton_idx = None
        self._water_idx = None

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
            'listeners': {
                'rida': {
                    'n_active_replisomes': {'_type': 'overwrite[integer]', '_default': 0},
                    'conversion_count': {'_type': 'overwrite[integer]', '_default': 0},
                },
            },
        }

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def _stochastic_round(self, x: float) -> int:
        if x <= 0:
            return 0
        floor_x = int(np.floor(x))
        frac = x - floor_x
        if self.random_state.random_sample() < frac:
            floor_x += 1
        return floor_x

    def update(self, states, interval=None):
        if self._atp_idx is None:
            bulk_ids = states["bulk"]["id"]
            self._atp_idx = bulk_name_to_idx(DNAA_ATP_ID, bulk_ids)
            self._adp_idx = bulk_name_to_idx(DNAA_ADP_ID, bulk_ids)
            self._pi_idx = bulk_name_to_idx(PI_ID, bulk_ids)
            self._proton_idx = bulk_name_to_idx(PROTON_ID, bulk_ids)
            self._water_idx = bulk_name_to_idx(WATER_ID, bulk_ids)

        next_time = states["global_time"] + states["timestep"]
        n_forks = int(states["active_replisomes"]["_entryState"].sum())

        effective_rate = self.rate_per_replisome_per_min * self.rate_multiplier
        dt_min = float(states["timestep"]) / 60.0
        rida_count = self._stochastic_round(effective_rate * n_forks * dt_min)

        # Cap by available free DnaA-ATP (can't inactivate more than exists).
        free_atp = int(counts(states["bulk"], self._atp_idx))
        if rida_count > free_atp:
            rida_count = free_atp

        update = {
            "next_update_time": next_time,
            "listeners": {
                "rida": {
                    "n_active_replisomes": n_forks,
                    "conversion_count": int(rida_count),
                },
            },
        }
        if rida_count > 0:
            # DnaA-ATP + WATER -> DnaA-ADP + Pi + PROTON
            update["bulk"] = [
                (self._atp_idx, -rida_count),
                (self._adp_idx, rida_count),
                (self._pi_idx, rida_count),
                (self._proton_idx, rida_count),
                (self._water_idx, -rida_count),
            ]
        return update
