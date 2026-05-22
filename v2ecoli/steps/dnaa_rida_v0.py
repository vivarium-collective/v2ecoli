"""
=================================
DnaA RIDA-v0 (fork-active hydrolysis)
=================================

A placeholder for the in-vivo Regulatory Inactivation of DnaA (RIDA)
mechanism: a fork-loaded β-clamp (via the Hda adapter) accelerates
DnaA-ATP → DnaA-ADP hydrolysis by ~100× during S phase, then drops to
background once replication completes.

This v0 implementation captures the order-of-magnitude effect without
modelling the clamp-loading or Hda kinetics explicitly. Production RIDA
will land in dnaa-05 with explicit Hda + β-clamp dynamics.

Mechanism (this Step)
---------------------

    if active_replisome count > 0:
        effective hydrolysis rate = rida_rate_per_min      (default ~4.6/min)
    else:
        effective hydrolysis rate = 0                       (no contribution)

At the bulk level: MONOMER0-160[c] → MONOMER0-4565[c], same channel as
DnaaIntrinsicHydrolysis. Listener field
``listeners.dnaA_cycle.rida_events`` records the per-step count for the
ATP-fraction-band test panel + downstream-isolation diagnostics.

Reads ``unique.active_replisome`` and counts entries with
``_entryState`` truthy (the v2ecoli convention for "this slot holds a
real replisome instance, not an empty placeholder"). When this count is
zero the Step is a no-op.

Why a separate Step (not a parameter on DnaaIntrinsicHydrolysis)
---------------------------------------------------------------

Intrinsic hydrolysis (Sekimizu 1987, k=0.046/min) is always on and is
constitutive biology — every bound DnaA-ATP hydrolyzes at the base rate
regardless of fork state. RIDA is conditional, ~100× faster, and goes
away once forks complete. Keeping them in separate Steps:

  - lets each have its own rate constant + literature anchor
  - lets the dnaa-02f comparison cleanly toggle RIDA on / off without
    touching the intrinsic rate
  - matches how the in-vivo network is structured (different molecular
    machinery for the two pathways)

Validates
---------
- dnaa-02f behavior_tests.atp-fraction-band-across-variants (variant E)
- dnaa-02f behavior_tests.downstream-isolation-monomer160 (variant E)
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-rida-v0"
TOPOLOGY = {
    "bulk":     ("bulk",),
    "unique":   ("unique",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaRidaV0(Step):
    """RIDA-v0 placeholder Step: fork-gated DnaA-ATP → DnaA-ADP hydrolysis.

    Parameters
    ----------
    rida_rate_per_min : float
        Effective hydrolysis rate per minute when at least one replisome
        is active. Default 4.6/min = 100× the Sekimizu intrinsic rate;
        the literature-motivated order-of-magnitude effect of RIDA on
        DnaA-ATP turnover during S phase.
    intrinsic_rate_per_min : float
        The companion DnaaIntrinsicHydrolysis rate. Used only to compute
        the listener-reported RIDA *multiplier* relative to intrinsic;
        does NOT change RIDA's own rate. Default 0.046 to match
        Sekimizu 1987.
    deterministic : bool
        Round expected events instead of Poisson draws. Tests only.
    seed : int
        RNG seed for the Poisson draw.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "rida_rate_per_min":      {"_type": "float", "_default": 4.6},
        "intrinsic_rate_per_min": {"_type": "float", "_default": 0.046},
        "deterministic":          {"_type": "boolean", "_default": False},
        "seed":                   {"_type": "integer", "_default": 0},
        "time_step":              {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rida_rate_per_min      = float(self.parameters["rida_rate_per_min"])
        self.intrinsic_rate_per_min = float(self.parameters["intrinsic_rate_per_min"])
        self.deterministic          = bool(self.parameters["deterministic"])
        self.seed                   = int(self.parameters["seed"])
        self.random_state           = np.random.RandomState(seed=self.seed + 7919)
        self._atp_idx: int | None = None
        self._adp_idx: int | None = None

    def inputs(self):
        return {
            "bulk":     {"_type": "bulk_array", "_default": []},
            "unique":   "node",
            "timestep": {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "dnaA_cycle": {
                    "rida_events":       {"_type": "overwrite[integer]", "_default": 0},
                    "rida_active":       {"_type": "overwrite[integer]", "_default": 0},
                    "rida_n_replisomes": {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    def update(self, states, interval=None):
        if self._atp_idx is None:
            self._atp_idx = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        # Count active replisomes via the v2ecoli unique-store convention:
        # the structured array has an ``_entryState`` field whose truthy
        # entries are real replisomes. A zero count means no S phase right
        # now and RIDA is dormant.
        unique = states.get("unique") or {}
        rep = unique.get("active_replisome") if isinstance(unique, dict) else None
        n_replisomes = 0
        if rep is not None and hasattr(rep, "dtype") and (
                "_entryState" in (rep.dtype.names or ())):
            n_replisomes = int(rep["_entryState"].sum())

        # Dormant branch: no transfer, listener emits zeros.
        if n_replisomes <= 0:
            return {
                "listeners": {
                    "dnaA_cycle": {
                        "rida_events":       0,
                        "rida_active":       0,
                        "rida_n_replisomes": 0,
                    },
                },
            }

        # Fork-active branch: hydrolyze at rida_rate_per_min.
        atp_count  = int(counts(states["bulk"], self._atp_idx))
        timestep_s = float(states.get("timestep", 1.0))
        dt_min     = timestep_s / 60.0
        mean_events = self.rida_rate_per_min * dt_min * atp_count

        if self.deterministic:
            events = int(round(mean_events))
        else:
            events = int(self.random_state.poisson(mean_events))

        events = max(0, min(events, atp_count))  # don't drive ATP-bulk negative

        if events <= 0:
            return {
                "listeners": {
                    "dnaA_cycle": {
                        "rida_events":       0,
                        "rida_active":       1,
                        "rida_n_replisomes": n_replisomes,
                    },
                },
            }

        idx   = np.array([self._atp_idx, self._adp_idx], dtype=int)
        delta = np.array([-events, events],              dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {
                "dnaA_cycle": {
                    "rida_events":       events,
                    "rida_active":       1,
                    "rida_n_replisomes": n_replisomes,
                },
            },
        }
