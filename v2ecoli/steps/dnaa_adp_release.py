"""
=====================
DnaA-ADP Slow Release
=====================

Slow, NON-EQUILIBRIUM dissociation of the DnaA-ADP complex:

    DnaA-ADP  --(k_release, slow)-->  apo-DnaA  ( + ADP )

At the v2ecoli bulk level this transmutes ``MONOMER0-4565[c]`` (DnaA-ADP
complex) back into ``PD03831[c]`` (apo-DnaA). The freed apo-DnaA is then
re-bound by the (still fast-equilibrium) ATP reaction MONOMER0-160_RXN —
and because cellular ATP greatly exceeds ADP, it re-forms DnaA-ATP.

Why this is a Step and not part of ecoli-equilibrium
----------------------------------------------------
Haochen (2026-05-31): the model wrongly assumed FAST EQUILIBRIUM for BOTH
DnaA+ATP<->DnaA-ATP and DnaA+ADP<->DnaA-ADP. The ADP side is not at
equilibrium — the reverse DnaA-ADP -> apo + ADP is slow (k_r ~1e-7/s,
<< the forward/binding ~1e-4/s). v2ecoli's ecoli-equilibrium step integrates
every reaction to steady state each tick, so it cannot represent a
kinetically-trapped DnaA-ADP: it instantly drains hydrolysis-produced
DnaA-ADP back to apo -> DnaA-ATP, pinning the DnaA-ATP fraction at ~0.997
(observed in dnaa-2 Steps 1-2). The fix is two-part:

  1. Zero MONOMER0-4565_RXN's rates in the equilibrium config so the
     equilibrium step leaves DnaA-ADP untouched (see
     scripts/patch_dnaa_adp_nonequilibrium.py). DnaA-ADP is then formed
     ONLY by intrinsic hydrolysis (DnaaIntrinsicHydrolysis) and drained
     ONLY by this slow kinetic release.
  2. This Step drains DnaA-ADP at the slow physiological k_release, so
     DnaA-ADP accumulates and the DnaA-ATP fraction falls toward the
     Boesen 2024 [0.2, 0.5] band.

Released ADP / consumed ATP
---------------------------
Like DnaaIntrinsicHydrolysis (which folds the released Pi into metabolism
rather than tracking it), this Step does NOT modify the cellular ATP[c] /
ADP[c] pools — those are ~1e6-molecule pools managed by metabolism/FBA, and
the few DnaA-bound nucleotides per tick are far below that balance's noise.
Only the DnaA-form bulk species move: -DnaA-ADP, +apo-DnaA.

Rate
----
k_release exposed as ``rate_per_min`` (default the slow ~1e-7/s ≈ 6e-6/min);
overridable per run via env var ``DNAA_ADP_RELEASE_RATE`` (see
v2ecoli/composites/_helpers.py). The value that lands the DnaA-ATP fraction
in [0.2, 0.5] is Haochen's to confirm; this Step makes the rate a knob.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-adp-release"
TOPOLOGY = {
    "bulk":     ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}

DNAA_APO_ID = "PD03831[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaAdpRelease(Step):
    """First-order slow DnaA-ADP -> apo-DnaA release Step.

    Reads the current DnaA-ADP-complex count, applies a Poisson draw with
    mean ``rate_per_min * dt_min * count``, and writes the transfer as a bulk
    delta to MONOMER0-4565 (-) / PD03831 (+). Stochastic by default; pass
    ``deterministic=True`` for tests.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # ~1e-7/s ≈ 6e-6/min. Deliberately slow (non-equilibrium); the value
        # that reproduces the [0.2,0.5] DnaA-ATP band is Haochen's to confirm.
        "rate_per_min": {"_type": "float", "_default": 6.0e-6},
        "deterministic": {"_type": "boolean", "_default": False},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rate_per_min = float(self.parameters["rate_per_min"])
        self.deterministic = bool(self.parameters["deterministic"])
        self.seed = int(self.parameters["seed"])
        self.random_state = np.random.RandomState(seed=self.seed)
        self._apo_idx: int | None = None
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
                    "adp_release_events": {
                        "_type": "overwrite[integer]",
                        "_default": 0,
                    },
                },
            },
        }

    def update(self, states, interval=None):
        if self._apo_idx is None:
            self._apo_idx = int(bulk_name_to_idx(DNAA_APO_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        adp_count = int(counts(states["bulk"], self._adp_idx))
        timestep_s = float(states.get("timestep", 1.0))
        dt_min = timestep_s / 60.0

        mean_events = self.rate_per_min * dt_min * adp_count
        if self.deterministic:
            events = int(round(mean_events))
        else:
            events = int(self.random_state.poisson(mean_events))
            events = min(events, adp_count)

        if events <= 0:
            return {"listeners": {"dnaA_cycle": {"adp_release_events": 0}}}

        # bulk delta: -events on DnaA-ADP, +events on apo-DnaA
        idx = np.array([self._adp_idx, self._apo_idx], dtype=int)
        delta = np.array([-events, events], dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {
                "dnaA_cycle": {"adp_release_events": events}
            },
        }
