"""DnaA-ATP intrinsic hydrolysis Step (DRAFT — not wired into baseline).

This is a proposal for dnaa-02-atp-hydrolysis. It converts DnaA-ATP into
DnaA-ADP at the intrinsic hydrolysis rate measured by Boesen 2024.

Why we need it
--------------
v2ecoli's baseline composite has the DnaA-ATP/ADP equilibrium machinery
(MONOMER0-160 / MONOMER0-4565 species + ecoli-equilibrium process), but
no flux pushes DnaA-ATP → DnaA-ADP. The metabolic reaction RXN0-7444
(catalyzed by CPLX0-10342) is declared in `metabolic_reactions.tsv` but
has NO kinetic constraint in `metabolism_kinetics.tsv`, so the FBA
solver never selects it.

Result: nearly 100% of DnaA ends up ATP-bound (Insight #3 of overnight
run). Boesen 2024 measured an INTRINSIC ATPase activity in DnaA monomer
itself — k_intrinsic ≈ 0.046 min^-1 (independent of any catalyst), so
this happens regardless of the metabolic context.

Biological references
---------------------
- Boesen et al. 2024 (MolMicrobiol): in-vitro DnaA-ATP hydrolysis,
  k_intrinsic = 0.046 min^-1.
- Katayama 2017 (Frontiers Microbiol): DnaA-ATP / DnaA-ADP cycle review.
- expected_outcome: dnaA-atp-fraction-in-physiological-range test target
  [0.2, 0.5] (Boesen 2024, Sekimizu 1991).

Mathematical model
------------------
Each timestep dt (seconds):

    expected_hydrolyzed = (k_intrinsic / 60) * n_DnaA_ATP * dt

  where k_intrinsic = 0.046 / min.

  n_actual = stochasticRound(expected_hydrolyzed)
  n_actual = min(n_actual, n_DnaA_ATP)   # can't hydrolyze more than we have

    bulk[MONOMER0-160] -= n_actual    (DnaA-ATP consumed)
    bulk[MONOMER0-4565] += n_actual   (DnaA-ADP produced)

(WATER and Pi are abundant; we don't account for them here. The full
metabolic-balance bookkeeping in RXN0-7444 takes care of those when /
if FBA fires.)

How to wire into baseline
-------------------------
1. Save this file as `v2ecoli/processes/dnaa_intrinsic_hydrolysis.py`.
2. In `v2ecoli/composites/baseline.py`, add to the imports:
       from v2ecoli.processes.dnaa_intrinsic_hydrolysis import (
           DnaAIntrinsicHydrolysis, NAME as DNAA_HYDROLYSIS_NAME,
       )
3. Insert `DNAA_HYDROLYSIS_NAME: DnaAIntrinsicHydrolysis` in the
   `composite['processes']` dict.
4. Add the topology + initial state for a `listeners.dnaa_states` block.

Verification protocol (proposed)
--------------------------------
- Run baseline-with-hydrolysis-seed{0..4} for 10 min each.
- Track DnaA-ATP fraction at each timestep.
- Pass criterion: median ATP-fraction across last 5 min in [0.2, 0.5].
- Also re-evaluate the dnaa-01 gate tests with this composite (to
  check the hydrolysis doesn't break the F-01/F-02 results).
"""
from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils.random import stochasticRound


NAME = "dnaa-intrinsic-hydrolysis"

DNAA_ATP_ID = "MONOMER0-160[c]"     # DnaA-ATP
DNAA_ADP_ID = "MONOMER0-4565[c]"    # DnaA-ADP

# Boesen 2024 in-vitro DnaA-ATP intrinsic hydrolysis rate.
DEFAULT_K_INTRINSIC_PER_MIN = 0.046

TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}


class DnaAIntrinsicHydrolysis(Step):
    """Stochastic per-step DnaA-ATP → DnaA-ADP intrinsic hydrolysis."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "k_intrinsic_per_min": {
            "_type": "float[1/min]",
            "_default": DEFAULT_K_INTRINSIC_PER_MIN,
        },
        "seed": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        self.k_per_s = float(self.parameters["k_intrinsic_per_min"]) / 60.0
        self.rng = np.random.RandomState(int(self.parameters["seed"]))
        # Indices resolved on first update (need bulk['id'] vector)
        self._atp_idx = None
        self._adp_idx = None

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "integer[s]", "_default": 1},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "dnaa_intrinsic_hydrolysis": {
                    "n_hydrolyzed": {"_type": "overwrite[integer]", "_default": 0},
                    "rate_per_s":   {"_type": "overwrite[float[1/s]]",    "_default": 0.0},
                },
            },
        }

    def update(self, states, interval=None):
        bulk = states["bulk"]
        dt = float(states["timestep"])  # seconds

        if self._atp_idx is None:
            ids = bulk["id"]
            self._atp_idx = bulk_name_to_idx([DNAA_ATP_ID], ids)[0]
            self._adp_idx = bulk_name_to_idx([DNAA_ADP_ID], ids)[0]

        n_atp = int(bulk["count"][self._atp_idx])
        expected = self.k_per_s * n_atp * dt
        n_hyd = int(stochasticRound(self.rng, np.asarray([expected]))[0])
        n_hyd = max(0, min(n_hyd, n_atp))

        bulk_delta = np.zeros(len(bulk["count"]), dtype=np.int64)
        bulk_delta[self._atp_idx] = -n_hyd
        bulk_delta[self._adp_idx] = +n_hyd

        return {
            "bulk": bulk_delta,
            "listeners": {
                "dnaa_intrinsic_hydrolysis": {
                    "n_hydrolyzed": n_hyd,
                    "rate_per_s": self.k_per_s * n_atp,
                },
            },
        }
