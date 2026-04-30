"""
Sustained ppGpp + RpoS Stress Process
=======================================
Maintains ppGpp and optionally RpoS at target levels each timestep.
Models nutrient-limited stationary phase where RelA/SpoT balance keeps
ppGpp chronically elevated and ClpXP degradation of RpoS is reduced.
"""

import numpy as np

from v2ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "ecoli-ppgpp-sustained-stress"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}


class PpGppSustainedStress(Step):
    """Maintains ppGpp (and optionally RpoS) at target counts each timestep."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "time_step": {"_type": "float", "_default": 1.0},
        "ppgpp_id": {"_type": "string", "_default": "GUANOSINE-5DP-3DP[c]"},
        "target_count": {"_type": "integer", "_default": 250000},
        "rpos_id": {"_type": "string", "_default": "RPOS-MONOMER[c]"},
        "rpos_target_count": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        self.ppgpp_id = self.parameters["ppgpp_id"]
        self.ppgpp_target = int(self.parameters["target_count"])
        self.rpos_id = self.parameters["rpos_id"]
        self.rpos_target = int(self.parameters["rpos_target_count"])
        self._ppgpp_idx = None
        self._rpos_idx = None

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float", "_default": self.parameters.get("time_step", 1.0)},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
        }

    def update(self, states, interval=None):
        bulk_ids = states["bulk"]["id"]

        # Resolve indices once
        if self._ppgpp_idx is None:
            self._ppgpp_idx = bulk_name_to_idx(self.ppgpp_id, bulk_ids)
        if self._rpos_idx is None and self.rpos_target > 0:
            try:
                self._rpos_idx = bulk_name_to_idx(self.rpos_id, bulk_ids)
            except Exception:
                self._rpos_idx = None

        updates = []

        # Maintain ppGpp
        curr_ppgpp = int(counts(states["bulk"], self._ppgpp_idx))
        delta_ppgpp = self.ppgpp_target - curr_ppgpp
        if delta_ppgpp != 0:
            updates.append((self._ppgpp_idx, delta_ppgpp))

        # Maintain RpoS (only if target > 0 and molecule exists)
        if self.rpos_target > 0 and self._rpos_idx is not None:
            curr_rpos = int(counts(states["bulk"], self._rpos_idx))
            # Only replenish if below target (don't cap natural accumulation)
            if curr_rpos < self.rpos_target:
                updates.append((self._rpos_idx, self.rpos_target - curr_rpos))

        if not updates:
            return {}
        return {"bulk": updates}
