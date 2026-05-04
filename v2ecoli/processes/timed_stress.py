"""
Timed Stress Onset — Applies stress mid-simulation
====================================================

A Step process that reads global_time and activates a stress condition
after a configurable onset time. Supports two stress modes:

  1. H₂O₂ challenge: sets external H₂O₂ production rate in the
     oxidative stress listener (read by OxidativeStress process)
  2. ppGpp starvation: clamps ppGpp at a target level (like
     PpGppSustainedStress but only after onset)

The stress can also be turned off after a configurable duration
(pulse mode) to study recovery kinetics.

Usage via feature_configs:
    feature_configs={
        'ecoli-timed-stress': {
            'stress_type': 'h2o2',       # or 'starvation'
            'onset_time': 600,           # seconds (10 min)
            'duration': 0,              # 0 = permanent, >0 = pulse
            'h2o2_rate_uM_per_s': 100,  # for h2o2 mode
            'ppgpp_target': 250000,     # for starvation mode
        }
    }

References
----------
Seaver LC, Imlay JA (2001) J Bacteriol 183:7182-7189
Traxler MF et al. (2008) Mol Microbiol 68:1128-1148
"""

import numpy as np

from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "ecoli-timed-stress"
TOPOLOGY = {
    "bulk":        ("bulk",),
    "global_time": ("global_time",),
    "listeners":   ("listeners",),
    "timestep":    ("timestep",),
}


class TimedStress(Step):
    """Applies stress at a configurable time during the simulation."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "stress_type":         {"_type": "string",  "_default": "h2o2"},
        "onset_time":          {"_type": "float",   "_default": 600.0},
        "duration":            {"_type": "float",   "_default": 0.0},
        "h2o2_rate_uM_per_s": {"_type": "float",   "_default": 100.0},
        "ppgpp_target":        {"_type": "integer", "_default": 250000},
        "ppgpp_id":            {"_type": "string",  "_default": "GUANOSINE-5DP-3DP[c]"},
        "rpos_target":         {"_type": "integer", "_default": 2000},
        "rpos_id":             {"_type": "string",  "_default": "RPOS-MONOMER[c]"},
        "time_step":           {"_type": "float",   "_default": 1.0},
    }

    def initialize(self, config):
        p = self.parameters
        self.stress_type = str(p.get("stress_type", "h2o2"))
        self.onset_time  = float(p.get("onset_time", 600.0))
        self.duration    = float(p.get("duration", 0.0))
        self.h2o2_rate   = float(p.get("h2o2_rate_uM_per_s", 100.0))
        self.ppgpp_target = int(p.get("ppgpp_target", 250000))
        self.ppgpp_id    = str(p.get("ppgpp_id", "GUANOSINE-5DP-3DP[c]"))
        self.rpos_target = int(p.get("rpos_target", 2000))
        self.rpos_id     = str(p.get("rpos_id", "RPOS-MONOMER[c]"))
        self._ppgpp_idx  = None
        self._rpos_idx   = None
        self._stress_active = False

    def inputs(self):
        return {
            "bulk":        {"_type": "bulk_array", "_default": []},
            "global_time": {"_type": "float", "_default": 0.0},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "float", "_default": 0.0},
                },
            },
            "timestep":    {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "timed_stress": {
                    "active":           {"_type": "overwrite[float]", "_default": 0.0},
                    "onset_time":       {"_type": "overwrite[float]", "_default": 0.0},
                    "stress_type":      {"_type": "overwrite[float]", "_default": 0.0},
                    "h2o2_signal_uM_s": {"_type": "overwrite[float]", "_default": 0.0},
                    "ppgpp_signal":     {"_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def _is_active(self, t):
        """Check if stress should be active at time t."""
        if t < self.onset_time:
            return False
        if self.duration > 0 and t > self.onset_time + self.duration:
            return False
        return True

    def update(self, states, interval=None):
        t = float(states.get("global_time", 0.0))
        active = self._is_active(t)
        self._stress_active = active

        bulk_updates = []
        h2o2_signal = 0.0
        ppgpp_signal = 0.0

        if active and self.stress_type == "starvation":
            # Clamp ppGpp at target level
            if self._ppgpp_idx is None:
                self._ppgpp_idx = bulk_name_to_idx(
                    self.ppgpp_id, states["bulk"]["id"])
            curr = int(counts(states["bulk"], self._ppgpp_idx))
            delta = self.ppgpp_target - curr
            if delta != 0:
                bulk_updates.append((self._ppgpp_idx, delta))
            ppgpp_signal = float(self.ppgpp_target)

            # Stabilize RpoS (ppGpp inhibits ClpXP/RssB degradation)
            # Zgurskaya et al. 1997: RpoS half-life 2 min → 30 min
            if self.rpos_target > 0:
                if self._rpos_idx is None:
                    try:
                        self._rpos_idx = bulk_name_to_idx(
                            self.rpos_id, states["bulk"]["id"])
                    except Exception:
                        self._rpos_idx = None
                if self._rpos_idx is not None:
                    curr_rpos = int(counts(states["bulk"], self._rpos_idx))
                    if curr_rpos < self.rpos_target:
                        bulk_updates.append(
                            (self._rpos_idx, self.rpos_target - curr_rpos))

        if active and self.stress_type == "h2o2":
            h2o2_signal = self.h2o2_rate

        result = {
            "listeners": {
                "timed_stress": {
                    "active": 1.0 if active else 0.0,
                    "onset_time": self.onset_time,
                    "stress_type": 1.0 if self.stress_type == "h2o2" else 2.0,
                    "h2o2_signal_uM_s": h2o2_signal,
                    "ppgpp_signal": ppgpp_signal,
                },
            },
        }
        if bulk_updates:
            result["bulk"] = bulk_updates
        return result
