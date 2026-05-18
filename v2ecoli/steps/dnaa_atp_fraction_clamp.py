"""
=========================
DnaA-ATP Fraction Clamp
=========================

Stand-in for the deferred dnaa-05 extrinsic conversion network
(RIDA / DDAH / DARS). When enabled, this Step redistributes
DnaA-ATP and DnaA-ADP counts each timestep to keep

    DnaA-ATP / (DnaA-apo + DnaA-ATP + DnaA-ADP)

inside a user-specified band [low, high]. Without this clamp the
intrinsic hydrolysis Step alone is too slow to balance the
ATP-binding equilibrium (Boesen 2024 PNAS: in-vivo target band is
[0.2, 0.5]) — see ``dnaa-02-EQ-01`` in the study yaml.

Conservation
------------
The clamp only TRANSFERS molecules between DnaA-ATP (MONOMER0-160[c])
and DnaA-ADP (MONOMER0-4565[c]). It does NOT change the total DnaA
pool, and it does NOT touch free-monomer counts (PD03831[c]) or the
free ATP/ADP pools. Mass is conserved up to a Pi/H2O difference
(~100 Da per transfer) which is folded into metabolism's noise.

Direction
---------
- atp_fraction > high → convert ATP→ADP (extra hydrolysis)
- atp_fraction < low  → convert ADP→ATP (extra reactivation)

Defaults
--------
- ``band``: ``None`` (clamp disabled by default; opt-in)
- when set: ``[0.2, 0.5]`` matches Boesen 2024 PNAS Delta4 cells

Scope note
----------
This is the dnaa-02 placeholder for the dnaa-05 work. Removed when
the full RIDA / DDAH / DARS / DARS2 network lands.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-atp-fraction-clamp"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
}

DNAA_APO_ID = "PD03831[c]"
DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"


class DnaaAtpFractionClamp(Step):
    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # ``band`` is a [low, high] pair; None disables the clamp.
        "band": {"_type": "any", "_default": None},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        band = self.parameters.get("band", None)
        self.low: float | None = None
        self.high: float | None = None
        if band is not None:
            try:
                lo, hi = float(band[0]), float(band[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    f"DnaaAtpFractionClamp.band must be [low, high] or None, "
                    f"got {band!r}"
                )
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"DnaaAtpFractionClamp.band requires 0 <= low < high <= 1, "
                    f"got [{lo}, {hi}]"
                )
            self.low, self.high = lo, hi
        self._idx_apo: int | None = None
        self._idx_atp: int | None = None
        self._idx_adp: int | None = None

    @property
    def enabled(self) -> bool:
        return self.low is not None

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "dnaA_cycle": {
                    "clamp_transfer":   {"_type": "overwrite[integer]", "_default": 0},
                    "clamp_direction":  {"_type": "overwrite[string]",  "_default": "off"},
                },
            },
        }

    def update(self, states, interval=None):
        if not self.enabled:
            return {
                "listeners": {
                    "dnaA_cycle": {"clamp_transfer": 0, "clamp_direction": "off"}
                }
            }
        if self._idx_apo is None:
            self._idx_apo = int(bulk_name_to_idx(DNAA_APO_ID, states["bulk"]["id"]))
            self._idx_atp = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._idx_adp = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        apo = int(counts(states["bulk"], self._idx_apo))
        atp = int(counts(states["bulk"], self._idx_atp))
        adp = int(counts(states["bulk"], self._idx_adp))
        total = apo + atp + adp
        if total <= 0:
            return {
                "listeners": {
                    "dnaA_cycle": {"clamp_transfer": 0, "clamp_direction": "noop"}
                }
            }

        atp_frac = atp / total

        # No-op if already inside the band
        if self.low <= atp_frac <= self.high:
            return {
                "listeners": {
                    "dnaA_cycle": {"clamp_transfer": 0, "clamp_direction": "in_band"}
                }
            }

        if atp_frac > self.high:
            # Too much ATP — convert ATP -> ADP
            target_atp = int(self.high * total)  # land at the boundary
            transfer = max(1, atp - target_atp)
            transfer = min(transfer, atp)
            direction = "atp_to_adp"
            idx = np.array([self._idx_atp, self._idx_adp], dtype=int)
            delta = np.array([-transfer, transfer], dtype=int)
        else:
            # Too little ATP — convert ADP -> ATP
            target_atp = int(self.low * total)
            transfer = max(1, target_atp - atp)
            transfer = min(transfer, adp)
            direction = "adp_to_atp"
            idx = np.array([self._idx_atp, self._idx_adp], dtype=int)
            delta = np.array([transfer, -transfer], dtype=int)

        return {
            "bulk": [(idx, delta)],
            "listeners": {
                "dnaA_cycle": {
                    "clamp_transfer": int(transfer),
                    "clamp_direction": direction,
                },
            },
        }
