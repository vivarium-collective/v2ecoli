"""RefGrowthDriver — teleonomic biomass-precursor producer for the PDMP composite.

This Step is a deliberately-simple stub that closes the W₂ gap exposed by
`compare_pdmp_vs_phase0.py`: the Millard 2017 kinetic ODE has no biomass
equation, so the PDMP composite has no driver for cell growth, and the
WCM's downstream consumers (transcription, translation) deplete their
substrate pools without replenishment.

The proper fix is task #21 in its full form: wire a kinetic biomass-flux
Process that consumes Millard's central metabolites at a rate set either
(a) by an outer LQR control loop, or (b) by mass-action kinetics on the
biosynthetic pathways outside Millard's scope. Both are real biology
projects.

This stub instead drives growth at a Phase-0 reference rate — measured
0.88/h across N=8 Phase-0 trajectories — by scaling the bulk counts of
amino acids, NTPs, and dNTPs at each tick. It is teleonomic by design:
the growth rate is a fixed parameter, not derived from cell state. That's
the WRONG abstraction for inference and causal-discovery work (the
investigation's eventual goal), but it's a sufficient scaffold for the
pdmp-01 acceptance gate viz and demonstrates the missing infrastructure.

Loud labelling: every emitted update is tagged with `_source: "ref_growth_driver"`
in the listener output so downstream tooling can see this is a teleonomic
patch, not biology.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


# Phase-0 reference growth rate: average across 8 M9-glucose trajectories
# (see scripts/compare_pdmp_vs_phase0.py loaders), ~0.88/h ≈ 2.44e-4 /s.
# Doubling time T₂ = ln(2)/μ ≈ 47 min, matching v2ecoli's standard cycle.
DEFAULT_GROWTH_RATE_PER_S = 2.44e-4

# Canonical 21-amino-acid list (matches sim_data.configs.ecoli-metabolism.aa_names).
AA_BULK_IDS = (
    "L-ALPHA-ALANINE[c]", "ARG[c]", "ASN[c]", "L-ASPARTATE[c]", "CYS[c]",
    "GLT[c]", "GLN[c]", "GLY[c]", "HIS[c]", "ILE[c]", "LEU[c]", "LYS[c]",
    "MET[c]", "PHE[c]", "PRO[c]", "SER[c]", "THR[c]", "TRP[c]", "TYR[c]",
    "L-SELENOCYSTEINE[c]", "VAL[c]",
)
NTP_BULK_IDS = ("ATP[c]", "GTP[c]", "CTP[c]", "UTP[c]")
DNTP_BULK_IDS = ("DATP[c]", "DGTP[c]", "DCTP[c]", "TTP[c]")


class RefGrowthDriver(Step):
    """Scale precursor bulk counts at a Phase-0 reference growth rate."""

    name = "ref-growth-driver"
    topology = {"bulk": ("bulk",)}

    config_schema = {
        "growth_rate_per_s": {"_default": DEFAULT_GROWTH_RATE_PER_S},
        # If False, this Step is a no-op — useful for A/B-comparing PDMP
        # with vs without the teleonomic growth driver in the report.
        "enabled": {"_default": True},
        "tick_s": {"_default": 1.0},
        "seed": {"_default": 0},
    }

    def initialize(self, config):
        self.growth_rate = float(self.parameters.get(
            "growth_rate_per_s", DEFAULT_GROWTH_RATE_PER_S))
        self.enabled = bool(self.parameters.get("enabled", True))
        self.tick_s = float(self.parameters.get("tick_s", 1.0))
        self._rng = np.random.RandomState(self.parameters.get("seed", 0))
        # Per-tick scaling factor: precursors[t+1] = precursors[t] * (1 + μ·dt).
        self._factor_minus_one = self.growth_rate * self.tick_s
        # Lazy index resolution (need bulk['id'] at first call).
        self._precursor_ids = AA_BULK_IDS + NTP_BULK_IDS + DNTP_BULK_IDS
        self._bulk_idx: np.ndarray | None = None

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def update(self, states, interval=None):
        if not self.enabled:
            return {}

        bulk = states.get("bulk")
        if bulk is None or not hasattr(bulk, "dtype") or not bulk.dtype.names:
            return {}

        if self._bulk_idx is None:
            bulk_ids = bulk["id"]
            resolved: list[int] = []
            for vid in self._precursor_ids:
                idx = bulk_name_to_idx(vid, bulk_ids, strict=False)
                if idx is None or (isinstance(idx, np.ndarray) and idx.size == 0):
                    continue
                resolved.append(int(idx))
            self._bulk_idx = np.asarray(resolved, dtype=np.int64)

        if self._bulk_idx.size == 0:
            return {}

        current = counts(bulk, self._bulk_idx).astype(np.float64, copy=False)
        # Δ = round(current × μ·dt). Floor at 1 when current > 0 so very
        # small pools still grow (otherwise rint() rounds the increment
        # to 0 forever and the pool stays stuck).
        raw_delta = current * self._factor_minus_one
        delta_int = np.rint(raw_delta).astype(np.int64)
        floor_mask = (raw_delta > 0) & (delta_int == 0) & (current > 0)
        if floor_mask.any():
            delta_int = np.where(floor_mask, 1, delta_int)

        if not delta_int.any():
            return {}

        return {"bulk": [(self._bulk_idx, delta_int)]}
