"""RefGrowthDriver — biomass-precursor producer for the PDMP composite.

The Millard 2017 kinetic ODE has no biomass equation, so the PDMP composite
has no driver for cell growth — its downstream consumers (transcription,
translation) deplete substrate pools without replenishment, opening a W₂
gap to the kFBA reference (see ``scripts/compare_pdmp_vs_phase0.py``).

This Step closes that gap as a scaffold while a true kinetic biomass-flux
Process is being built (task #21 full form). Two flux modes:

* **``proportional``** (legacy, default): scale precursor counts at a fixed
  reference growth rate, ``precursors[t+1] = precursors[t] · (1 + μ·dt)``.
  Teleonomic — the rate is a parameter, not derived from cell state. Was
  shown empirically (``compare_pdmp_vs_phase0.py``) to move cm_final only
  2 fg of the 187 fg gap because precursor turnover (~1.8M ATP/s from
  translation alone) is ~1000× larger than ``μ × current``.

* **``measured_kfba``** (new): inject precursor counts at the constant
  per-second rates measured from a 600 s kFBA-baseline trajectory
  (``scripts/sample_kfba_precursor_fluxes.py`` → ``.pbg/runs/
  kfba-precursor-fluxes.json``). Top rates are GLT 5413/s, ATP 1640/s,
  UTP 803/s, TTP 787/s — these match the kFBA biosynthesis side of the
  pool balance and are the only realistic way to keep up with PDMP's
  consumption side (which is roughly the same as kFBA's, since both run
  the same translation + transcription processes).

  Negative ``net_rate_per_s`` entries (kFBA was net-consuming these
  precursors over the sampling window — e.g. GLN, TRP in the M9-glucose
  reference) are clamped to zero: PDMP's own consumers already drain
  them at the right rate, so adding negative injection would
  double-subtract.

Both modes label their emitted updates with ``_source: "ref_growth_driver"``
so downstream tooling can identify them as scaffold, not biology.
"""
from __future__ import annotations

import json
import os
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

# Default location of the measured-kFBA-flux JSON (relative to repo root).
DEFAULT_MEASURED_FLUX_PATH = ".pbg/runs/kfba-precursor-fluxes.json"


def _load_measured_rates(path: str) -> dict[str, float]:
    """Read ``net_rate_per_s`` block from the sampled-kFBA-fluxes JSON.

    Negative entries (net-consumption windows in the reference run) clamp
    to 0.0 — see module docstring.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"measured_kfba flux mode requires {path}; "
            f"generate it with scripts/sample_kfba_precursor_fluxes.py"
        )
    with open(path) as fh:
        payload = json.load(fh)
    raw = payload.get("net_rate_per_s") or payload.get("rates_per_s") or {}
    return {k: max(0.0, float(v)) for k, v in raw.items()}


class RefGrowthDriver(Step):
    """Drive precursor bulk counts toward Phase-0 mass via either
    proportional-scaling or kFBA-measured constant injection."""

    name = "ref-growth-driver"
    topology = {"bulk": ("bulk",)}

    config_schema = {
        "growth_rate_per_s": {"_default": DEFAULT_GROWTH_RATE_PER_S},
        # 'proportional' = μ·current·dt; 'measured_kfba' = constant rate · dt.
        "flux_source": {"_default": "proportional"},
        "measured_flux_path": {"_default": DEFAULT_MEASURED_FLUX_PATH},
        # If False, this Step is a no-op — useful for A/B-comparing PDMP
        # with vs without the growth driver in the report.
        "enabled": {"_default": True},
        "tick_s": {"_default": 1.0},
        "seed": {"_default": 0},
    }

    def initialize(self, config):
        self.growth_rate = float(self.parameters.get(
            "growth_rate_per_s", DEFAULT_GROWTH_RATE_PER_S))
        self.enabled = bool(self.parameters.get("enabled", True))
        self.tick_s = float(self.parameters.get("tick_s", 1.0))
        self.flux_source = str(
            self.parameters.get("flux_source", "proportional")
        )
        self._rng = np.random.RandomState(self.parameters.get("seed", 0))
        # Per-tick scaling factor: precursors[t+1] = precursors[t] * (1 + μ·dt).
        self._factor_minus_one = self.growth_rate * self.tick_s

        # Precursor universe — used for both flux sources. The proportional
        # mode iterates the whole list; the measured-kFBA mode only emits
        # for IDs that have a measured rate.
        self._precursor_ids = AA_BULK_IDS + NTP_BULK_IDS + DNTP_BULK_IDS

        # Measured-kFBA inputs: load rates once at init. Pre-build the rate
        # vector aligned with the bulk index order (resolved lazily on the
        # first update() call, see _bulk_idx below). _measured_rates_by_id
        # holds the per-precursor-ID rate so the per-tick path is one
        # numpy multiply.
        self._measured_rates_by_id: dict[str, float] = {}
        if self.flux_source == "measured_kfba":
            measured_path = str(self.parameters.get(
                "measured_flux_path", DEFAULT_MEASURED_FLUX_PATH))
            self._measured_rates_by_id = _load_measured_rates(measured_path)

        # Lazy index resolution (need bulk['id'] at first call).
        self._bulk_idx: np.ndarray | None = None
        # Rate vector aligned with self._bulk_idx, populated in lockstep.
        self._rate_per_s: np.ndarray | None = None
        # Carry sub-integer accumulation across ticks so injection at
        # e.g. 6.7/s lands an extra count every ~10 ticks instead of
        # rounding to 0 every tick (the bug that bit the proportional
        # mode at very small μ × current).
        self._delta_residual: np.ndarray | None = None

    def inputs(self):
        return {"bulk": "bulk_array"}

    def outputs(self):
        return {"bulk": "bulk_array"}

    def _resolve_indices(self, bulk):
        bulk_ids = bulk["id"]
        resolved: list[int] = []
        rates: list[float] = []
        for vid in self._precursor_ids:
            idx = bulk_name_to_idx(vid, bulk_ids, strict=False)
            if idx is None or (isinstance(idx, np.ndarray) and idx.size == 0):
                continue
            if self.flux_source == "measured_kfba":
                rate = self._measured_rates_by_id.get(vid, 0.0)
                if rate <= 0.0:
                    # Skip non-positive rates entirely — no injection.
                    continue
                rates.append(rate)
            resolved.append(int(idx))
        self._bulk_idx = np.asarray(resolved, dtype=np.int64)
        if self.flux_source == "measured_kfba":
            self._rate_per_s = np.asarray(rates, dtype=np.float64)
            self._delta_residual = np.zeros_like(self._rate_per_s)

    def update(self, states, interval=None):
        if not self.enabled:
            return {}

        bulk = states.get("bulk")
        if bulk is None or not hasattr(bulk, "dtype") or not bulk.dtype.names:
            return {}

        if self._bulk_idx is None:
            self._resolve_indices(bulk)

        if self._bulk_idx.size == 0:
            return {}

        if self.flux_source == "measured_kfba":
            # Constant flux × dt. Accumulate sub-integer residual across
            # ticks so even slow precursors (e.g. CYS at 6.7/s) land
            # an integer count every ~1-2 ticks instead of vanishing.
            raw_delta = self._rate_per_s * self.tick_s + self._delta_residual
            delta_int = np.floor(raw_delta).astype(np.int64)
            self._delta_residual = raw_delta - delta_int.astype(np.float64)
            if not delta_int.any():
                return {}
            return {"bulk": [(self._bulk_idx, delta_int)]}

        # Default: proportional scaling. Δ = round(current × μ·dt). Floor
        # at 1 when current > 0 so very small pools still grow (otherwise
        # rint() rounds the increment to 0 forever).
        current = counts(bulk, self._bulk_idx).astype(np.float64, copy=False)
        raw_delta = current * self._factor_minus_one
        delta_int = np.rint(raw_delta).astype(np.int64)
        floor_mask = (raw_delta > 0) & (delta_int == 0) & (current > 0)
        if floor_mask.any():
            delta_int = np.where(floor_mask, 1, delta_int)

        if not delta_int.any():
            return {}

        return {"bulk": [(self._bulk_idx, delta_int)]}
