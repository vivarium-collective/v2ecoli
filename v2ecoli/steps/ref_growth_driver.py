"""RefGrowthDriver — biomass-precursor producer for the PDMP composite.

The Millard 2017 kinetic ODE has no biomass equation, so the PDMP composite
has no driver for cell growth — its downstream consumers (transcription,
translation) deplete substrate pools without replenishment, opening a W₂
gap to the kFBA reference (see ``scripts/compare_pdmp_vs_phase0.py``).

Three flux modes:

* **``proportional``** (legacy, default): scale precursor counts at a fixed
  reference growth rate, ``precursors[t+1] = precursors[t] · (1 + μ·dt)``.
  Teleonomic; moves cm_final only ~2 fg of the 187 fg gap because precursor
  turnover (~1.8M ATP/s from translation alone) is ~1000× larger than
  ``μ × current``.

* **``measured_kfba``**: inject precursor counts at constant per-second
  rates measured from a 600 s kFBA-baseline trajectory
  (``scripts/sample_kfba_precursor_fluxes.py`` → ``.pbg/runs/
  kfba-precursor-fluxes.json``). Negative net rates clamp to zero. This
  matches the **net** kFBA pool growth but UNDERSHOOTS biosynthesis
  because kFBA's net = biosynthesis − consumption, and PDMP's consumption
  is comparable to kFBA's (same translation/transcription processes).
  Empirically: ATP drains to zero by ~t=300 s, translation stalls.

* **``consumption_matched``** (default for closing the W₂ gap): per tick,
  observe the actual delta in each precursor pool and infer
  ``other_processes_delta = bulk_delta − my_previous_injection``.
  Next-tick injection = kfba_net_rate + max(0, -other_processes_delta) —
  enough to fully compensate for whatever the other processes consumed,
  plus the small kFBA-net growth on top. This is a one-step feedback
  controller that tracks the actual per-tick consumption regardless of
  whether translation is ATP-starved or running flat-out.

All modes label their emitted updates with ``_source: "ref_growth_driver"``
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

# Byproducts that translation + tRNA-charging produce as side effects
# (every NTP → NMP + PPi during peptide-bond formation, etc.). In Phase-0
# these are recycled by metabolism (adenylate kinase, pyrophosphatase,
# OxPhos); in PDMP nothing consumes them, so they accumulate as bulk
# mass. The ``consumption_matched`` mode targets a per-tick net delta of
# zero for these (i.e. tracks consumption AND production), letting the
# feedback controller emit *negative* injection to drain over-production.
BYPRODUCT_BULK_IDS = ("AMP[c]", "PPI[c]", "ADP[c]")

# Water tracking — the kFBA reference grows cell water by ~138 fg over 600 s
# of M9-glucose (cell_mass grows 1274.5 → 1461.85 fg, dry_mass grows 388.93
# → 438.84 fg, balance is water). At 18 Da per water that's ~7.7×10⁶ H2O
# molecules per second of net production. The driver injects water at this
# rate via the same consumption_matched feedback as the AAs / NTPs, so that
# cell_mass tracks dry_mass; without water injection cell_mass undershoots
# by exactly the missing growth.
#
# The rate now lives in ``.pbg/runs/kfba-precursor-fluxes.json`` next to
# the AAs/NTPs/dNTPs — sampled by ``scripts/sample_kfba_precursor_fluxes.py``
# from a live baseline trajectory rather than hardcoded. ``WATER_RATE_PER_S``
# below stays as a fallback when the JSON doesn't include ``WATER[c]`` (back-
# compat with pre-water sampler runs).
WATER_BULK_IDS = ("WATER[c]",)
WATER_RATE_PER_S = 7.7e6

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

        # Precursor universe — used for all flux sources.
        #
        # Proportional and measured_kfba modes only add positive injections;
        # they track precursors only. consumption_matched also tracks the
        # translation byproducts (AMP, PPi, ADP) at zero target rate so the
        # feedback can emit *negative* injection to drain them, mimicking
        # the recycling that kFBA's metabolism does in Phase-0.
        if self.flux_source == "consumption_matched":
            self._precursor_ids = (
                AA_BULK_IDS + NTP_BULK_IDS + DNTP_BULK_IDS
                + BYPRODUCT_BULK_IDS + WATER_BULK_IDS
            )
        else:
            self._precursor_ids = AA_BULK_IDS + NTP_BULK_IDS + DNTP_BULK_IDS

        # Measured-kFBA inputs: load rates once at init.
        self._measured_rates_by_id: dict[str, float] = {}
        if self.flux_source in ("measured_kfba", "consumption_matched"):
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
        # consumption_matched mode bookkeeping: pool-count snapshot from
        # the previous tick + last injection (so we can infer the
        # consumption rate from other processes).
        self._prev_counts: np.ndarray | None = None
        self._prev_injection: np.ndarray | None = None
        self._first_tick_done: bool = False

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
            if self.flux_source in ("measured_kfba", "consumption_matched"):
                rate = self._measured_rates_by_id.get(vid, 0.0)
                # Water — prefer the rate from the kFBA fluxes JSON (the
                # sampler in scripts/sample_kfba_precursor_fluxes.py now
                # walks WATER[c] alongside the AAs / NTPs). Fall back to
                # the in-source ``WATER_RATE_PER_S`` constant when the JSON
                # predates the water column (back-compat with pre-water
                # sampler runs).
                if vid in WATER_BULK_IDS and rate <= 0.0:
                    rate = WATER_RATE_PER_S
                if self.flux_source == "measured_kfba" and rate <= 0.0:
                    # measured_kfba mode skips non-positive rates entirely.
                    continue
                # consumption_matched mode keeps every precursor (even
                # zero-rate ones) so we can still match consumption on
                # them.
                rates.append(max(0.0, rate))
            resolved.append(int(idx))
        self._bulk_idx = np.asarray(resolved, dtype=np.int64)
        if self.flux_source in ("measured_kfba", "consumption_matched"):
            self._rate_per_s = np.asarray(rates, dtype=np.float64)
            self._delta_residual = np.zeros_like(self._rate_per_s)
            self._prev_counts = np.zeros_like(self._rate_per_s)
            self._prev_injection = np.zeros_like(self._rate_per_s)

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

        if self.flux_source == "consumption_matched":
            # One-step feedback controller — tracks what the OTHER processes
            # did to each pool during the previous tick, and compensates.
            #
            # Between tick t (where we read self._prev_counts) and tick t+1
            # (where we read `current`):
            #   current = self._prev_counts + self._prev_injection + other_delta
            #     where other_delta = sum of all OTHER per-tick deltas applied
            #     to these bulk indices between the two reads.
            #   => other_delta = current - self._prev_counts - self._prev_injection
            #
            # We want THIS tick's emitted change (my_injection + concurrent
            # other_delta) to track target_delta = kfba_net_rate · tick_s.
            # Assuming the per-tick consumption pattern is roughly steady-
            # state (other processes don't change abruptly):
            #   my_injection = target_delta - other_delta
            #
            # For precursors (target_delta > 0): floor at 0 — never drive
            #   pools negative.
            # For byproducts (target_delta == 0): allow negative — that's
            #   the whole point, drain pool of translation byproducts
            #   (AMP / PPi / ADP) that no PDMP process recycles. Clip the
            #   removal to current pool size so we never overdraw.
            current = counts(bulk, self._bulk_idx).astype(np.float64, copy=False)
            if self._first_tick_done:
                other_delta = (
                    current - self._prev_counts
                    - self._prev_injection.astype(np.float64)
                )
                target_delta = self._rate_per_s * self.tick_s
                desired_injection = target_delta - other_delta
                # Precursors keep their non-negative floor; byproducts
                # (rate==0) are allowed to go negative down to -current
                # (drain to zero, never below).
                is_byproduct = self._rate_per_s == 0.0
                desired_injection = np.where(
                    is_byproduct,
                    np.maximum(desired_injection, -current),
                    np.maximum(desired_injection, 0.0),
                )
                raw_delta = desired_injection + self._delta_residual
            else:
                # No history yet — fall back to constant-rate seed for
                # the first tick (byproducts get 0, which is harmless).
                raw_delta = self._rate_per_s * self.tick_s + self._delta_residual

            # Floor toward zero rather than negative infinity so the
            # residual carries the correct sign for the next tick.
            delta_int = np.trunc(raw_delta).astype(np.int64)
            self._delta_residual = raw_delta - delta_int.astype(np.float64)
            # For next tick, remember the bulk reading we just observed
            # (pre-injection from this tick's perspective) plus this tick's
            # injection so we can recover other_delta next time.
            self._prev_counts = current
            self._prev_injection = delta_int
            self._first_tick_done = True
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
