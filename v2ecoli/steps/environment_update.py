"""
Environment update — close the depletion loop.

Reads the per-step exchange counts that ``metabolism.py`` writes to
``environment.exchange`` (negative for import / consumption, positive for
secretion), converts them to a concentration change given a configurable
environment volume, and emits the delta into ``boundary.external``. The
resulting ``Δ[M]_ext`` is summed onto the store by bigraph-schema's
default additive ``apply`` — the same accumulator convention
``media_update.py`` already uses for media changes.

Without this step ``boundary.external`` is static: metabolism imports
glucose but the external pool never drops, so Michaelis-Menten glucose
uptake in ``metabolic_kinetics.py`` can't slow down and the cell never
enters stationary phase. With it, uptake → [GLC_ext] decay → MM bound
drop → growth-rate falloff emerges mechanically.

Placement: runs immediately after ``ecoli-metabolism`` so the boundary
values MK reads next cycle reflect this cycle's uptake.
"""

import os

import numpy as np

from v2ecoli.steps.base import V2Step as Step


def _nutrient_growth_on() -> bool:
    return os.environ.get("V2ECOLI_NUTRIENT_GROWTH", "0") == "1"


# Avogadro's number; mmol→count conversion.
AVOGADRO = 6.02214076e23


class EnvironmentUpdate(Step):
    """Subtract cell exchange flux from boundary.external concentrations."""

    name = "environment_update"

    config_schema = {
        "time_step": {"_default": 1},
        # Environment volume per cell (litres). 1e-13 L = 100 fL ≈ late-log
        # (OD ~1) cell density. Override for dilute cultures or colony sims.
        "env_volume_L": {"_type": "float", "_default": 1e-13},
    }

    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
    }

    def initialize(self, config):
        self.parameters = config or {}
        self.env_volume_L = float(self.parameters.get("env_volume_L", 1e-13))
        # Precompute the count → mM conversion factor:
        #   Δ[M]_mM = count / (N_A · V_L) · 1000
        self._count_to_mM = 1000.0 / (AVOGADRO * self.env_volume_L)

    def inputs(self):
        from v2ecoli.types.stores import InPlaceDict
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def outputs(self):
        from v2ecoli.types.stores import InPlaceDict
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def next_update(self, timestep, states):
        # When the nutrient-growth feature set is OFF, this step is a
        # no-op and boundary.external stays at media concentrations —
        # matches vEcoli 1.0 behaviour.
        if not _nutrient_growth_on():
            return {}
        exchange = states.get("environment", {}).get("exchange", {}) or {}
        if not exchange:
            return {}

        # boundary.external is an InPlaceDict: leaf writes replace, they
        # don't accumulate. So we read the current concentration, apply
        # the delta, and emit the new absolute value. Clamp at zero — a
        # negative concentration is nonphysical and upstream FBA warns
        # about it.
        current = states.get("boundary", {}).get("external", {}) or {}
        new_values: dict[str, float] = {}
        for mol, count in exchange.items():
            try:
                c = float(count)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(c) or c == 0:
                continue
            prev = current.get(mol, 0.0)
            try:
                prev_f = float(prev.asNumber()) if hasattr(prev, "asNumber") else float(prev)
            except (TypeError, ValueError):
                prev_f = 0.0
            new_values[mol] = max(0.0, prev_f + c * self._count_to_mM)

        if not new_values:
            return {}
        return {"boundary": {"external": new_values}}

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)
