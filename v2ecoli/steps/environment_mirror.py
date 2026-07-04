"""EnvironmentMirror — propagate top-level driver concentrations to per-agent boundary.

Closes the two architectural gaps documented in
`studies/mbp-01-time-varying-environment/study.yaml` open_questions
(`env-store-topology-mismatch` + `env-driver-molecule-id-convention`)
that block the 5 currently-skipped mbp-01 plumbing/extreme tests.

Background
----------
`baseline_time_varying_env` adds a TOP-LEVEL `environment` store that
`EnvironmentDriver` (also top-level) writes to. But `MediaUpdate` lives
inside each cell (`agents.<id>`) and only reacts to `media_id` transitions
— it doesn't (and shouldn't) consume `external_concentrations` directly,
because the per-agent `environment` store has no typed slot for it and
adding one fights PBG's schema-inference path.

This Step bridges the gap by writing the driver's per-tick values directly
into each agent's `boundary.external` store as a delta — exactly the form
`MediaUpdate` already produces on media-ID transitions. The agent's
`boundary.external` already has a fully-typed schema (set up by
`sim_data` initialization with mM-quantity entries for every molecule
metabolism imports), so the writes apply cleanly without per-agent
pre-seed gymnastics.

Convention
----------
Driver / coupler writes use BARE molecule names (`GLC`, `ACET`, `FUM`,
...) matching `boundary.external` keys. Any unmatched name is silently
skipped — failing closed rather than crashing if a future trajectory
spec includes a molecule metabolism doesn't track.

Ordering
--------
Wired into `baseline_time_varying_env` BEFORE the FLUSH barrier that
precedes `media_update`'s layer, so the mirror's writes commit (via the
FLUSH) before `exchange_data` reads `boundary.external` and re-derives
metabolism's exchange constraints. End-to-end: driver writes top-level
env → mirror writes per-agent boundary delta → FLUSH → exchange_data
reads updated boundary → metabolism sees new constraint within one tick.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


class EnvironmentMirror(Step):
    """Propagate top-level environment.external_concentrations to each agent's boundary.external."""

    name = "environment_mirror"
    config_schema = {
        "time_step": "float",
    }
    topology = {
        "environment": ("environment",),
        "agents":      ("agents",),
    }

    def initialize(self, config: dict | None = None) -> None:
        # No persistent state; per-tick read + delta-write.
        pass

    def inputs(self) -> dict[str, Any]:
        return {"environment": InPlaceDict(), "agents": InPlaceDict()}

    def outputs(self) -> dict[str, Any]:
        return {"agents": InPlaceDict()}

    def next_update(self, timestep, states):
        env = states.get("environment") or {}
        external = env.get("external_concentrations") or {}
        if not external:
            # No driver writes this tick — keep baseline path byte-identical.
            return {}

        agents = states.get("agents") or {}
        if not agents:
            return {}

        # For each agent: compute boundary.external delta (driver_conc - current).
        # PBG strips pint units at store boundaries; both inputs arrive as bare
        # floats with the same implicit mM convention. Strip residual pint
        # quantities defensively and work in raw float space.
        agent_updates: dict[str, Any] = {}
        for agent_id, agent_state in agents.items():
            boundary = (agent_state or {}).get("boundary") or {}
            boundary_ext = boundary.get("external") or {}
            conc_update: dict[str, float] = {}
            for mol, conc_raw in external.items():
                curr_raw = boundary_ext.get(mol)
                if curr_raw is None:
                    # metabolism doesn't track this molecule; fail closed.
                    continue
                conc = float(conc_raw.magnitude) if hasattr(conc_raw, "magnitude") else float(conc_raw)
                curr = float(curr_raw.magnitude) if hasattr(curr_raw, "magnitude") else float(curr_raw)
                diff = conc - curr
                if np.isnan(diff):
                    diff = 0.0
                conc_update[mol] = diff
            if conc_update:
                agent_updates[agent_id] = {"boundary": {"external": conc_update}}

        if not agent_updates:
            return {}
        return {"agents": agent_updates}

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)
