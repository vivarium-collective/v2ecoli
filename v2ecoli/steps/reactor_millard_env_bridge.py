"""ReactorMillardEnvBridge — feed reactor boundary concentrations to the Millard cell.

Closes the reactor->Millard dissolved-O2 feedback loop (#225 item #4).

The bioreactor coupler (:class:`ReactorCellCoupler`) writes the reactor's
dissolved O2/CO2 into the TOP-LEVEL ``environment.external_concentrations`` store
(mM, keyed by v2ecoli molecule names like ``OXYGEN-MOLECULE[p]``). The WCM arm
consumes that via :class:`EnvironmentMirror` -> each agent's
``boundary.external``. But the Millard kinetic cell's metabolism Process
(:class:`MillardPDMPMetabolism`) reads its own per-agent
``environment.external_concentrations`` store instead — which nothing else
populates — so the reactor's O2 never reached the Millard ODE and CYTBO
respiration ran unthrottled at the model's fixed default.

This Step is the Millard analogue of :class:`EnvironmentMirror`: each tick it
copies the top-level reactor-derived boundary concentrations into every agent's
``environment.external_concentrations`` (overwrite). The Millard step then
aliases those v2ecoli names to its SBML ids (see
``millard_pdmp_metabolism.EXTERNAL_NAME_TO_SBML``) and overwrites the matching
boundary species' value before integrating, so the CYTBO rate law (linear in
[O2]) throttles as the reactor's dissolved O2 falls.

Ordering: wired right after ``environment_mirror`` (early in the flow), so it
reads the previous tick's coupler write and exposes it to the Millard step before
``millard-pdmp-metabolism`` runs. The reactor<->cell O2 loop therefore carries a
one-tick lag — harmless for a slow reactor-scale negative feedback.
"""
from __future__ import annotations

import math
from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.steps.millard_pdmp_metabolism import EXTERNAL_NAME_TO_SBML
from v2ecoli.types.stores import InPlaceDict


class ReactorMillardEnvBridge(Step):
    """Mirror top-level environment.external_concentrations into each agent's."""

    name = "reactor_millard_env_bridge"
    config_schema: dict[str, Any] = {}
    topology = {
        "environment": ("environment",),
        "agents":      ("agents",),
    }

    def initialize(self, config: dict | None = None) -> None:
        pass

    def inputs(self) -> dict[str, Any]:
        return {"environment": InPlaceDict(), "agents": InPlaceDict()}

    def outputs(self) -> dict[str, Any]:
        return {"agents": InPlaceDict()}

    def next_update(self, timestep, states):
        env = states.get("environment") or {}
        external = env.get("external_concentrations") or {}
        if not external:
            return {}
        agents = states.get("agents") or {}
        if not agents:
            return {}

        # Keep only boundaries the Millard model recognises (so the per-agent
        # external_concentrations store stays small and SBML-resolvable), keyed
        # by the v2ecoli name the Millard step already aliases. Values are
        # absolute mM (overwrite); the Millard reader clamps negatives.
        passthrough: dict[str, float] = {}
        for name, conc in external.items():
            if name not in EXTERNAL_NAME_TO_SBML:
                continue
            val = float(conc.magnitude) if hasattr(conc, "magnitude") else float(conc)
            if not math.isfinite(val):
                continue
            passthrough[name] = val
        if not passthrough:
            return {}

        agent_updates = {
            aid: {"environment": {"external_concentrations": dict(passthrough)}}
            for aid in agents
        }
        return {"agents": agent_updates}

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)
