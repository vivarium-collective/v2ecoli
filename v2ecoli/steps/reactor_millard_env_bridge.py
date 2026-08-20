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
``environment.external_concentrations`` (overwrite) -- **except** for the
species listed in ``_NOT_REACTOR_DRIVEN_SPECIES`` below, which are deliberately
withheld. Today that is glucose: the coupler publishes the reactor's medium
glucose for the WCM arm, and letting it through here would overwrite this
model's own calibrated ``GLCx``. See the block comment on that constant. The Millard step then
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


# Molecules the reactor must NOT drive in the Millard cell, even though the
# alias table can resolve them.
#
# `ReactorCellCoupler` publishes the reactor's medium glucose into the shared
# top-level environment store so the WCM arm's cell can finally see it. That
# store is also this bridge's input, and `GLC[p]` is already aliased to the
# SBML species `GLCx` — so without this guard the reactor's pool (~22.2 mM in a
# default batch) overwrites the Millard model's own calibrated external glucose
# (0.00633 mM) on every tick, a ~3500x step change in the driver of its glucose
# rate law, and overrides both `GLCx`'s own dynamics and its `_GLC_FEED`
# chemostat. Measured: it breaks the mbp-07 O2 negative-feedback behaviour
# (CYTBO stops throttling at low DO).
#
# ⚠ This is a DEFERRAL, not a judgement. Whether a coupled reactor *should* set
# the Millard cell's external glucose is a real modelling question — arguably it
# should — but answering it means re-calibrating that model and regenerating
# mbp-07's committed figures and the millard_vs_beulig report card. That is a
# deliberate decision to make on its own, not a side effect of wiring the WCM
# arm's glucose path.
# Keyed on the RESOLVED SBML species, not on the spelling. `EXTERNAL_NAME_TO_SBML`
# maps `GLC`, `GLC[p]` AND `GLC[c]` all onto `GLCx`, so a set of v2ecoli names
# would go silently inert the moment a producer spelled it differently --
# including the plausible next step of the coupler emitting bare ids to match
# the boundary convention. Resolving first makes the guard independent of that.
_NOT_REACTOR_DRIVEN_SPECIES: frozenset[str] = frozenset({"GLCx"})


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
            species = EXTERNAL_NAME_TO_SBML.get(name)
            if species is None:
                continue
            if species in _NOT_REACTOR_DRIVEN_SPECIES:
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
