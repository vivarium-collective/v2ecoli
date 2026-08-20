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

import re
import warnings

import numpy as np

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


_COMPARTMENT_SUFFIX = re.compile(r"^(?P<bare>.+)\[[a-z]\]$")


def _resolve_boundary_keys(external: dict, boundary_ext: dict) -> dict[str, str]:
    """Map each driver/coupler molecule id onto the boundary key it addresses.

    Two conventions meet here. `EnvironmentDriver` writes BARE names (`GLC`),
    matching `boundary.external` directly. `ReactorCellCoupler` writes
    compartment-tagged v2ecoli ids (`OXYGEN-MOLECULE[p]`) because the Millard
    arm's alias table and `reactor_millard_env_bridge` are built around that
    form. `boundary.external` is keyed bare, so before this the coupler's writes
    matched NOTHING and were dropped by the fail-closed rule below -- silently,
    every tick, for every molecule. Measured on `reactor_bird_coupled`: the
    mirror saw {'CARBON-DIOXIDE[p]', 'OXYGEN-MOLECULE[p]'} and matched zero of
    them, so the reactor's dissolved gases never reached the cell at all.

    Exact match wins; otherwise the compartment tag is stripped and retried.

    An id that resolves to a boundary key claimed by another id is AMBIGUOUS
    (`X[p]` and `X[c]` both reducing to `X`) and both are dropped rather than
    silently letting the last one win.
    """
    resolved: dict[str, str] = {}
    claimed_by: dict[str, list[str]] = {}
    for mol in external:
        if mol in boundary_ext:
            key = mol
        else:
            match = _COMPARTMENT_SUFFIX.match(str(mol))
            bare = match.group("bare") if match else None
            key = bare if bare is not None and bare in boundary_ext else None
        if key is None:
            continue
        resolved[mol] = key
        claimed_by.setdefault(key, []).append(mol)

    for key, claimants in claimed_by.items():
        if len(claimants) > 1:
            for mol in claimants:
                resolved.pop(mol, None)
    return resolved


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
        # Per-tick read + delta-write; the only persistent state is the
        # unmatched-id tally below.
        #
        # This Step drops silently in two places -- an id it cannot resolve, and
        # a non-finite delta -- and a silent drop here is how the reactor's
        # entire dissolved-gas channel went missing without anything failing.
        # Count what is dropped so the condition is observable to a test or a
        # caller instead of being invisible, and name the offenders once.
        self.skipped_unmatched: dict[str, int] = {}
        self._warned_unmatched: set[str] = set()

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
            boundary_keys = _resolve_boundary_keys(external, boundary_ext)
            for mol, conc_raw in external.items():
                boundary_key = boundary_keys.get(mol)
                if boundary_key is None:
                    # metabolism doesn't track this molecule (or the id is
                    # ambiguous); fail closed, but on the record.
                    self.skipped_unmatched[mol] = self.skipped_unmatched.get(mol, 0) + 1
                    if mol not in self._warned_unmatched:
                        self._warned_unmatched.add(mol)
                        warnings.warn(
                            f"EnvironmentMirror: no boundary.external key for "
                            f"{mol!r}; its writes are being dropped. This is "
                            f"silent by design (fail closed) -- see "
                            f"skipped_unmatched for the running tally.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    continue
                curr_raw = boundary_ext.get(boundary_key)
                if curr_raw is None:
                    continue
                conc = float(conc_raw.magnitude) if hasattr(conc_raw, "magnitude") else float(conc_raw)
                curr = float(curr_raw.magnitude) if hasattr(curr_raw, "magnitude") else float(curr_raw)
                diff = conc - curr
                if not np.isfinite(diff):
                    # A non-finite delta means the CURRENT boundary value is
                    # non-finite: metabolism seeds an unlimited molecule (O2 is
                    # the canonical one) at inf, and `0.0 - inf` is -inf, which
                    # clears an isnan check and then accumulates to NaN in the
                    # additive boundary store -- silently poisoning that
                    # molecule for the rest of the run. A delta cannot express
                    # a change to an unlimited boundary at all, so fail closed
                    # and leave it untouched.
                    #
                    # NOTE this makes the failure safe, not fixed: inf-valued
                    # molecules remain UNREACHABLE by the driver. Driver
                    # control of dissolved O2 (the DO-limitation story) needs
                    # replace semantics on boundary.external and is expressly
                    # NOT provided here.
                    diff = 0.0
                conc_update[boundary_key] = diff
            if conc_update:
                agent_updates[agent_id] = {"boundary": {"external": conc_update}}

        if not agent_updates:
            return {}
        return {"agents": agent_updates}

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)
