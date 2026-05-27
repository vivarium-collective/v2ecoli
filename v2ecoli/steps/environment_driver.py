"""EnvironmentDriver — externally-driven time-varying environment hook.

Build-phase scaffold for mbp-01-time-varying-environment (see
`studies/mbp-01-time-varying-environment/study.yaml`, req-1-env-driver +
chris_feedback_2026_05_26 §3).

PBG Step that, each timestep, writes to ``environment.external_concentrations``
from one of:

  - ``env_driver_mode = "static"``           — no-op; preserves baseline composite
  - ``env_driver_mode = "external_store"``   — reads from an external source store
                                                 (default for reactor coupling under
                                                 mbp-03)
  - ``env_driver_mode = "synthetic_trajectory"`` — drives concentrations from a
                                                    deterministic function of time
                                                    (this is what the mbp-01 plumbing
                                                    sims use)

Hooks the existing ``media_update`` to read from ``environment.external_concentrations``
when ``env_driver_mode != static``. Backward-compatible default is ``static`` to
preserve every existing baseline regression (open_question
``static-env-default-key`` in mbp-01 study.yaml tracks the final default).

The synthetic-trajectory mode supports the perturbation kinds used by mbp-01's
behavior tests:

  - ``linear_decline``         — glucose 5 g/L → 0 g/L over the run
  - ``clamp_to_value``         — hold concentration at a fixed value (used by
                                   zero-substrate-blocks-uptake,
                                   saturating-substrate-respects-vmax,
                                   plateau-across-saturating-range)
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.quantity import ureg as units
from v2ecoli.types.stores import InPlaceDict


# --- env-driver-mode constants ----------------------------------------------

ENV_DRIVER_MODE_STATIC = "static"
ENV_DRIVER_MODE_EXTERNAL_STORE = "external_store"
ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY = "synthetic_trajectory"


# --- trajectory kinds -------------------------------------------------------

TRAJ_LINEAR_DECLINE = "linear_decline"
TRAJ_CLAMP_TO_VALUE = "clamp_to_value"


class EnvironmentDriver(Step):
    """Drive ``environment.external_concentrations`` from an external source.

    See module docstring; spec in
    ``studies/mbp-01-time-varying-environment/study.yaml`` req-1-env-driver.
    """

    name = "environment_driver"
    # See PopulationAggregator config_schema for the type-vs-_default note.
    # synthetic_trajectory_spec is a free-form dict keyed by molecule ID, so
    # leave it as "map" (or empty mapping); see initialize() for the read.
    config_schema = {
        "env_driver_mode":           "string",
        "synthetic_trajectory_spec": "map",
        "time_step":                 "float",
    }
    topology = {
        "environment": ("environment",),
        "global_time": ("global", "time"),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.env_driver_mode = cfg.get("env_driver_mode") or ENV_DRIVER_MODE_STATIC
        self.synthetic_spec = cfg.get("synthetic_trajectory_spec") or {}

    def inputs(self) -> dict[str, Any]:
        return {
            "environment": InPlaceDict(),
            "global_time": 0.0,
        }

    def outputs(self) -> dict[str, Any]:
        return {"environment": InPlaceDict()}

    # --- main update -------------------------------------------------------

    def next_update(self, timestep, states):
        # static: no-op (preserves baseline behavior; regression-guarded by
        # static-env-baseline-unchanged in mbp-01).
        if self.env_driver_mode == ENV_DRIVER_MODE_STATIC:
            return {}

        if self.env_driver_mode == ENV_DRIVER_MODE_EXTERNAL_STORE:
            # external_store mode: ReactorCellCoupler (mbp-03) writes
            # environment.external_concentrations directly each step; this
            # Step is a no-op when the coupler is the source of truth.
            # Kept in the topology so media_update / exchange_data downstream
            # see the right store path regardless of mode.
            return {}

        if self.env_driver_mode == ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY:
            t_min = float(states.get("global_time", 0.0)) / 60.0
            updates: dict[str, Any] = {}
            for mol_id, spec in self.synthetic_spec.items():
                val = self._evaluate_trajectory(spec, t_min)
                if val is not None:
                    # Stored in mM (v2ecoli convention); spec values supplied in g/L
                    # are converted via molar-mass — TODO once metabolism's external
                    # exchange path is wired in.
                    updates[mol_id] = val * units.mM
            if not updates:
                return {}
            return {"environment": {"external_concentrations": updates}}

        raise ValueError(f"unknown env_driver_mode: {self.env_driver_mode!r}")

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)

    # --- trajectory evaluation -------------------------------------------

    @staticmethod
    def _evaluate_trajectory(spec: dict, t_min: float) -> float | None:
        kind = spec.get("kind")

        if kind == TRAJ_CLAMP_TO_VALUE:
            # Spec keys: value_gL OR value_mmolL
            if "value_mmolL" in spec:
                return float(spec["value_mmolL"])
            if "value_gL" in spec:
                # TODO units conversion: needs molar mass per molecule_id
                return float(spec["value_gL"])
            return None

        if kind == TRAJ_LINEAR_DECLINE:
            start = float(spec["start_gL"])
            end = float(spec["end_gL"])
            duration = float(spec["duration_min"])
            frac = max(0.0, min(1.0, t_min / duration))
            return start + (end - start) * frac

        return None
