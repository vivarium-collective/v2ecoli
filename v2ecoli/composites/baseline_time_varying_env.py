"""baseline_time_varying_env — baseline composite with EnvironmentDriver hook.

Build-phase wire-up for mbp-01-time-varying-environment
(req-2-time-varying-composite). Adds a top-level ``environment`` data store
(with ``external_concentrations`` and ``media_id``) and three Steps —
``EnvironmentDriver``, ``EnvironmentMirror``, plus the existing per-cell
``media_update`` — wired in that order so a driver write at tick N is
visible to metabolism's exchange constraint at tick N.

Default ``env_driver_mode = "static"`` preserves baseline parity — in
static mode the Step is a no-op, so the composite must produce a
byte-identical trajectory to ``v2ecoli.composites.baseline`` (regression
guarded by mbp-01's ``static-env-baseline-unchanged`` test).

In ``synthetic_trajectory`` mode the driver writes to
``state["environment"]["external_concentrations"][mol]``. The
``EnvironmentMirror`` Step (landed 2026-05-28) reads that top-level dict
each tick and copies it into every agent's
``state["agents"][i]["environment"]["external_concentrations"]`` so that
``MediaUpdate`` (which lives inside ``agents.<id>``) sees the same data on
its per-cell ``("environment",)`` topology port. MediaUpdate's consumption
path (also landed 2026-05-28) then propagates the values into
``boundary.external`` as deltas; ExchangeData re-derives metabolism's
exchange constraints from there.

Closes the two architectural gaps captured 2026-05-28 in
``studies/mbp-01-time-varying-environment/study.yaml`` open_questions —
``env-store-topology-mismatch`` and ``env-driver-molecule-id-convention``.
Resolution adopts option (c) of each (single tick-leading mirror Step +
bare-molecule-name convention matching ``boundary.external``).
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.composites.baseline import baseline as _baseline_builder
from v2ecoli.steps.environment_driver import (
    ENV_DRIVER_MODE_STATIC,
    EnvironmentDriver,
)
from v2ecoli.steps.environment_mirror import EnvironmentMirror


ENVIRONMENT_DRIVER_STEP_NAME = "environment_driver"
ENVIRONMENT_MIRROR_STEP_NAME = "environment_mirror"


def _empty_environment_store() -> dict[str, Any]:
    """Top-level environment store (in static mode, never written to)."""
    return {
        "external_concentrations": {},
        "media_id":                "minimal",
    }


@composite_generator(
    name="baseline_time_varying_env",
    description=(
        "v2ecoli baseline + EnvironmentDriver/Mirror hooks so external "
        "physics can drive environment.external_concentrations each tick. "
        "Default env_driver_mode=static preserves baseline parity."
    ),
    parameters={
        "seed": {"type": "int", "default": 0},
        "cache_dir": {"type": "string", "default": "out/cache"},
        # See v2ecoli.steps.environment_driver for the mode enum + spec shape.
        "env_driver_mode": {"type": "string", "default": "static"},
        "synthetic_trajectory_spec": {"type": "object", "default": {}},
    },
)
def baseline_time_varying_env(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    env_driver_mode: str = ENV_DRIVER_MODE_STATIC,
    synthetic_trajectory_spec: dict | None = None,
) -> dict:
    """Build the baseline_time_varying_env document.

    Returns a process-bigraph document dict with the same shape as
    ``v2ecoli.composites.baseline.baseline`` plus an added top-level
    ``environment`` store, an ``environment_driver`` Step, and an
    ``environment_mirror`` Step.

    In ``static`` mode (default) the driver is a no-op so the composite is
    a byte-equivalent superset of the baseline.
    """
    document = _baseline_builder(core, seed=seed, cache_dir=cache_dir)

    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    state = document["state"]

    # Top-level environment store. The driver owns the
    # external_concentrations write path; pre-seed with empty dict so the
    # document is structurally complete before the first tick.
    state.setdefault("environment", _empty_environment_store())

    # EnvironmentMirror writes the driver-derived delta DIRECTLY to each
    # agent's boundary.external — which already has a typed schema via
    # sim_data initialization. No per-agent external_concentrations
    # pre-seed is needed (and we avoid the schema-inference fragility
    # the previous design ran into when the per-agent store was empty).
    # The agent-level environment store keeps its original shape
    # (media_id / exchange_data / exchange); only boundary.external
    # receives the driver's writes, exactly as the static media_id-
    # transition path already does.

    driver_config = {
        "env_driver_mode":           env_driver_mode,
        "synthetic_trajectory_spec": synthetic_trajectory_spec or {},
    }
    driver = _make_instance(EnvironmentDriver, driver_config, core)
    state[ENVIRONMENT_DRIVER_STEP_NAME] = make_edge(
        driver, EnvironmentDriver.topology, edge_type="step",
        config=driver_config,
    )

    mirror_config: dict[str, Any] = {}
    mirror = _make_instance(EnvironmentMirror, mirror_config, core)
    state[ENVIRONMENT_MIRROR_STEP_NAME] = make_edge(
        mirror, EnvironmentMirror.topology, edge_type="step",
        config=mirror_config,
    )

    # Register in flow_order. Critical layer-placement detail: place driver
    # and mirror BEFORE the unique_update FLUSH barrier that PRECEDES
    # media_update, so the FLUSH barrier separates the two layers and
    # media_update reads the FLUSH-committed state.
    #
    # Without this care, driver/mirror/media_update all land in the same
    # PBG layer (the one between unique_update_1 and unique_update_2),
    # in which case PBG's per-layer semantics read inputs at LAYER START
    # — media_update sees the empty external_concentrations and never
    # reacts to the driver's same-tick write. Probed and confirmed
    # 2026-05-28.
    #
    # Target shape (Layer 0 -> FLUSH -> Layer 1):
    #   [post-division-mass-listener, environment_driver, environment_mirror,
    #    unique_update_1, media_update, ...]
    flow_order = document.setdefault("flow_order", [])
    if "media_update" in flow_order:
        media_idx = flow_order.index("media_update")
        # Find the unique_update FLUSH immediately preceding media_update.
        flush_idx = media_idx - 1
        while flush_idx > 0 and not flow_order[flush_idx].startswith("unique_update_"):
            flush_idx -= 1
        # Insert driver + mirror BEFORE that FLUSH so they land in the
        # earlier layer and the FLUSH commits their writes before
        # media_update reads.
        insert_at = flush_idx if flow_order[flush_idx].startswith("unique_update_") else media_idx
        flow_order.insert(insert_at, ENVIRONMENT_MIRROR_STEP_NAME)
        flow_order.insert(insert_at, ENVIRONMENT_DRIVER_STEP_NAME)
    else:
        flow_order.extend([ENVIRONMENT_DRIVER_STEP_NAME, ENVIRONMENT_MIRROR_STEP_NAME])

    return document
