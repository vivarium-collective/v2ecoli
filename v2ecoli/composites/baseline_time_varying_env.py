"""baseline_time_varying_env — baseline composite with EnvironmentDriver hook.

Build-phase wire-up for mbp-01-time-varying-environment
(req-2-time-varying-composite). Adds a top-level ``environment`` data store
(with ``external_concentrations`` and ``media_id``) and an
``EnvironmentDriver`` Step alongside the existing ``agents`` /
``global_time`` top-level keys.

Default ``env_driver_mode = "static"`` preserves baseline parity — in
static mode the Step is a no-op, so the composite must produce a
byte-identical trajectory to ``v2ecoli.composites.baseline`` (regression
guarded by mbp-01's ``static-env-baseline-unchanged`` test).

In ``synthetic_trajectory`` mode the Step writes to
``environment.external_concentrations.<mol>`` per the spec — but the
existing media_update / exchange_data path does NOT yet consume from that
store. Driving metabolism from external_concentrations requires modifying
media_update (a PartitionedProcess in 3 architectures per AGENTS.md) and
re-running ParCa cache — tracked as a follow-up TODO. Until that lands,
the synthetic-mode plumbing tests in
``tests/test_mbp_01_time_varying_environment.py`` remain @pytest.mark.skip.
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


ENVIRONMENT_DRIVER_STEP_NAME = "environment_driver"


def _empty_environment_store() -> dict[str, Any]:
    """Top-level environment store (in static mode, never written to)."""
    return {
        "external_concentrations": {},
        "media_id":                "minimal",
    }


@composite_generator(
    name="baseline_time_varying_env",
    description=(
        "v2ecoli baseline + EnvironmentDriver hook so external physics can "
        "drive environment.external_concentrations each timestep. Default "
        "env_driver_mode=static preserves baseline parity."
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
    ``environment`` store and ``environment_driver`` Step.

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

    driver_config = {
        "env_driver_mode":           env_driver_mode,
        "synthetic_trajectory_spec": synthetic_trajectory_spec or {},
    }
    driver = _make_instance(EnvironmentDriver, driver_config, core)
    state[ENVIRONMENT_DRIVER_STEP_NAME] = make_edge(
        driver, EnvironmentDriver.topology, edge_type="step",
        config=driver_config,
    )

    # Register in flow_order so the executor knows about the new step.
    document.setdefault("flow_order", []).append(ENVIRONMENT_DRIVER_STEP_NAME)

    return document
