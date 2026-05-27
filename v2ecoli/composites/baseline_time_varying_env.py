"""baseline_time_varying_env — baseline composite with EnvironmentDriver hook.

Build-phase scaffold for mbp-01-time-varying-environment
(req-2-time-varying-composite). Thin wrapper around ``v2ecoli.composites.baseline``
that injects the ``EnvironmentDriver`` Step and the
``environment.external_concentrations`` store. Backward-compatible default:
``env_driver_mode = "static"`` — the composite must produce a byte-identical
trajectory to ``v2ecoli.composites.baseline`` in that mode (regression
guarded by mbp-01's ``static-env-baseline-unchanged`` test).

See ``studies/mbp-01-time-varying-environment/study.yaml`` for the full spec.

TODO (Build phase):
  - Wire EnvironmentDriver into the existing baseline document. Inject the
    Step into the Layer-1 step layer (where media_update lives — see
    v2ecoli/composites/_helpers.py for layer constants).
  - Modify media_update / exchange_data to read from
    environment.external_concentrations when env_driver_mode != static. The
    AGENTS.md convention is "do not modify a process in one architecture
    without a plan for the other two" — this hook lands across all three.
  - Confirm the cache fingerprint picks up this file (add to INPUT_FILES
    in v2ecoli/library/cache_version.py).
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

# Reuse the partitioned-baseline builder.
from v2ecoli.composites.baseline import baseline as _baseline_builder


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
    core: Any,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    env_driver_mode: str = "static",
    synthetic_trajectory_spec: dict | None = None,
) -> dict:
    """Build the baseline_time_varying_env document.

    Build-phase scaffold: today this just delegates to the unmodified
    baseline. The TODO above tracks the actual wire-up work.
    """
    document = _baseline_builder(core, seed=seed, cache_dir=cache_dir)

    # TODO: inject EnvironmentDriver Step into document["composition"][...]
    # and wire its outputs to environment.external_concentrations.
    # Add a regression assertion: at env_driver_mode="static", the resulting
    # document is byte-identical to the unmodified baseline document.

    # Stash config on the document for downstream wiring once the hook lands.
    document.setdefault("_v2ecoli_meta", {})["env_driver_mode"] = env_driver_mode
    document["_v2ecoli_meta"]["synthetic_trajectory_spec"] = synthetic_trajectory_spec or {}

    return document
