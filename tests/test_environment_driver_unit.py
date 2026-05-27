"""Unit tests for :class:`v2ecoli.steps.environment_driver.EnvironmentDriver`.

Targets the Step's ``next_update`` + ``_evaluate_trajectory`` math directly
— no composite, no metabolism, no cache. Fast, runs without the
@pytest.mark.sim marker.

These do NOT substitute for the behavior tests in
``tests/test_mbp_01_time_varying_environment.py`` (which require the full
env-store → media_update → metabolism wire-up to lift their
@pytest.mark.skip). They validate the new code's pure trajectory + mode-
dispatch math under the chris_feedback_2026_05_26 §3 plumbing+extreme design.
"""

from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.environment_driver import (
    ENV_DRIVER_MODE_EXTERNAL_STORE,
    ENV_DRIVER_MODE_STATIC,
    ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
    EnvironmentDriver,
    TRAJ_CLAMP_TO_VALUE,
    TRAJ_LINEAR_DECLINE,
)


# Shared core (build_core is expensive).
@pytest.fixture(scope="module")
def core():
    return build_core()


# --- Construction / config plumbing -----------------------------------------

def test_default_mode_is_static(core):
    """env_driver_mode defaults to 'static' so unmodified baseline parity is
    preserved (mbp-01 static-env-baseline-unchanged regression guard).
    """
    e = EnvironmentDriver(config={}, core=core)
    assert e.env_driver_mode == ENV_DRIVER_MODE_STATIC
    assert e.synthetic_spec == {}


def test_config_overrides_take_effect(core):
    """Regression guard: bigraph-schema config_schema must use type names
    (not {"_default": ...}) for overrides to propagate. See the schema
    comment block in environment_driver.py."""
    spec = {"GLC[p]": {"kind": TRAJ_CLAMP_TO_VALUE, "value_gL": 5.0}}
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": spec,
        },
        core=core,
    )
    assert e.env_driver_mode == ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY
    assert e.synthetic_spec == spec


# --- Mode dispatch ----------------------------------------------------------

def test_static_mode_is_noop(core):
    """In static mode, next_update returns {} regardless of input state
    (preserves baseline composite parity)."""
    e = EnvironmentDriver(config={"env_driver_mode": ENV_DRIVER_MODE_STATIC}, core=core)
    assert e.next_update(1.0, {"environment": {"external_concentrations": {}}, "global_time": 0.0}) == {}
    # Even at later time, no synthetic spec, etc. — still no-op
    assert e.next_update(1.0, {"environment": {"external_concentrations": {"GLC[p]": 9.9}}, "global_time": 3600.0}) == {}


def test_external_store_mode_is_noop(core):
    """In external_store mode, ReactorCellCoupler (mbp-03) is the source of
    truth; the driver Step is a no-op."""
    e = EnvironmentDriver(config={"env_driver_mode": ENV_DRIVER_MODE_EXTERNAL_STORE}, core=core)
    assert e.next_update(1.0, {"environment": {}, "global_time": 0.0}) == {}


def test_unknown_mode_raises(core):
    """Defensive: an unknown env_driver_mode should fail fast rather than
    silently doing nothing."""
    e = EnvironmentDriver(config={"env_driver_mode": "nonsense"}, core=core)
    with pytest.raises(ValueError, match="unknown env_driver_mode"):
        e.next_update(1.0, {"environment": {}, "global_time": 0.0})


# --- Synthetic trajectory: clamp_to_value (mbp-01 extreme tests) ------------

@pytest.mark.parametrize("value_gL", [0.0, 5.0, 50.0])
def test_clamp_to_value_at_any_time(core, value_gL):
    """clamp_to_value emits the same concentration at every time (backs
    mbp-01 zero-substrate-blocks-uptake, saturating-substrate-respects-vmax,
    and plateau-across-saturating-range sims)."""
    spec = {"GLC[p]": {"kind": TRAJ_CLAMP_TO_VALUE, "value_gL": value_gL}}
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": spec,
        },
        core=core,
    )
    for t_sec in (0.0, 600.0, 1800.0, 3600.0):
        out = e.next_update(1.0, {"environment": {}, "global_time": t_sec})
        concs = out["environment"]["external_concentrations"]
        # Stored as pint Quantity in mM
        assert concs["GLC[p]"].magnitude == pytest.approx(value_gL)


def test_clamp_to_value_mmolL_takes_precedence_over_gL(core):
    """If both value_mmolL and value_gL are present, mmolL wins (explicit
    > implicit unit conversion)."""
    spec = {"GLC[p]": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": 28.0, "value_gL": 999.0}}
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": spec,
        },
        core=core,
    )
    out = e.next_update(1.0, {"environment": {}, "global_time": 0.0})
    assert out["environment"]["external_concentrations"]["GLC[p]"].magnitude == pytest.approx(28.0)


def test_clamp_without_value_keys_is_skipped(core):
    """A malformed clamp spec (no value_gL / value_mmolL) should yield no
    update for that molecule rather than crash."""
    spec = {"GLC[p]": {"kind": TRAJ_CLAMP_TO_VALUE}}   # missing value_*
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": spec,
        },
        core=core,
    )
    out = e.next_update(1.0, {"environment": {}, "global_time": 0.0})
    assert out == {}   # nothing to update


# --- Synthetic trajectory: linear_decline ----------------------------------

def test_linear_decline_at_endpoints(core):
    """linear_decline returns start_gL at t=0 and end_gL at t=duration_min."""
    spec = {"GLC[p]": {"kind": TRAJ_LINEAR_DECLINE, "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0}}
    val = EnvironmentDriver._evaluate_trajectory(spec["GLC[p]"], t_min=0.0)
    assert val == pytest.approx(5.0)
    val = EnvironmentDriver._evaluate_trajectory(spec["GLC[p]"], t_min=60.0)
    assert val == pytest.approx(0.0)


def test_linear_decline_at_midpoint(core):
    """At t = duration_min / 2, value is the average of start and end."""
    val = EnvironmentDriver._evaluate_trajectory(
        {"kind": TRAJ_LINEAR_DECLINE, "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0},
        t_min=30.0,
    )
    assert val == pytest.approx(2.5)


def test_linear_decline_clamps_past_duration(core):
    """Past duration_min, the trajectory clamps at end_gL (doesn't overshoot
    into negative or extrapolate)."""
    val = EnvironmentDriver._evaluate_trajectory(
        {"kind": TRAJ_LINEAR_DECLINE, "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0},
        t_min=120.0,
    )
    assert val == pytest.approx(0.0)


def test_linear_decline_clamps_before_zero(core):
    """Negative t_min (shouldn't happen in practice but defensive) clamps
    to start_gL."""
    val = EnvironmentDriver._evaluate_trajectory(
        {"kind": TRAJ_LINEAR_DECLINE, "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0},
        t_min=-10.0,
    )
    assert val == pytest.approx(5.0)


# --- Synthetic trajectory: unknown kind / empty spec ------------------------

def test_unknown_trajectory_kind_is_skipped(core):
    """A spec with an unrecognized 'kind' should yield no update for that
    molecule (defensive; the test catches typos in study yamls)."""
    spec = {"GLC[p]": {"kind": "bogus", "value_gL": 1.0}}
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": spec,
        },
        core=core,
    )
    assert e.next_update(1.0, {"environment": {}, "global_time": 0.0}) == {}


def test_empty_synthetic_spec_emits_no_update(core):
    """env_driver_mode=synthetic + empty spec → no update (no molecules to drive)."""
    e = EnvironmentDriver(
        config={
            "env_driver_mode": ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            "synthetic_trajectory_spec": {},
        },
        core=core,
    )
    assert e.next_update(1.0, {"environment": {}, "global_time": 100.0}) == {}
