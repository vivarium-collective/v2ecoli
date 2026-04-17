"""Tests for the EcoliStep.perform_update gate.

Pins the contract that:

  1. By default, `invoke` calls `update` — the base `perform_update`
     returns True.
  2. When a subclass overrides `perform_update` to return False,
     `invoke` short-circuits with an empty update and does NOT call
     `update` (no hot-path work, no state changes).
  3. `perform_update` receives the same `state` dict that `invoke`
     was called with, so subclasses can gate on `global_time`,
     `timestep`, or any other port value.

This is the gate machinery ported from vEcoli composite-branch
commit 35003119. Listeners that define a legacy
`update_condition(timestep, states)` were previously dead code in
v2ecoli because `invoke` never consulted them; moving the logic into
`perform_update` restores the intended skip-on-non-emit-tick
behavior.
"""
from __future__ import annotations

import pytest

from v2ecoli.library.ecoli_step import EcoliStep


pytestmark = pytest.mark.fast


class _CountingStep(EcoliStep):
    """Test double: records every update() call and whether it ran."""

    config_schema = {}

    def initialize(self, config):
        self.update_calls = 0
        self.last_state = None

    def update(self, state, interval=None):
        self.update_calls += 1
        self.last_state = state
        return {'ran': True}


class _GatedStep(_CountingStep):
    """A step that only runs when `state['gate']` is truthy."""

    def perform_update(self, state):
        return bool(state.get('gate', False))


class _TimestepGatedStep(_CountingStep):
    """Listener-style gate: run only when global_time is a timestep multiple."""

    def perform_update(self, state):
        global_t = state.get('global_time')
        timestep = state.get('timestep')
        if global_t is None or timestep is None or timestep == 0:
            return True
        return (global_t % timestep) == 0


def test_default_perform_update_is_true():
    """A step that doesn't override perform_update always runs."""
    step = _CountingStep(parameters={})
    result = step.invoke({'foo': 1}, interval=1.0)

    assert step.update_calls == 1
    # SyncUpdate wraps the return value; the inner dict is what update returned
    assert result.update == {'ran': True}


def test_perform_update_false_skips_update():
    """When perform_update returns False, invoke returns an empty update
    and the update() body is not executed."""
    step = _GatedStep(parameters={})
    result = step.invoke({'gate': False}, interval=1.0)

    assert step.update_calls == 0
    assert step.last_state is None
    assert result.update == {}


def test_perform_update_true_runs_update():
    """Sanity: when perform_update returns True, update runs normally."""
    step = _GatedStep(parameters={})
    result = step.invoke({'gate': True}, interval=1.0)

    assert step.update_calls == 1
    assert result.update == {'ran': True}


def test_perform_update_sees_invoke_state():
    """The state dict passed to invoke reaches perform_update unchanged —
    so subclasses can gate on global_time/timestep/etc."""
    step = _TimestepGatedStep(parameters={})

    # global_time=10, timestep=5 → 10 % 5 == 0 → runs.
    r1 = step.invoke({'global_time': 10.0, 'timestep': 5.0}, interval=1.0)
    assert step.update_calls == 1
    assert r1.update == {'ran': True}

    # global_time=13, timestep=5 → 13 % 5 == 3 → skipped.
    r2 = step.invoke({'global_time': 13.0, 'timestep': 5.0}, interval=1.0)
    assert step.update_calls == 1  # unchanged
    assert r2.update == {}
