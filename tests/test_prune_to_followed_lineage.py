"""Regression test for `prune_to_followed_lineage`.

The bug being guarded against (2026-05-25, fixed in 7f3d0a2):
process-bigraph's Composite caches `process_paths` / `step_paths` /
`front` from the most recent `find_instance_paths` call. Deleting an
agent from `composite.state['agents']` without rebuilding those caches
leaves dangling None refs that crash on the next tick with
`TypeError: 'NoneType' object is not subscriptable` in run_process.

These tests pin the contract:
  1. siblings are deleted from state.agents
  2. composite.find_instance_paths(composite.state) is called after the
     delete (the actual fix — without it, run() crashes next tick)
  3. count of dropped agents is returned

Tests use a tiny mock composite (state + find_instance_paths spy) so
they don't need a full v2ecoli composite — fast + isolated.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from v2ecoli.library.sqlite_run import prune_to_followed_lineage


def _fake_composite(agent_ids: list[str]) -> Any:
    """Tiny mock composite. `state['agents']` populated with mock agents;
    `find_instance_paths` is a Mock so we can assert it was called."""
    comp = MagicMock()
    comp.state = {"agents": {aid: {"_mock_agent": aid} for aid in agent_ids}}
    return comp


def test_drops_all_but_followed_lineage():
    comp = _fake_composite(["00", "01"])
    dropped = prune_to_followed_lineage(comp, "00")
    assert dropped == 1
    assert sorted(comp.state["agents"].keys()) == ["00"]


def test_no_op_when_only_followed_present():
    comp = _fake_composite(["00"])
    dropped = prune_to_followed_lineage(comp, "00")
    assert dropped == 0
    assert sorted(comp.state["agents"].keys()) == ["00"]


def test_handles_missing_followed_in_state():
    # followed_id not in state — drop everything.
    comp = _fake_composite(["00", "01"])
    dropped = prune_to_followed_lineage(comp, "doesnotexist")
    assert dropped == 2
    assert comp.state["agents"] == {}


def test_drops_many_siblings():
    comp = _fake_composite(["0", "00", "01", "10", "11"])
    dropped = prune_to_followed_lineage(comp, "0")
    assert dropped == 4
    assert sorted(comp.state["agents"].keys()) == ["0"]


def test_calls_find_instance_paths_after_delete():
    """THE REGRESSION TEST. Without this call, the composite's cached
    process_paths/step_paths/front hold dangling refs and the next tick
    crashes."""
    comp = _fake_composite(["00", "01"])
    prune_to_followed_lineage(comp, "00")
    comp.find_instance_paths.assert_called_once()
    # And it must be called with the NOW-pruned state.
    args, _ = comp.find_instance_paths.call_args
    assert args[0] is comp.state
    assert sorted(args[0]["agents"].keys()) == ["00"]


def test_call_order_state_mutation_before_find_instance_paths():
    """The state must be pruned BEFORE find_instance_paths is called —
    otherwise find_instance_paths sees the stale (un-pruned) state and
    re-registers paths we're about to drop."""
    comp = _fake_composite(["00", "01"])
    captured = {}

    def capture(state):
        captured["agents_at_call_time"] = sorted(state.get("agents", {}).keys())

    comp.find_instance_paths.side_effect = capture
    prune_to_followed_lineage(comp, "00")
    assert captured["agents_at_call_time"] == ["00"]


def test_empty_state_no_crash():
    comp = MagicMock()
    comp.state = {}
    dropped = prune_to_followed_lineage(comp, "00")
    assert dropped == 0
    # find_instance_paths still called (defensive: composite may have
    # other internal state we want re-derived).
    comp.find_instance_paths.assert_called_once()
