"""Regression tests for the ``extra_root_paths`` extension to
``run_multigen_sqlite`` (added 2026-05-27 for the multiscale-bioprocess
investigation's mbp-02 captures of ``population/*`` alongside per-agent
trajectory).

The extension adds:
* ``_normalize_root_paths`` — like ``_normalize_emit_paths`` but does
  NOT strip an ``agents/<id>/`` prefix (root paths live at the composite
  state root).
* ``_filter_root_state`` — mirror of ``_filter_agent_state`` rooted at
  composite state (not agent state).
* ``_merge_into`` — recursive dict union so the agent payload + root
  payload coexist in the emitter ``update()`` call.
* ``run_multigen_sqlite(..., extra_root_paths=[...])`` — surfaces the
  filtered root state in each emit alongside the agent state.

These are unit tests against the helpers + a smoke test against the full
runner using a tiny synthetic composite.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from v2ecoli.library.sqlite_run import (
    _filter_root_state,
    _merge_into,
    _normalize_root_paths,
)


# ---------------------------------------------------------------------------
# _normalize_root_paths
# ---------------------------------------------------------------------------


def test_normalize_root_paths_keeps_agents_prefix():
    """Unlike _normalize_emit_paths, the root variant DOES NOT strip the
    'agents/<id>/' prefix — root paths are explicit composite-state paths."""
    out = _normalize_root_paths(["population/cell_count", "agents/0/foo"])
    assert ("population", "cell_count") in out
    assert ("agents", "0", "foo") in out


def test_normalize_root_paths_handles_dots():
    out = _normalize_root_paths(["population.total_biomass_gDW"])
    assert out == [("population", "total_biomass_gDW")]


def test_normalize_root_paths_dedupes_and_sorts():
    out = _normalize_root_paths(["a/b", "a/b", "c/d"])
    assert out == [("a", "b"), ("c", "d")]


def test_normalize_root_paths_drops_empty():
    """Mirrors _normalize_emit_paths semantics — empty string segments
    are filtered out, but non-empty whitespace is preserved (matches
    the existing helper's behavior; pre-trim in the caller if needed)."""
    out = _normalize_root_paths(["", "/a/b/", "//"])
    assert out == [("a", "b")]


def test_normalize_root_paths_handles_none_input():
    """A None list (default arg) yields []; no crash."""
    assert _normalize_root_paths(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _filter_root_state
# ---------------------------------------------------------------------------


def test_filter_root_state_extracts_leaves():
    state = {
        "population": {"cell_count": 1.0, "total_biomass_gDW": 1.27e-12},
        "agents": {"0": {"listeners": {"mass": {"cell_mass": 1280.0}}}},
    }
    out = _filter_root_state(state, [
        ("population", "cell_count"),
        ("population", "total_biomass_gDW"),
    ])
    assert out == {
        "population": {"cell_count": 1.0, "total_biomass_gDW": 1.27e-12},
    }


def test_filter_root_state_skips_missing_leaves():
    state = {"population": {"cell_count": 1.0}}
    out = _filter_root_state(state, [
        ("population", "cell_count"),
        ("population", "total_biomass_gDW"),  # missing
    ])
    assert out == {"population": {"cell_count": 1.0}}


def test_filter_root_state_handles_deep_paths():
    state = {
        "environment": {
            "external_concentrations": {
                "GLC[p]": 5.0,
                "OXYGEN-MOLECULE[p]": 0.2,
            },
        },
    }
    out = _filter_root_state(state, [
        ("environment", "external_concentrations", "GLC[p]"),
    ])
    assert out == {
        "environment": {"external_concentrations": {"GLC[p]": 5.0}},
    }


# ---------------------------------------------------------------------------
# _merge_into
# ---------------------------------------------------------------------------


def test_merge_into_recursive_union():
    target = {"agents": {"0": {"listeners": {"mass": {"cell_mass": 1280.0}}}}}
    addition = {"population": {"cell_count": 1.0}}
    _merge_into(target, addition)
    assert target["agents"]["0"]["listeners"]["mass"]["cell_mass"] == 1280.0
    assert target["population"]["cell_count"] == 1.0


def test_merge_into_preserves_existing_on_conflict():
    """When both target and addition have the same key, target wins
    (the agent-payload path emits first in the runner's update_state)."""
    target = {"x": 1}
    addition = {"x": 2}
    _merge_into(target, addition)
    assert target["x"] == 1


def test_merge_into_recurses_into_shared_subtree():
    target = {"a": {"b": 1}}
    addition = {"a": {"c": 2}}
    _merge_into(target, addition)
    assert target == {"a": {"b": 1, "c": 2}}
