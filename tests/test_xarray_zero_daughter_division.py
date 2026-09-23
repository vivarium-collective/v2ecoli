"""P1-15: a zero-daughter division must not silently truncate a lineage.

``run_multigen_xarray`` ends the current generation when its division
detector fires. If that division leaves zero survivors to follow (e.g. a
custom ``division_detector`` flags division based on something other than
plain agent-count growth, and every daughter died or was pruned elsewhere),
the loop used to just ``break`` and return normally — the process exits 0
with fewer generations than requested and no signal that anything went
wrong (CD2 audit §3.6). This must now warn by default and raise under
``raise_on_zero_daughters=True``.

These tests drive the runner with a tiny fake composite (same pattern as
``test_multigen_last_generation.py``) that "divides" into zero daughters at
a scheduled tick, paired with a custom ``division_detector`` that flags a
drop in agent count as a division (representing an out-of-band division
signal, since the runner's default detector only fires on agent-count
GROWTH and would never see a zero-survivor case).
"""
from __future__ import annotations

import warnings

import pytest

# Warm up build_core()'s heavy import chain (ray/cobra/scipy/sympy/ruamel...)
# at COLLECTION time, before any test installs a `pytest.warns`/`catch_warnings`
# recorder. Several of those third-party imports mutate the *global*
# `warnings.filters` list on first import (including, transitively, a blanket
# `ignore` entry for the bare `Warning` base class) — if that first import
# happens lazily INSIDE a `pytest.warns(...)` block (as it would if
# `build_core()` were only called from inside a test), the mutation clobbers
# the block's own recording filter and warnings raised afterward in the same
# block go uncaptured. Importing here once, up front, makes the mutation a
# no-op by the time any test's `pytest.warns` runs.
from v2ecoli.core import build_core
build_core()


class _FakeCompositeZeroDaughters:
    """Single-cell composite that divides into ZERO daughters at a scheduled
    tick — the mother is removed and nothing replaces her."""

    def __init__(self, divide_at: int = 50, dry0: float = 350.0):
        self.divide_at = divide_at
        self.dry0 = dry0
        self._t = 0
        self.state = {"agents": {"0": self._cell(dry0)}, "global_time": 0.0}

    @staticmethod
    def _cell(dry_mass: float) -> dict:
        return {"listeners": {"mass": {"dry_mass": float(dry_mass)}}}

    def run(self, n: int) -> None:
        for _ in range(int(n)):
            self._t += 1
            agents = self.state["agents"]
            if agents and self._t >= self.divide_at:
                agents.clear()  # zero-daughter division
            elif agents:
                cur_id = next(iter(agents))
                agents[cur_id]["listeners"]["mass"]["dry_mass"] = (
                    self.dry0 + self._t * 2.0)
        self.state["global_time"] = float(self._t)

    def find_instance_paths(self, state):
        return {}

    core = None


def _zero_survivor_detector(prev_ids, curr_ids):
    """Flags a division whenever the agent set shrank to nothing — standing
    in for an out-of-band division signal a real composite might raise even
    when no daughter agent id shows up."""
    if len(curr_ids) < len(prev_ids):
        return True, None
    return False, None


def _run(tmp_path, *, raise_on_zero_daughters=False, max_generations=3):
    from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths
    from v2ecoli.core import build_core

    comp = _FakeCompositeZeroDaughters(divide_at=50, dry0=350.0)
    comp.core = build_core()
    return run_multigen_xarray(
        comp,
        store_path=str(tmp_path / "zero-daughter.zarr"),
        view=view_from_emit_paths(["listeners.mass.dry_mass"]),
        metadata_base={"experiment_id": "zero-daughter-test", "engine": "fake",
                       "condition": "test", "variant": 0, "lineage_seed": 7,
                       "time_step": 1.0, "max_duration": 900.0, "agent_id": "0"},
        max_steps=900,
        max_generations=max_generations,   # last_gen = 3, division at gen 1
        chunk=10,
        division_detector=_zero_survivor_detector,
        raise_on_zero_daughters=raise_on_zero_daughters,
    )


def test_zero_daughter_division_warns_by_default(tmp_path):
    with pytest.warns(RuntimeWarning, match="zero-daughter"):
        res = _run(tmp_path)
    # Truncated at generation 1 instead of running the requested 3.
    assert res["generations"] == [1]


def test_zero_daughter_division_raises_under_strict_flag(tmp_path):
    with pytest.raises(RuntimeError, match="zero-daughter"):
        _run(tmp_path, raise_on_zero_daughters=True)


class _FakeCompositeNormalDivision:
    """Single-cell composite that divides normally on a schedule, reusing the
    mother id for daughter "…0" (mirrors ``test_multigen_last_generation``'s
    ``_FakeComposite`` — duplicated locally rather than cross-imported, since
    a bare ``tests`` module name collides with an unrelated installed
    "tests" package on some environments' sys.path)."""

    def __init__(self, divide_period: int = 200, dry0: float = 350.0):
        self.divide_period = divide_period
        self.dry0 = dry0
        self._age = 0
        self._t = 0
        self.state = {"agents": {"0": self._cell(dry0)}, "global_time": 0.0}

    @staticmethod
    def _cell(dry_mass: float) -> dict:
        import numpy as np
        return {
            "listeners": {"mass": {"dry_mass": float(dry_mass)}},
            "bulk": np.array([(b"X", 10)], dtype=[("id", "S1"), ("count", "i8")]),
        }

    def run(self, n: int) -> None:
        for _ in range(int(n)):
            self._age += 1
            self._t += 1
            agents = self.state["agents"]
            cur_id = next(iter(agents))
            if self._age >= self.divide_period:
                self._age = 0
                del agents[cur_id]
                agents["00"] = self._cell(self.dry0)
                agents["01"] = self._cell(self.dry0)
            else:
                agents[cur_id]["listeners"]["mass"]["dry_mass"] = (
                    self.dry0 + self._age * 2.0)
        self.state["global_time"] = float(self._t)

    def find_instance_paths(self, state):
        return {}

    core = None


def test_normal_division_unaffected(tmp_path):
    """A composite that divides normally (agent count grows) must not warn or
    raise — the new guard only fires on a zero-survivor division."""
    from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths
    from v2ecoli.core import build_core

    comp = _FakeCompositeNormalDivision(divide_period=200, dry0=350.0)
    comp.core = build_core()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_multigen_xarray(
            comp,
            store_path=str(tmp_path / "normal-division.zarr"),
            view=view_from_emit_paths(["listeners.mass.dry_mass"]),
            metadata_base={"experiment_id": "normal-division-test",
                           "engine": "fake", "condition": "test", "variant": 0,
                           "lineage_seed": 0, "time_step": 1.0,
                           "max_duration": 900.0, "agent_id": "0"},
            max_steps=900,
            max_generations=4,
            chunk=20,
            single_daughters=True,
        )
    # Only assert on OUR guard, not on unrelated third-party warnings (e.g. a
    # lazy-import DeprecationWarning) that may fire the first time a library
    # is touched in a given test process.
    assert not any("zero-daughter" in str(w.message) for w in caught)
    assert res["generations"] == [1, 2, 3, 4]
