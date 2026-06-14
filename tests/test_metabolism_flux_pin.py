"""Unit tests for the FBA-bridge flux-pin helper in metabolism.py.

The helper hard-pins each mapped reaction before the LP solve and, for any
reaction whose hard pin makes the LP infeasible, relaxes it to a soft kinetic
target so the LP still solves. The fba-mutating operations are injected as
callables so the logic is testable without a real FBA solver and so the call
site can bind the real (API-specific) fba methods.
"""

from v2ecoli.processes.metabolism import apply_flux_pins_with_fallback


class _FakeFBA:
    """Minimal stand-in: a reaction is 'infeasible' if hard-bounded AND in the
    bad set."""

    def __init__(self, infeasible_ids):
        self.bounds = {}
        self.targets = {}
        self._bad = set(infeasible_ids)

    def set_bounds(self, rid, v):
        self.bounds[rid] = (v, v)

    def open_bounds(self, rid):
        self.bounds.pop(rid, None)

    def set_target(self, rid, v):
        self.targets[rid] = v

    def feasible(self):
        return not (set(self.bounds) & self._bad)


def _apply(fba, pins):
    return apply_flux_pins_with_fallback(
        fba, pins,
        open_bounds=fba.open_bounds,
        set_soft_target=fba.set_target,
        is_feasible=fba.feasible,
        set_hard_bound=fba.set_bounds,
    )


def test_feasible_pins_stay_hard():
    fba = _FakeFBA([])
    relaxed = _apply(fba, {"R1": 1.0, "R2": 2.0})
    assert relaxed == []
    assert fba.bounds == {"R1": (1.0, 1.0), "R2": (2.0, 2.0)}
    assert fba.targets == {}


def test_infeasible_pin_relaxed_to_target():
    fba = _FakeFBA(["R2"])
    relaxed = _apply(fba, {"R1": 1.0, "R2": 2.0})
    assert relaxed == ["R2"]
    assert "R2" not in fba.bounds and fba.targets["R2"] == 2.0
    assert fba.bounds["R1"] == (1.0, 1.0)


def test_no_pins_is_noop():
    fba = _FakeFBA([])
    relaxed = _apply(fba, {})
    assert relaxed == []
    assert fba.bounds == {}
    assert fba.targets == {}
