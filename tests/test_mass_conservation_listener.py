"""MassConservationListener: per-tick dry-mass change vs net environment exchange.

Exercises the Step's update() logic directly with synthetic states (no full
composite): first-tick baseline, exact conservation (no warning), and a
violation (residual + warning).
"""

import warnings

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.listeners.mass_conservation import MassConservationListener
from v2ecoli.types.quantity import ureg as units


def _make_step(tolerance=1.0e-2):
    step = MassConservationListener({}, build_core())
    # Set config-derived state directly (bypass config realize): 0.5 fg per
    # GLC molecule.
    step.exchange_masses = {"GLC": 0.5 * units.fg}
    step.tolerance = tolerance
    step._prev_dry_mass = None
    return step


def _state(dry_mass_fg, glc_count):
    return {
        "listeners": {"mass": {"dry_mass": dry_mass_fg * units.fg}},
        "environment": {"exchange": {"GLC": glc_count}},
    }


def test_first_tick_establishes_baseline_no_residual():
    step = _make_step()
    out = step.update(_state(1000.0, -100))
    m = out["listeners"]["mass"]
    assert m["conservation_residual"].to(units.fg).magnitude == 0.0
    assert m["conservation_residual_relative"] == 0.0
    # net mass IN = -(-100)*0.5 fg = +50 fg (uptake -> mass enters cell)
    assert m["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(50.0)


def test_exact_conservation_no_warning():
    step = _make_step()
    step.update(_state(1000.0, -100))  # baseline
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        # grew exactly the imported 50 fg
        out = step.update(_state(1050.0, -100))
    m = out["listeners"]["mass"]
    assert m["conservation_residual"].to(units.fg).magnitude == pytest.approx(0.0)
    assert m["conservation_residual_relative"] == pytest.approx(0.0)


def test_violation_emits_residual_and_warns():
    step = _make_step(tolerance=1.0e-2)
    step.update(_state(1000.0, -100))  # baseline
    with pytest.warns(UserWarning, match="mass-conservation residual"):
        # imported 50 fg but only grew 10 fg -> residual -40 fg
        out = step.update(_state(1010.0, -100))
    m = out["listeners"]["mass"]
    assert m["conservation_residual"].to(units.fg).magnitude == pytest.approx(-40.0)
    assert m["conservation_residual_relative"] == pytest.approx(4.0)


def test_secretion_sign_removes_mass():
    step = _make_step()
    step.update(_state(1000.0, 0))  # baseline
    # secretion: +100 to environment -> mass leaves cell (-50 fg)
    out = step.update(_state(950.0, 100))
    m = out["listeners"]["mass"]
    assert m["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(-50.0)
    # dry mass fell 50, net_in -50 -> residual 0
    assert m["conservation_residual"].to(units.fg).magnitude == pytest.approx(0.0)
