"""MassConservationListener: per-tick Δcell_mass vs net environment exchange.

environment.exchange is CUMULATIVE (monotonic media depletion); the Step diffs
it to recover the per-tick exchange. Balance is against total cell_mass.
"""

import warnings

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.listeners.mass_conservation import MassConservationListener
from v2ecoli.types.quantity import ureg as units


def _make_step(tolerance=1.0e-2):
    step = MassConservationListener({}, build_core())
    step.exchange_masses = {"GLC": 0.5 * units.fg}   # 0.5 fg per GLC molecule
    step.tolerance = tolerance
    step._prev_cell_mass = None
    step._prev_exchange = {}
    return step


def _state(cell_mass_fg, glc_cumulative):
    # glc_cumulative is the CUMULATIVE count added to the environment.
    return {
        "listeners": {"mass": {"cell_mass": cell_mass_fg * units.fg}},
        "environment": {"exchange": {"GLC": glc_cumulative}},
    }


def test_first_tick_establishes_baseline():
    step = _make_step()
    # cumulative GLC -100 (uptake) -> per-tick -100 -> mass in +50 fg
    out = step.update(_state(1000.0, -100))["listeners"]["mass"]
    assert out["conservation_residual"].to(units.fg).magnitude == 0.0
    assert out["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(50.0)


def test_exact_conservation_no_warning():
    step = _make_step()
    step.update(_state(1000.0, -100))             # baseline; prev cumulative -100
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # cumulative -200 -> per-tick -100 -> +50 fg in; cell grew exactly 50
        out = step.update(_state(1050.0, -200))["listeners"]["mass"]
    assert out["conservation_residual"].to(units.fg).magnitude == pytest.approx(0.0)


def test_violation_warns():
    step = _make_step(tolerance=1.0e-2)
    step.update(_state(1000.0, -100))             # baseline
    with pytest.warns(UserWarning, match="cumulative mass-conservation drift"):
        # per-tick +50 in but cell grew only 10 -> residual -40
        out = step.update(_state(1010.0, -200))["listeners"]["mass"]
    assert out["conservation_residual"].to(units.fg).magnitude == pytest.approx(-40.0)
    assert out["conservation_residual_relative"] == pytest.approx(4.0)


def test_cumulative_exchange_is_diffed_not_summed():
    """A constant per-tick uptake shows up as a constant mass_in, even though
    the cumulative store keeps growing."""
    step = _make_step()
    step.update(_state(1000.0, -100))             # baseline, per-tick -100
    out2 = step.update(_state(1050.0, -200))["listeners"]["mass"]   # per-tick -100
    out3 = step.update(_state(1100.0, -300))["listeners"]["mass"]   # per-tick -100
    assert out2["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(50.0)
    assert out3["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(50.0)


def test_secretion_sign():
    step = _make_step()
    step.update(_state(1000.0, 0))                # baseline
    # cumulative +100 secreted -> per-tick +100 -> mass out 50; cell falls 50
    out = step.update(_state(950.0, 100))["listeners"]["mass"]
    assert out["exchange_mass_in"].to(units.fg).magnitude == pytest.approx(-50.0)
    assert out["conservation_residual"].to(units.fg).magnitude == pytest.approx(0.0)
