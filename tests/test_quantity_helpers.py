"""Tolerant fg-accessor helpers for the units-on-ports migration.

These must be exact pass-throughs for bare numbers (so adopting them changes
nothing while ports are still float[fg]) and correct for pint Quantities (so
they Just Work once a port becomes quantity[...]).
"""

import numpy as np
import pytest

from v2ecoli.library.quantity_helpers import (
    as_quantity,
    fg_magnitude,
    magnitude_in,
)
from v2ecoli.types.quantity import ureg as units


def test_fg_magnitude_passthrough_for_bare_numbers():
    # Exact pass-through — behavior is identical to float(x) today.
    assert fg_magnitude(1234.5) == 1234.5
    assert fg_magnitude(0) == 0.0
    assert isinstance(fg_magnitude(1234.5), float)


def test_fg_magnitude_for_quantity():
    assert fg_magnitude(1234.5 * units.fg) == 1234.5
    # Converts to fg first (1 pg = 1000 fg, modulo float precision).
    assert fg_magnitude(1.0 * units.pg) == pytest.approx(1000.0)


def test_magnitude_in_units():
    assert magnitude_in(5.0, units.fg) == 5.0
    assert magnitude_in(5.0 * units.fg, units.fg) == 5.0
    assert magnitude_in((2.0 * units.amino_acid / units.s), units.aa / units.s) == 2.0


def test_as_quantity_wraps_and_converts():
    q = as_quantity(7.0, units.fg)
    assert q.magnitude == 7.0 and q.units == units.fg
    # already a Quantity -> converted, not double-wrapped
    q2 = as_quantity(2.0 * units.pg, units.fg)
    assert q2.to(units.fg).magnitude == pytest.approx(2000.0)


def test_fg_magnitude_handles_numpy_scalar():
    assert fg_magnitude(np.float64(42.0)) == 42.0
