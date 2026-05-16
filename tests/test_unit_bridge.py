"""Round-trip tests for the Unum<->pint bridge."""

import numpy as np
import pint
import pytest
from unum import Unum
from wholecell.utils import units as wc_units

from v2ecoli.types.quantity import ureg
from v2ecoli.library.unit_bridge import pint_to_unum, unum_to_pint


def _close(a, b, rtol=1e-12):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol)


# (Unum-builder, pint-string) cases that should round-trip identically.
CASES = [
    (lambda: 5.0 * wc_units.mmol / wc_units.L, "mmol / L"),
    (lambda: 250.0 * wc_units.fg, "fg"),
    (lambda: 1.5 * wc_units.g / wc_units.L, "g / L"),
    (lambda: 0.2 * (1 / wc_units.s), "1 / s"),
    (lambda: 6.022e23 * (1 / wc_units.mol), "1 / mol"),
    (lambda: 12.0 * wc_units.count, "count"),
    (lambda: Unum({"nucleotide": 1}, 21.0), "nucleotide"),
    (lambda: 17.0 * wc_units.aa / wc_units.s, "amino_acid / s"),
    (lambda: 0.001 * wc_units.fg / wc_units.mol, "fg / mol"),
]


@pytest.mark.parametrize("make_unum,pint_str", CASES)
def test_unum_to_pint_then_back(make_unum, pint_str):
    u = make_unum()
    q = unum_to_pint(u)
    assert isinstance(q, pint.Quantity)
    # value preserved
    assert _close(q.magnitude, u._value)
    # round-trip through unit string
    back = pint_to_unum(q)
    assert isinstance(back, Unum)
    # numeric agreement after dividing by the original unit
    diff = (back / u).asNumber()
    assert _close(diff, 1.0)


@pytest.mark.parametrize("make_unum,pint_str", CASES)
def test_pint_with_target_unum(make_unum, pint_str):
    u = make_unum()
    q = ureg.Quantity(u._value, pint_str)
    target_unit = make_unum() / Unum(make_unum()._unit, make_unum()._value)
    # build target from a unit-only Unum (value 1)
    target_only = Unum(dict(make_unum()._unit), 1.0)
    back = pint_to_unum(q, target=target_only)
    assert isinstance(back, Unum)
    assert _close((back / u).asNumber(), 1.0)


def test_unum_array_preserves_shape_and_values():
    arr = np.array([1.0, 2.0, 3.0])
    u = wc_units.mmol * arr / wc_units.L
    q = unum_to_pint(u)
    assert q.units == ureg.Unit("mmol / L")
    assert _close(q.magnitude, arr)
    back = pint_to_unum(q)
    assert _close(back.asNumber(wc_units.mmol / wc_units.L), arr)


def test_passthrough_for_non_unum_and_non_pint():
    assert unum_to_pint(3.5) == 3.5
    assert pint_to_unum("hello") == "hello"


def test_compound_units_mol_per_fg_min():
    u = 1e-9 * wc_units.mol / wc_units.fg / wc_units.min
    q = unum_to_pint(u)
    assert _close(q.magnitude, 1e-9)
    # round-trip
    back = pint_to_unum(q)
    assert _close((back / u).asNumber(), 1.0)


def test_compound_units_mmol_per_g_h():
    u = 4.2 * wc_units.mmol / wc_units.g / wc_units.h
    q = unum_to_pint(u)
    assert _close(q.magnitude, 4.2)
    back = pint_to_unum(q)
    assert _close((back / u).asNumber(), 1.0)
