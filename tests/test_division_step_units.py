"""Regression: the Division step must handle a pint Quantity[fg] dry_mass.

Under units-on-ports, listeners.mass.dry_mass is a pint Quantity[fg]. Before
the fix, Division.next_update did plain float arithmetic/comparison against it
and raised DimensionalityError every tick. process-bigraph swallows exceptions
raised inside a step's next_update, so division_threshold stayed stuck on the
string "mass_distribution" and the structural _add/_remove split never fired —
a bare composite.run() loop never divided.
"""
import numpy as np

from v2ecoli.steps.division import Division
from v2ecoli.types.quantity import ureg as units


def _make_division():
    d = Division.__new__(Division)
    d.initialize({"dry_mass_inc_dict": {"minimal": units.Quantity(150.0, "fg")}})
    return d


def test_threshold_init_handles_quantity_dry_mass():
    """The threshold-initialization branch must return a numeric threshold
    (not raise) when dry_mass is a pint Quantity[fg]."""
    d = _make_division()
    states = {
        "division_threshold": "mass_distribution",
        "media_id": "minimal",
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
    }
    update = d.next_update(1.0, states)
    thr = update["division_threshold"]
    assert isinstance(thr, float)          # raised DimensionalityError before the fix
    assert thr == 550.0                    # 400 (fg magnitude) + 150 * multiplier(=1)


def test_division_check_handles_quantity_dry_mass():
    """The division-check comparison (dry_mass < threshold) must not raise when
    dry_mass is a Quantity[fg] and threshold is a plain float."""
    d = _make_division()
    states = {
        "division_threshold": 760.0,       # already numeric -> check branch
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
        "unique": {"full_chromosome": None},
    }
    # dry_mass (400) < threshold (760): no division, and crucially no raise.
    assert d.next_update(1.0, states) == {}
