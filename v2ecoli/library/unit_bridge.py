"""Unum <-> pint translation at the boundary between v2ecoli (pint) and
upstream vEcoli/wholecell (Unum). All call sites that cross into upstream
Unum-native code must convert here. The v2ecoli internal codebase should
never touch Unum directly outside this module.
"""

from __future__ import annotations
from typing import Optional, Any

import numpy as np
import pint
from unum import Unum
from wholecell.utils import units as wc_units

from v2ecoli.types.quantity import ureg

# Register bio-specific units that pint doesn't know about. The upstream
# Unum library treats these as named base units; we mirror that as
# dimensionless-like base units in pint so unit algebra closes (e.g.
# nucleotide / s remains nucleotide/s).
for _name, _aliases in [
    ("nucleotide", ("nt",)),
    ("amino_acid", ("aa",)),
    ("count", ()),
]:
    if _name not in ureg:
        ureg.define(f"{_name} = [{_name}]" + (" = " + " = ".join(_aliases) if _aliases else ""))


def _unit_string_from_unum(u: Unum) -> str:
    """Build a pint-parseable unit expression from a Unum's _unit dict."""
    parts = []
    for name, exp in u._unit.items():
        parts.append(f"{name}**{exp}" if exp != 1 else name)
    return " * ".join(parts) if parts else "dimensionless"


def unum_to_pint(u: Any) -> Any:
    """Convert an Unum quantity (scalar or ndarray-valued) to a pint Quantity
    on the shared v2ecoli registry. Non-Unum inputs are returned unchanged."""
    if not isinstance(u, Unum):
        return u
    magnitude = u._value
    if not u._unit:
        return ureg.Quantity(magnitude)
    unit_expr = _unit_string_from_unum(u)
    return ureg.Quantity(magnitude, unit_expr)


# Pint short-symbol -> wholecell.utils.units name overrides for symbols that
# wc_units doesn't expose by abbreviation (or where casing differs).
_PINT_SYMBOL_TO_WC = {
    "l": "L",
    "amino_acid": "aa",
}


def _resolve_wc_unit(symbol: str) -> Unum:
    """Find the wholecell.utils.units Unum that corresponds to a pint short
    symbol. Falls back to constructing a bare Unum for names Unum registers
    globally but wc_units doesn't re-export (e.g. 'nucleotide')."""
    name = _PINT_SYMBOL_TO_WC.get(symbol, symbol)
    wc = getattr(wc_units, name, None)
    if isinstance(wc, Unum):
        return wc
    return Unum({symbol: 1}, 1.0)


def _unum_unit_from_pint(q: pint.Quantity) -> Unum:
    """Build an Unum unit object matching a pint Quantity's dimensionality
    using pint short symbols (e.g. 'mmol' rather than 'millimole')."""
    unum_unit = Unum({}, 1.0)
    for name, exp in q.units._units.items():
        symbol = f"{ureg.Unit(name):~}"
        unum_unit = unum_unit * (_resolve_wc_unit(symbol) ** exp)
    return unum_unit


def pint_to_unum(q: Any, target: Optional[Unum] = None) -> Any:
    """Convert a pint Quantity to an Unum quantity. If target is provided
    (an Unum unit-only object, e.g. units.mmol/units.L), the pint quantity
    is converted into that unit first; otherwise the pint unit names are
    mapped 1:1 to wholecell.utils.units. Non-pint inputs are returned
    unchanged. Bare pint Unit objects are promoted to a 1.0-valued Quantity
    so the unit alone can be translated."""
    if isinstance(q, pint.Unit):
        q = ureg.Quantity(1.0, q)
    if not isinstance(q, pint.Quantity):
        return q
    if target is not None:
        target_pint_str = _unit_string_from_unum(target)
        magnitude = q.to(target_pint_str).magnitude
        return magnitude * target
    if q.dimensionless and not q.units._units:
        return Unum({}, q.magnitude)
    # Unum-on-the-left so numpy ndarray magnitudes don't take operator
    # precedence and strip the unit.
    return _unum_unit_from_pint(q) * q.magnitude
