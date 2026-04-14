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


def rebind_cache_quantities(obj, _seen=None):
    """Walk a loaded cache dict/list/tuple/ndarray and rebind every pint
    Quantity to the shared ureg. Use after `dill.load` on cache.dill to
    guard against stale-registry Quantities from cross-process unpickle
    or from side-effectful imports (e.g. ecoli.library.bigraph_types
    replacing pint.application_registry at import time)."""
    import numpy as np
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))

    def _rebind(q):
        if isinstance(q, pint.Quantity) and q._REGISTRY is not ureg:
            return ureg.Quantity(q.magnitude, str(q.units))
        return q

    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, pint.Quantity):
                obj[k] = _rebind(v)
            else:
                rebind_cache_quantities(v, _seen)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, pint.Quantity):
                obj[i] = _rebind(v)
            else:
                rebind_cache_quantities(v, _seen)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        flat = obj.ravel()
        for i in range(flat.shape[0]):
            v = flat[i]
            if isinstance(v, pint.Quantity):
                flat[i] = _rebind(v)
            else:
                rebind_cache_quantities(v, _seen)
    return obj


def _unit_string_from_unum(u: Unum) -> str:
    """Build a pint-parseable unit expression from a Unum's _unit dict."""
    parts = []
    for name, exp in u._unit.items():
        parts.append(f"{name}**{exp}" if exp != 1 else name)
    return " * ".join(parts) if parts else "dimensionless"


def unum_to_pint(u: Any) -> Any:
    """Convert an Unum quantity (scalar or ndarray-valued) to a pint Quantity
    on the shared v2ecoli registry. Non-Unum inputs are rebound to ureg if
    they are pint Quantities on a different registry, otherwise returned
    unchanged. Rebinding handles the case where a cache.dill round-trip —
    or a side-effectful import (e.g. ecoli.library.bigraph_types, which
    replaces pint.application_registry at import time) — leaves a
    Quantity tied to a stale UnitRegistry instance. Without this rebind,
    `rna_conc_molar / self.Kms` raises 'different registries' or silently
    produces garbage units, which was the root cause of the ~8x mRNA
    accumulation seen in daughter sims."""
    if isinstance(u, pint.Quantity):
        if u._REGISTRY is ureg:
            return u
        # Rebind to the shared registry via string parsing.
        return ureg.Quantity(u.magnitude, str(u.units))
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
    # pint uses unicode µ in short-symbol form; wholecell uses ASCII u.
    "µmol": "umol",
    "µg": "ug",
    "µL": "uL",
    "µm": "um",
    "µs": "us",
    "µM": "uM",
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
