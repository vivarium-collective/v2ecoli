"""
Units module for v2ecoli — pint-backed with unum-compatible API.

Provides the same interface as wholecell.utils.units so existing code
works without modification. Uses pint instead of unum.

Usage:
    from v2ecoli.library.units import units
    mass = 380 * units.fg
    mass_in_grams = mass.asNumber(units.g)
"""

import scipy.constants
import numpy as np
import pint

# ---------------------------------------------------------------------------
# Registry and custom units
# ---------------------------------------------------------------------------

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Custom biology units
ureg.define('count = mol / {}'.format(scipy.constants.Avogadro))
ureg.define('nucleotide = count = nt')
ureg.define('amino_acid = count = aa')

# Unum had 'dmol' (decimole) — pint doesn't define it by default
ureg.define('decimole = 0.1 * mole = dmol')


# ---------------------------------------------------------------------------
# Monkey-patch Quantity for unum compatibility
# ---------------------------------------------------------------------------

def _asNumber(self, target_unit=None):
    """Convert to a plain number, optionally in target units.

    Matches unum's .asNumber(unit) API.
    """
    if target_unit is None:
        return self.magnitude
    if isinstance(target_unit, pint.Unit):
        return self.to(target_unit).magnitude
    # target_unit might be a Quantity with magnitude 1
    if isinstance(target_unit, pint.Quantity):
        return self.to(target_unit.units).magnitude
    return self.magnitude


def _asUnit(self, target_unit):
    """Convert to target units. Matches unum's .asUnit(unit) API."""
    if isinstance(target_unit, pint.Quantity):
        return self.to(target_unit.units)
    return self.to(target_unit)


def _checkNoUnit(self):
    """Assert this quantity is dimensionless."""
    if not self.dimensionless:
        raise ValueError(f"Expected dimensionless, got {self.units}")


def _normalize(self):
    """No-op for pint compatibility (unum uses this internally)."""
    pass


# Patch onto Quantity class
Q_.asNumber = _asNumber
Q_.asUnit = _asUnit
Q_.checkNoUnit = _checkNoUnit
Q_.normalize = _normalize


# ---------------------------------------------------------------------------
# Unit module — provides attribute-style access (units.fg, units.mol, etc.)
# ---------------------------------------------------------------------------

class _Units:
    """Attribute-based unit access with numpy helper functions."""

    # Mass
    g = ureg.gram
    kg = ureg.kilogram
    mg = ureg.milligram
    fg = ureg.femtogram
    pg = ureg.picogram
    ng = ureg.nanogram
    ug = ureg.microgram
    micrograms = ureg.microgram  # alias
    Da = ureg.dalton

    # Length
    m = ureg.meter
    cm = ureg.centimeter
    mm = ureg.millimeter
    um = ureg.micrometer
    nm = ureg.nanometer

    # Volume
    L = ureg.liter
    mL = ureg.milliliter
    uL = ureg.microliter
    fL = ureg.femtoliter

    # Amount
    mol = ureg.mole
    mmol = ureg.millimole
    umol = ureg.micromole
    nmol = ureg.nanomole
    dmol = ureg.decimole

    # Time
    s = ureg.second
    sec = ureg.second  # alias
    min = ureg.minute
    h = ureg.hour

    # Concentration
    M = ureg.molar
    mM = ureg.millimolar
    uM = ureg.micromolar

    # Energy / temperature / electrical
    J = ureg.joule
    K = ureg.kelvin
    C = ureg.coulomb
    volt = ureg.volt

    # Custom biology
    count = ureg.count
    nt = ureg.nucleotide
    aa = ureg.amino_acid

    # Type alias for compatibility (replaces unum.Unum)
    Unum = Q_

    # -----------------------------------------------------------------------
    # Numpy wrapper functions (match wholecell.utils.units API)
    # -----------------------------------------------------------------------

    @staticmethod
    def hasUnit(value):
        return isinstance(value, Q_)

    @staticmethod
    def getUnit(value):
        if not isinstance(value, Q_):
            raise ValueError("Only works on Quantity!")
        return Q_(1, value.units)

    @staticmethod
    def strip_empty_units(value):
        if isinstance(value, Q_):
            if value.dimensionless:
                return value.magnitude
            raise ValueError(f"Expected dimensionless, got {value.units}")
        return value

    @staticmethod
    def dot(a, b, out=None):
        a_units = 1
        if isinstance(a, Q_):
            a_units = Q_(1, a.units)
            a = a.magnitude
        b_units = 1
        if isinstance(b, Q_):
            b_units = Q_(1, b.units)
            b = b.magnitude
        result = np.dot(a, b, out)
        return a_units * b_units * result

    @staticmethod
    def matmul(a, b, out=None):
        a_units = 1
        if isinstance(a, Q_):
            a_units = Q_(1, a.units)
            a = a.magnitude
        b_units = 1
        if isinstance(b, Q_):
            b_units = Q_(1, b.units)
            b = b.magnitude
        return a_units * b_units * np.matmul(a, b, out)

    @staticmethod
    def multiply(a, b):
        a_units = 1
        if isinstance(a, Q_):
            a_units = Q_(1, a.units)
            a = a.magnitude
        b_units = 1
        if isinstance(b, Q_):
            b_units = Q_(1, b.units)
            b = b.magnitude
        return a_units * b_units * np.multiply(a, b)

    @staticmethod
    def divide(a, b):
        a_units = 1
        if isinstance(a, Q_):
            a_units = Q_(1, a.units)
            a = a.magnitude
        b_units = 1
        if isinstance(b, Q_):
            b_units = Q_(1, b.units)
            b = b.magnitude
        return a_units / b_units * np.divide(a, b)

    @staticmethod
    def sum(array, axis=None, dtype=None, out=None, keepdims=False):
        if not isinstance(array, Q_):
            raise ValueError("Only works on Quantity!")
        u = Q_(1, array.units)
        return u * np.sum(array.magnitude, axis, dtype, out, keepdims)

    @staticmethod
    def abs(array):
        if not isinstance(array, Q_):
            raise ValueError("Only works on Quantity!")
        u = Q_(1, array.units)
        return u * np.abs(array.magnitude)

    @staticmethod
    def floor(x):
        if not isinstance(x, Q_):
            raise ValueError("Only works on Quantity!")
        u = Q_(1, x.units)
        return u * np.floor(x.magnitude)

    @staticmethod
    def transpose(array, axis=None):
        u = Q_(1, array.units)
        return u * np.transpose(array.magnitude, axis)

    @staticmethod
    def hstack(tup):
        u = Q_(1, tup[0].units)
        values = []
        for arr in tup:
            if not isinstance(arr, Q_):
                raise ValueError("Only works on Quantity!")
            values.append(arr.to(u.units).magnitude)
        return u * np.hstack(tuple(values))

    @staticmethod
    def isnan(value):
        if isinstance(value, Q_):
            return np.isnan(value.magnitude)
        return np.isnan(value)

    @staticmethod
    def isfinite(value):
        if isinstance(value, Q_):
            return np.isfinite(value.magnitude)
        return np.isfinite(value)


units = _Units()


# ---------------------------------------------------------------------------
# Monkey-patch unum to accept pint units in asNumber/asUnit (transition)
# ---------------------------------------------------------------------------

try:
    from unum import Unum as _UnumType

    _orig_asNumber = _UnumType.asNumber

    def _patched_asNumber(self, target=None):
        """asNumber that accepts both unum and pint target units."""
        if target is None:
            return _orig_asNumber(self)
        if isinstance(target, (pint.Unit, pint.Quantity)):
            # Convert self to pint first, then extract magnitude
            converted = convert_unum_to_pint(self)
            if isinstance(converted, Q_):
                if isinstance(target, pint.Quantity):
                    return converted.to(target.units).magnitude
                return converted.to(target).magnitude
            return converted  # dimensionless
        return _orig_asNumber(self, target)

    _UnumType.asNumber = _patched_asNumber
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Unum → Pint conversion (for loading legacy caches)
# ---------------------------------------------------------------------------

def convert_unum_to_pint(obj, depth=0):
    """Recursively convert unum.Unum objects to pint Quantities.

    Used when loading cached configs that were pickled with unum.
    """
    if depth > 10:
        return obj

    try:
        from unum import Unum as UnumType
    except ImportError:
        return obj  # unum not installed, nothing to convert

    if isinstance(obj, UnumType):
        # Extract value and unit string from unum
        # unum._unit is a dict of {str: int} mapping unit names to exponents
        value = obj._value
        unit_dict = obj._unit
        if not unit_dict:
            return value  # dimensionless
        # Build pint unit from unum unit components
        UNIT_MAP = {
            'fg': 'femtogram', 'g': 'gram', 'kg': 'kilogram',
            'mg': 'milligram', 'ug': 'microgram', 'ng': 'nanogram',
            'pg': 'picogram', 'micrograms': 'microgram',
            'L': 'liter', 'mL': 'milliliter', 'fL': 'femtoliter',
            'mol': 'mole', 'mmol': 'millimole', 'umol': 'micromole',
            'nmol': 'nanomole', 'dmol': 'decimole',
            's': 'second', 'sec': 'second', 'min': 'minute', 'h': 'hour',
            'J': 'joule', 'K': 'kelvin', 'C': 'coulomb', 'V': 'volt',
            'm': 'meter', 'cm': 'centimeter', 'mm': 'millimeter',
            'um': 'micrometer', 'nm': 'nanometer',
            'count': 'count', 'nucleotide': 'nucleotide',
            'amino_acid': 'amino_acid',
        }
        pint_unit = ureg.dimensionless
        for name, exp in unit_dict.items():
            pint_name = UNIT_MAP.get(str(name), str(name))
            try:
                pint_unit = pint_unit * getattr(ureg, pint_name) ** exp
            except Exception:
                return value  # unknown unit — return raw number
        return Q_(value, pint_unit)

    elif isinstance(obj, dict):
        return {k: convert_unum_to_pint(v, depth + 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_unum_to_pint(v, depth + 1) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_unum_to_pint(v, depth + 1) for v in obj)

    return obj
