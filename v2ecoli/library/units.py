"""Units module — re-exports from wholecell.utils.units with pint-style aliases."""

from unum import Unum
from wholecell.utils import units  # noqa: F401

# Add pint-style unit aliases that some modules expect
_ALIASES = {
    'sec': lambda: units.s,
    'mL': lambda: units.L * 1e-3,
    'fL': lambda: units.L * 1e-15,
    'micrograms': lambda: units.g * 1e-6,
    'uM': lambda: units.umol / units.L,
    'mM': lambda: units.mmol / units.L,
    'nM': lambda: units.nmol / units.L,
    'volt': lambda: units.J / units.C if hasattr(units, 'C') else 1.0,
}

for _name, _factory in _ALIASES.items():
    if not hasattr(units, _name):
        try:
            setattr(units, _name, _factory())
        except Exception:
            pass


# Add pint-style .to() and .magnitude as aliases for Unum's .asUnit()/.asNumber()
if not hasattr(Unum, 'to'):
    def _unum_to(self, target_unit):
        """Pint-style .to() — convert to target unit."""
        if isinstance(target_unit, str):
            target_unit = getattr(units, target_unit, None)
            if target_unit is None:
                return self
        return self.asUnit(target_unit)
    Unum.to = _unum_to

if not hasattr(Unum, 'magnitude'):
    @property
    def _unum_magnitude(self):
        """Pint-style .magnitude — return the numeric value."""
        return self.asNumber()
    Unum.magnitude = _unum_magnitude
