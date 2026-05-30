"""Tolerant accessors for the units-on-ports migration.

During the migration, a dimensioned port may carry either a bare number
(legacy ``float[unit]`` metadata, where the magnitude is a plain float and
the unit lives only on the schema) or a real ``pint.Quantity`` (the
``quantity[...]`` type, whose runtime value carries its unit). Consumers that
must keep working across the flip should read through these helpers rather
than assuming one representation.

``magnitude_in(x, unit)`` and ``fg_magnitude(x)`` are pure pass-throughs for
bare numbers (``fg_magnitude(1234.5) == 1234.5``), so adopting them changes
nothing while ports are still bare floats — and Just Works once the same port
becomes a ``Quantity``.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.types.quantity import ureg as units


def _is_quantity(value: Any) -> bool:
    """Duck-type a pint.Quantity (has magnitude + units) without importing pint."""
    return hasattr(value, "magnitude") and hasattr(value, "units")


def magnitude_in(value: Any, unit: Any) -> Any:
    """Magnitude of ``value`` expressed in ``unit``.

    Accepts a bare number (assumed already in ``unit``) or a pint.Quantity
    (converted to ``unit`` first). Bare numbers pass through unchanged.
    """
    if _is_quantity(value):
        return value.to(unit).magnitude
    return value


def as_quantity(value: Any, unit: Any):
    """Return ``value`` as a pint.Quantity in ``unit``.

    Accepts a bare number (assumed in ``unit``, wrapped) or an existing
    Quantity (converted to ``unit``).
    """
    if _is_quantity(value):
        return value.to(unit)
    return value * unit


def fg_magnitude(value: Any) -> float:
    """Femtogram magnitude as a plain float, for a mass that may be a bare
    ``float[fg]`` value or a pint.Quantity[fg]."""
    return float(magnitude_in(value, units.fg))
