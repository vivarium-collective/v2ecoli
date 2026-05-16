"""Shared constants for the polypeptide elongation submodules."""

from v2ecoli.types.quantity import ureg as units


MICROMOLAR_UNITS = units.umol / units.L
"""Units used for all concentrations in polypeptide elongation math."""

REMOVED_FROM_CHARGING = {"L-SELENOCYSTEINE[c]"}
"""Amino acids to remove from charging when running with ``steady_state_trna_charging``."""
