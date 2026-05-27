"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``pbg_superpowers.composite_generator._REGISTRY``.
"""

from v2ecoli.composites import (  # noqa: F401
    baseline,
    baseline_population,
    baseline_time_varying_env,
    colony,
    departitioned,
    reconciled,
)

__all__ = [
    "baseline",
    "baseline_population",
    "baseline_time_varying_env",
    "colony",
    "departitioned",
    "reconciled",
]
