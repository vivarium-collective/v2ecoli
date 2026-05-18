"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``pbg_superpowers.composite_generator._REGISTRY``.
"""

from v2ecoli.composites import baseline, colony, departitioned, reconciled  # noqa: F401
from v2ecoli.composites import baseline_recipes  # noqa: F401  — registers dnaa-* gate recipes

__all__ = ["baseline", "colony", "departitioned", "reconciled", "baseline_recipes"]
