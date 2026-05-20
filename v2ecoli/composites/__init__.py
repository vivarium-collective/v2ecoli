"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``pbg_superpowers.composite_generator._REGISTRY``.
"""

from v2ecoli.composites import baseline, colony, departitioned, plasmids, reconciled  # noqa: F401

__all__ = ["baseline", "colony", "departitioned", "plasmids", "reconciled"]
