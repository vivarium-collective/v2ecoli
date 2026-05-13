"""Visualization Steps for v2ecoli architectures.

Importing this package forces each per-Step module to load, which fires
their bigraph-schema link-registry side-effects via Step subclass
discovery. After ``import v2ecoli``, all Visualization Steps are
auto-registered in any ``allocate_core()``'s ``link_registry``.
"""

from v2ecoli.visualizations import network, benchmark, v1_v2  # noqa: F401

__all__: list[str] = ["network", "benchmark", "v1_v2"]
