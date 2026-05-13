"""Visualization Steps for v2ecoli architectures.

Importing this package forces each per-Step module to load, which fires
their bigraph-schema link-registry side-effects via Step subclass
discovery. After ``import v2ecoli``, all Visualization Steps are
auto-registered in any ``allocate_core()``'s ``link_registry``.

Steps are added one at a time during the port; this file's imports
populate as each Step lands.
"""

# Step imports populate as Tasks 2-8 land each Visualization.
# Example (added during Task 2):
#   from v2ecoli.visualizations import network  # noqa: F401

__all__: list[str] = []
