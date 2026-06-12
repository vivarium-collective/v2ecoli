"""Visualization Steps for v2ecoli architectures.

Importing this package forces each per-Step module to load, which fires
their bigraph-schema link-registry side-effects via Step subclass
discovery. After ``import v2ecoli``, all Visualization Steps are
auto-registered in any ``allocate_core()``'s ``link_registry``.
"""

from v2ecoli.visualizations import (
    network, benchmark, v1_v2, multigeneration, colony, workflow,
)  # noqa: F401

__all__: list[str] = [
    "network", "benchmark", "v1_v2", "multigeneration", "colony", "workflow",
]

# Register the v2ecoli units resolver onto the shared Visualization base so
# every v2ecoli visualization labels axes from the declared port schema.
# Guard with getattr: an installed pbg_superpowers that predates the units
# hook (no ``units_resolver`` attribute) degrades to no axis-unit labeling
# rather than crashing every visualization import.
from pbg_superpowers.visualization import Visualization as _Visualization
from v2ecoli.library.units_resolver import V2EcoliUnitsResolver as _V2EcoliUnitsResolver

if hasattr(_Visualization, "units_resolver") and getattr(
    _Visualization, "units_resolver", None
) is None:
    _Visualization.units_resolver = _V2EcoliUnitsResolver()
