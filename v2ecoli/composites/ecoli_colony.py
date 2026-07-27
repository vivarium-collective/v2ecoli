"""Colony composite generator — multi-cell E. coli in pymunk 2D physics.

Exposes ``v2ecoli.colony.make_colony_document`` as a process-bigraph
``@composite_generator`` so the colony shows up alongside ``baseline``
in workspace catalogs / dashboards.

Each cell embeds the full whole-cell model via the ``EcoliWCM`` bridge; a
``PymunkProcess`` drives 2D physics. See ``v2ecoli/colony.py`` for the
document body — this module is a thin registration shim.
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.colony import make_colony_document


def _register_colony_core(core=None):
    """Return a core with everything the colony composite needs registered:
    viva_munk base + pymunk types (notably ``pymunk_agent``), v2ecoli
    ``ECOLI_TYPES``, and the ``EcoliWCM`` link. Declared as a ``core_extension``
    so tooling can INSTANTIATE the colony on a proper core — notably the loom
    Explorer, which instantiates it to drill into / preview a cell's inner model.
    Builds a FRESH ``core_import()`` core (viva_munk's base) rather than mutating
    the passed ``allocate_core()`` — whose base types conflict with viva_munk's
    (e.g. ``positive_float``). ``apply_core_extensions`` captures this return."""
    from viva_munk import core_import
    from v2ecoli.bridge import EcoliWCM
    from v2ecoli.types import ECOLI_TYPES

    core = core_import()
    core.register_types(ECOLI_TYPES)
    core.register_link("EcoliWCM", EcoliWCM)
    return core


# Visualizations for the multi-cell colony composite.
#
# The colony declares NO canonical visualizations, so the run's analysis flush
# falls back to the standard `_render_default_viz` → `TimeSeriesFromObservables`,
# which reads this run's emitter output and plots every numeric observable
# (per-cell mass, length, position, colony totals) over time — the same
# "standard flush visualization of simulation outputs" every other composite
# gets. The former `ColonyVisualization` needed a flat `history` list the run
# never supplies (so it rendered all "?"), and the bigraph `topology` view added
# no simulation-output insight; both are dropped. Richer colony-specific views
# (spatial GIF, per-cell traces) are being added as post-hoc analyses that read
# the emitter output in the flush — not per-tick Steps.
DEFAULT_COLONY_VISUALIZATIONS: list = []


@composite_generator(
    name="ecoli_colony",
    description=(
        "Multi-cell colony — whole-cell E. coli agents embedded in a "
        "pymunk 2D physics environment via the EcoliWCM bridge."
    ),
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "Base RNG seed; per-cell seed is offset by cell index.",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to the ParCa cache directory.",
        },
        "n_cells": {
            "type": "integer",
            "default": 2,
            "description": "Number of initial cells in the colony.",
        },
        "env_size": {
            "type": "number",
            "default": 30,
            "description": "Side length of the 2D environment (micrometers).",
        },
        "physics_interval": {
            "type": "number",
            "default": 1.0,
            "description": "Seconds between PymunkProcess updates.",
        },
        "ecoli_interval": {
            "type": "number",
            "default": 1.0,
            "description": "Seconds between per-cell EcoliWCM updates.",
        },
    },
    visualizations=DEFAULT_COLONY_VISUALIZATIONS,
    core_extensions=[_register_colony_core],
)
def colony(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    n_cells: int = 2,
    env_size: float = 30,
    physics_interval: float = 1.0,
    ecoli_interval: float = 1.0,
) -> dict:
    """Build the colony composite document.

    The colony requires both the ``viva_munk`` package (``PymunkProcess``,
    ``build_microbe``) and v2ecoli's ``EcoliWCM`` bridge. If ``core`` is not
    supplied, one is bootstrapped with both registries; otherwise the caller
    is responsible for having ``ECOLI_TYPES`` registered and ``EcoliWCM``
    linked.

    Args:
        core: bigraph-schema core. If None, builds a colony-ready core.
        seed: RNG seed for stochastic initialisation; per-cell seeds are
            ``seed + cell_index``.
        cache_dir: Path to the ParCa cache directory.
        n_cells: Number of cells in the initial colony.
        env_size: 2D environment edge length (micrometers).
        physics_interval: PymunkProcess step interval (seconds).
        ecoli_interval: Per-cell EcoliWCM step interval (seconds).

    Returns:
        Process-bigraph document dict with a single ``state`` key.
    """
    if core is None:
        from viva_munk import core_import
        from v2ecoli.bridge import EcoliWCM
        from v2ecoli.types import ECOLI_TYPES

        core = core_import()
        core.register_types(ECOLI_TYPES)
        core.register_link("EcoliWCM", EcoliWCM)

    doc = make_colony_document(
        n_cells=n_cells,
        env_size=env_size,
        physics_interval=physics_interval,
        ecoli_interval=ecoli_interval,
        cache_dir=cache_dir,
        seed=seed,
    )

    return {"state": doc}
