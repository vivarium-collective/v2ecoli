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

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.colony import make_colony_document


# Canonical visualizations for the multi-cell colony composite. ColonyVisualization
# is the integrated colony report (per-cell positions over time + colony total
# mass + cell-count trajectory). Standalone TimeSeriesPlots aren't appropriate
# here — colony state is per-cell, so the canonical view is the whole-colony
# report. Topology is included for inspecting the colony's process wiring.
DEFAULT_COLONY_VISUALIZATIONS = [
    {
        'name': 'colony-report',
        'address': 'local:ColonyVisualization',
        'config': {'title': 'v2ecoli colony — agent positions + mass over time'},
    },
    {
        'name': 'topology',
        'address': 'local:NetworkVisualization',
        'config': {'title': 'Colony composite topology'},
    },
]


@composite_generator(
    name="colony",
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
