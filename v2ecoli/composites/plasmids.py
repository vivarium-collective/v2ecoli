"""Plasmids composite generator — baseline whole-cell model + plasmid replication.

Architecturally identical to the ``baseline`` generator; the only difference is
the ParCa cache it loads. A plasmid-enabled cache (built by
``scripts/build_plasmid_cache.py`` with ``has_plasmid=True``) carries the
``ecoli-plasmid-replication`` config, so the ``ecoli-plasmid-replication``
requester/evolver layers that already sit in baseline's execution flow get
instantiated instead of pruned (baseline drops them when the config is absent —
see ``baseline.py``'s ``next_update_time`` note). The plasmid (ColE1 / pBR322)
then replicates independently of the chromosome under Brendel & Perelson 1993
copy-number control.

Thin registration shim — the document body is baseline's. See
``v2ecoli/composites/baseline.py`` and
``v2ecoli/processes/plasmid_replication.py``.
"""
from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS
from v2ecoli.composites.baseline import baseline


@composite_generator(
    name="plasmids",
    description=(
        "Baseline whole-cell E. coli model with an independently-replicating "
        "plasmid (ColE1 / pBR322) under Brendel & Perelson 1993 copy-number "
        "control. Requires a plasmid-enabled cache built with has_plasmid=True."
    ),
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache_plasmid",
            "description": (
                "Path to a plasmid-enabled ParCa cache "
                "(scripts/build_plasmid_cache.py, has_plasmid=True)."
            ),
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
def plasmids(core: Any = None, *, seed: int = 0,
             cache_dir: str = "out/cache_plasmid",
             bundle: dict | None = None) -> dict:
    """Build the plasmid-enabled whole-cell composite document.

    Delegates to :func:`v2ecoli.composites.baseline.baseline` — the plasmid
    process is wired in automatically when ``cache_dir`` points at a cache
    that contains the ``ecoli-plasmid-replication`` config.

    Args:
        core: bigraph-schema core. If None, baseline builds one via build_core().
        seed: Random seed for stochastic initialisation.
        cache_dir: Path to a plasmid-enabled ParCa cache directory.
        bundle: Optional pre-loaded cache bundle (passed through to baseline,
            e.g. for multiseed runners that mutate the bundle in place).

    Returns:
        Process-bigraph document dict (``state``, ``skip_initial_steps``,
        ``sequential_steps``, ``flow_order``) — same shape as baseline.
    """
    return baseline(core, seed=seed, cache_dir=cache_dir, bundle=bundle)
