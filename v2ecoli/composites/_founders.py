"""Multi-founder documents: N founder lineages sharing one environment.

A single-founder document holds one cell at ``agents['0']``. A multi-founder
document holds N cells at ``agents[founder_ids(N)[k]]``, each drawn from its
OWN ParCa founder draw (``baseline(independent_founders=True)``, seeded
``seed + k``), and declares ``lineage.founder_id_length`` so the
LineageBookkeeper, PopulationAggregator and ReactorCellCoupler treat each
founder as its own lineage (see ``v2ecoli/steps/population_aggregator.py``).

The only id-bearing part of a cell is its Division step (``agent_id`` in both
its config and its live instance), which names the daughters it creates. Each
founder's Division is re-pointed at the founder's id -- the same device
``Division`` itself uses when it builds daughters -- so founder ``"3"``
divides into ``"30"``/``"31"`` and never collides with another lineage.

Founders' in-document emitters are built ``null``: with N cells, N per-agent
sinks would all write the same partition. The multigen runner owns emission.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.population_aggregator import (
    LINEAGE_FOUNDER_ID_LENGTH_KEY,
    LINEAGE_STORE_NAME,
    founder_ids,
)


def repoint_division(cell_state: dict, agent_id: str) -> None:
    """Point a cell's Division step at ``agent_id`` (config and live instance)."""
    div_edge = cell_state.get("division")
    if not isinstance(div_edge, dict):
        raise ValueError("cell has no 'division' step to re-point")
    if isinstance(div_edge.get("config"), dict):
        div_edge["config"]["agent_id"] = agent_id
    inst = div_edge.get("instance")
    if inst is not None:
        inst.agent_id = agent_id


def add_founders(
    document: dict,
    core: Any,
    *,
    n_founders: int,
    seed: int,
    founder_sim_data: str,
    cache_dir: str,
    config_overrides: dict | None = None,
    injected_processes: dict | None = None,
) -> dict:
    """Replace a single-founder document's cell with ``n_founders`` distinct founders.

    ``document`` must hold exactly one agent (the single-founder build) and a
    ``lineage`` store (``add_population_aggregator`` seeds one). Founder ``k`` is
    ``baseline(seed=seed + k, independent_founders=True)``, so every founder --
    including the first -- is its own ParCa draw rather than the shared cached
    cell. ``cells_per_agent`` is untouched: it stays per agent, and a caller that
    wants the same inoculum at any N divides it by N.
    """
    from v2ecoli.composites.ecoli_baseline import baseline

    if n_founders < 2:
        raise ValueError(f"add_founders needs n_founders >= 2, got {n_founders}")
    if not founder_sim_data:
        raise ValueError(
            "n_founders > 1 needs founder_sim_data: each founder is its own ParCa "
            "draw. Founders copied from the one cached cell would be synchronised "
            "clones, which is the case the feature exists to avoid.")
    state = document["state"]
    agents = state["agents"]
    if len(agents) != 1:
        raise ValueError(f"expected a single-founder document, got agents {sorted(agents)}")
    lineage = state.get(LINEAGE_STORE_NAME)
    if not isinstance(lineage, dict):
        raise ValueError("document has no lineage store; build it with add_population_aggregator")

    ids = founder_ids(n_founders)
    founders: dict[str, dict] = {}
    for k, fid in enumerate(ids):
        doc_k = baseline(
            core=core, seed=seed + k, cache_dir=cache_dir,
            config_overrides=config_overrides,
            injected_processes=injected_processes,
            independent_founders=True, founder_sim_data=founder_sim_data,
            emitter="null",
        )
        cell = doc_k["state"]["agents"]["0"]
        repoint_division(cell, fid)
        founders[fid] = cell

    agents.clear()
    agents.update(founders)
    lineage[LINEAGE_FOUNDER_ID_LENGTH_KEY] = float(len(ids[0]))
    return document
