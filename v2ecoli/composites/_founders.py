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

Phase offsets (``founder_cycle_s = T``): founders started together divide in
lockstep, and a brief per-cell event then shows up in the population at its
full per-cell size. Founder ``k`` is instead PRE-ADVANCED -- run on its own, in
the baseline medium, for ``k * T / N`` seconds -- and carried into the shared
document at that phase (``overlay_cell_data``, the device Division uses for
daughters). Each founder's base weight follows an exponentially growing
culture's age distribution, ``n(a) ∝ 2**(1 - a)`` at phase ``a = k / N``,
normalised to a mean of 1 so the total represented cells stay
``n_founders * cells_per_agent``.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.population_aggregator import (
    LINEAGE_FOUNDER_ID_LENGTH_KEY,
    LINEAGE_STORE_NAME,
    founder_ids,
    founder_weight_key,
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


def phase_weights(n_founders: int) -> list[float]:
    """Base weight of founder ``k`` at phase ``k / N``: ``2**(1 - k/N)``, mean 1."""
    raw = [2.0 ** (1.0 - k / n_founders) for k in range(n_founders)]
    mean = sum(raw) / n_founders
    return [w / mean for w in raw]


def pre_advance_snapshot(
    core: Any, *, seed: int, seconds: int, founder_sim_data: str, cache_dir: str,
    config_overrides: dict | None = None, injected_processes: dict | None = None,
) -> dict:
    """Run one founder on its own for ``seconds`` and snapshot its cell state.

    The snapshot has :func:`divide_cell`'s shape (core stores, injected
    agent-root stores, ``_carried_listeners``), ready for ``overlay_cell_data``.
    Raises if the founder divided: the offset must be shorter than its cycle.
    """
    import copy

    from process_bigraph import Composite

    from v2ecoli.composites.ecoli_baseline import baseline
    from v2ecoli.library.division import collect_carried_listeners, extra_store_keys

    doc = baseline(
        core=core, seed=seed, cache_dir=cache_dir,
        config_overrides=config_overrides, injected_processes=injected_processes,
        independent_founders=True, founder_sim_data=founder_sim_data,
        emitter="null",
    )
    composite = Composite(doc, core=core)
    composite.run(int(seconds))
    agents = composite.state["agents"]
    if list(agents) != ["0"]:
        raise ValueError(
            f"founder (seed {seed}) divided during a {seconds}s pre-advance (agents "
            f"{sorted(agents)}); founder_cycle_s must be shorter than its cell cycle")
    cell = agents["0"]
    snapshot = {k: copy.deepcopy(cell[k])
                for k in ("bulk", "unique", "environment", "boundary") if k in cell}
    for key in extra_store_keys(cell):
        snapshot[key] = copy.deepcopy(cell[key])
    carried = collect_carried_listeners(cell)
    if carried:
        snapshot["_carried_listeners"] = carried
    return snapshot


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
    founder_cycle_s: float = 0.0,
    weighted: bool = True,
) -> dict:
    """Replace a single-founder document's cell with ``n_founders`` distinct founders.

    ``document`` must hold exactly one agent (the single-founder build) and a
    ``lineage`` store (``add_population_aggregator`` seeds one). Founder ``k`` is
    ``baseline(seed=seed + k, independent_founders=True)``, so every founder --
    including the first -- is its own ParCa draw rather than the shared cached
    cell. ``cells_per_agent`` is untouched: it stays per agent, and a caller that
    wants the same inoculum at any N divides it by N.

    ``weighted=False`` (the literal mode) leaves ``lineage`` undeclared, so the
    founders and every daughter they produce are plain ``cells_per_agent``
    agents summed uniformly -- the reference the weighted mode is checked against.

    ``founder_cycle_s > 0`` spreads the founders across one cell cycle (see the
    module docstring): founder ``k`` is pre-advanced ``k * founder_cycle_s / N``
    seconds and weighted by :func:`phase_weights`.
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

    if founder_cycle_s < 0:
        raise ValueError(f"founder_cycle_s must be >= 0, got {founder_cycle_s}")
    if founder_cycle_s > 0 and not weighted:
        raise ValueError("founder_cycle_s needs weighted=True (phase weights)")
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
        offset = int(round(k * founder_cycle_s / n_founders))
        if offset > 0:
            from v2ecoli.library.division import overlay_cell_data
            snapshot = pre_advance_snapshot(
                core, seed=seed + k, seconds=offset,
                founder_sim_data=founder_sim_data, cache_dir=cache_dir,
                config_overrides=config_overrides,
                injected_processes=injected_processes)
            overlay_cell_data(cell, snapshot, core)
        repoint_division(cell, fid)
        founders[fid] = cell

    agents.clear()
    agents.update(founders)
    if not weighted:
        return document
    lineage[LINEAGE_FOUNDER_ID_LENGTH_KEY] = float(len(ids[0]))
    if founder_cycle_s > 0:
        for fid, weight in zip(ids, phase_weights(n_founders)):
            lineage[founder_weight_key(fid)] = weight
    return document
