"""LineageBookkeeper — in-composite division bookkeeping (fixes #588).

Context (issue #588)
--------------------
The multigen runners (``run_multigen_{sqlite,parquet,xarray}``) step the
composite ``chunk`` ticks at a time and then, at the *chunk boundary*, do two
things externally:

  * ``prune_to_followed_lineage`` — drop the un-followed sibling daughter, and
  * ``set_lineage_doublings`` — advance ``lineage.doublings`` for the new
    generation.

But ``composite.run(chunk)`` does not stop at a division: it completes all
``chunk`` ticks with *both* daughters live and ``doublings`` still at the old
value. The ``PopulationAggregator`` and ``ReactorCellCoupler`` run every tick
*inside* that window, so for ``chunk_boundary - division_tick`` ticks they
integrate the wrong represented population into the reactor. The number of such
ticks depends on ``chunk`` — an emit-cadence knob — so no ``chunk`` value gives
a reproducible reactor trajectory.

The fix
-------
Move the bookkeeping *inside* the composite so it fires on the division tick,
not the chunk boundary. This Step runs every tick (placed just before the
``PopulationAggregator`` in ``flow_order``) and, when following a single
lineage (``single_daughters=True``):

  * prunes any agent that is not the followed lineage via a structural
    ``{'agents': {'_remove': [...]}}`` update (the framework rebuilds the
    Composite's instance-path caches for structural updates, so no manual
    ``find_instance_paths`` is needed — unlike the runner's raw ``del``), and
  * writes ``lineage.doublings`` / ``lineage.generation`` derived *statelessly*
    from the followed agent's phylogeny id.

Statelessness is the point: agent keys ARE phylogeny ids
(``"0"`` -> ``"00"``/``"01"`` -> ...; see
``v2ecoli.library.division.daughter_phylogeny_id``), so the followed lineage
after generation ``g`` is a string of ``g`` zeros and
``doublings = len(followed_id) - 1``. There is no division-event to detect and
no counter to get wrong at a chunk boundary — the represented population is a
pure function of the current state, identical at every ``chunk``.

Scope / safety
--------------
When ``single_daughters=False`` (the default) this Step is a pure no-op, so
every existing multi-agent / ``fixed``-mode / single-cell-baseline composite is
byte-identical. It only acts in the single-lineage-following mode the mbp
coupled studies use. The runner's own ``prune``/``set_lineage_doublings`` calls
become harmless idempotent reassertions of the same state.

The follow-policy — keep the daughter whose id ends in ``"0"`` — is the SAME
rule the runners use to choose which lineage to follow and rotate the emitter
onto (``run_multigen_sqlite``: ``next(i for i in sorted(ids) if i.endswith("0"))``),
so the in-composite pruned lineage and the runner's emitted lineage always
agree without a shared signal.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict

from v2ecoli.steps.population_aggregator import (
    LINEAGE_DOUBLINGS_KEY,
    LINEAGE_GENERATION_KEY,
)


def followed_lineage_id(agent_ids: list[str]) -> str | None:
    """Pick the single followed lineage from the current agent ids.

    Mirrors the runners' rule exactly: prefer the (sorted) first id ending in
    ``"0"`` — the all-zeros lineage under
    ``daughter_phylogeny_id`` (``mother -> mother+'0'`` / ``mother+'1'``) — else
    fall back to the first sorted id. Returns ``None`` for an empty population.
    """
    ids = sorted(str(a) for a in agent_ids)
    if not ids:
        return None
    return next((i for i in ids if i.endswith("0")), ids[0])


def doublings_for(followed_id: str) -> float:
    """Represented-population doublings for a followed phylogeny id.

    Agent keys are phylogeny ids, so the followed lineage at generation ``g`` is
    ``g`` characters long and ``doublings = generation - 1 = len(id) - 1``.
    Floored at 0 so the founder ``"0"`` (generation 1) is 0 doublings -> factor
    ``2**0 = 1``.
    """
    return float(max(0, len(str(followed_id)) - 1))


class LineageBookkeeper(Step):
    """Maintain single-lineage population bookkeeping inside the composite.

    See the module docstring and issue #588. Active only when
    ``single_daughters=True``; a no-op otherwise.
    """

    name = "lineage_bookkeeper"
    # Bare type name so a user/composite override actually takes effect (see the
    # PopulationAggregator note on bigraph-schema's `core.fill` of `_default`).
    config_schema = {
        "single_daughters": "boolean",
    }
    topology = {
        "agents":  ("agents",),
        "lineage": ("lineage",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.single_daughters = bool(cfg.get("single_daughters") or False)

    def inputs(self) -> dict[str, Any]:
        return {"agents": InPlaceDict(), "lineage": InPlaceDict()}

    def outputs(self) -> dict[str, Any]:
        # ``agents`` is a structural output (``_remove``), same schema the
        # Division step uses for its ``_add``/``_remove`` update.
        return {"agents": {"_type": "map", "_value": "node"}, "lineage": InPlaceDict()}

    def next_update(self, timestep, states):
        # Pure no-op unless we are following a single lineage. Guarantees every
        # existing multi-agent / fixed-mode / baseline composite is unaffected.
        if not self.single_daughters:
            return {}

        agents = states.get("agents", {}) or {}
        agent_ids = list(agents.keys())
        followed = followed_lineage_id(agent_ids)
        if followed is None:
            return {}

        update: dict[str, Any] = {
            "lineage": {
                LINEAGE_DOUBLINGS_KEY:  doublings_for(followed),
                LINEAGE_GENERATION_KEY: doublings_for(followed) + 1.0,
            }
        }

        # Prune every non-followed lineage AT the division tick, so the
        # aggregator/coupler (which run after this Step in flow_order) never
        # integrate a sibling. Structural _remove; the framework rebuilds the
        # instance-path caches (Division relies on the same).
        to_remove = [aid for aid in agent_ids if aid != followed]
        if to_remove:
            update["agents"] = {"_remove": to_remove}

        return update

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)
