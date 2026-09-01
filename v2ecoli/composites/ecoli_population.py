"""baseline_population — baseline composite with PopulationAggregator.

Build-phase wire-up for mbp-02-population-aggregation (see
``studies/mbp-02-population-aggregation/study.yaml`` req-2-population-composite
+ chris_feedback_2026_05_26 §4).

Adds a top-level ``population`` data store and a ``PopulationAggregator``
Step alongside the existing ``agents`` / ``global_time`` top-level keys.
The aggregator reads ``agents.*.listeners.mass.cell_mass`` and writes the
reactor-scale observables. Per-cell state is NEVER touched (regression-
guarded by per-cell-observables-unchanged-vs-baseline +
per-cell-observables-invariant-under-scaling).

Default ``cells_per_agent = 1.0`` preserves literal-sum so regression tests
against the unaggregated baseline remain byte-identical. Default
``reactor_volume_L = 1.0`` is overridden by mbp-03's
``v2ecoli.composites.reactor_bird_coupled`` once that composite reads
``reactor.volume_L`` from the BiRD store.
"""

from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.composites.ecoli_baseline import baseline as _baseline_builder
from v2ecoli.steps.lineage_bookkeeper import LineageBookkeeper
from v2ecoli.steps.population_aggregator import (
    DEFAULT_CELLS_PER_AGENT,
    DEFAULT_OD_TO_GDW,
    DEFAULT_POPULATION_GROWTH_MODE,
    DEFAULT_REACTOR_VOLUME_L,
    LINEAGE_DOUBLINGS_KEY,
    LINEAGE_GENERATION_KEY,
    LINEAGE_STORE_NAME,
    PopulationAggregator,
)


# Step names used in the top-level state document and in flow_order.
POPULATION_AGGREGATOR_STEP_NAME = "population_aggregator"
LINEAGE_BOOKKEEPER_STEP_NAME = "lineage_bookkeeper"

# Metabolism process name in the cache configs (config_overrides target).
_METABOLISM_PROC = "ecoli-metabolism"


def _carbon_arrest_overrides(
    enabled: bool, carbon_source_ids: list | None
) -> dict | None:
    """Metabolism ``config_overrides`` that enable the #572 substrate-exhaustion
    arrest, or ``None`` when disabled. Glucose is the default carbon source (M9);
    pass ``carbon_source_ids`` for other media. See
    ``v2ecoli.processes.metabolism`` (``carbon_exhaustion_arrest``)."""
    if not enabled:
        return None
    return {
        f"{_METABOLISM_PROC}.carbon_exhaustion_arrest": True,
        f"{_METABOLISM_PROC}.carbon_source_ids": list(carbon_source_ids or ["GLC[p]"]),
    }


def _empty_population_store() -> dict[str, float]:
    """Zero-initialized population store (populated by the aggregator at run)."""
    return {
        "total_biomass_gDW":          0.0,
        "cell_count":                 0.0,
        "biomass_concentration_gL":   0.0,
        "OD600":                      0.0,
    }


def _initial_lineage_store() -> dict[str, float]:
    """Top-level lineage store: generation count + represented doublings.

    Seeded at generation 1 / 0 doublings (factor 2^0 = 1, so the first
    generation never scales). The multigen runner advances ``doublings`` as it
    follows the lineage past divisions (#225 item #1); the aggregator reads it
    in ``representative_doubling`` mode. In the default ``fixed`` mode the store
    is harmless (the aggregator ignores it).
    """
    return {
        LINEAGE_GENERATION_KEY: 1.0,
        LINEAGE_DOUBLINGS_KEY:  0.0,
    }


def add_population_aggregator(
    document: dict,
    core: Any = None,
    *,
    cells_per_agent: float = DEFAULT_CELLS_PER_AGENT,
    od_to_gdw: float = DEFAULT_OD_TO_GDW,
    reactor_volume_L: float = DEFAULT_REACTOR_VOLUME_L,
    population_growth_mode: str = DEFAULT_POPULATION_GROWTH_MODE,
    single_daughters: bool = False,
) -> dict:
    """Add the top-level ``population`` store + ``PopulationAggregator`` Step.

    Shared helper so any cell-base document (the WCM ``baseline`` or the Millard
    ``baseline_millard``) gets the SAME cell-engine-agnostic aggregator. Mutates
    and returns ``document`` in place. The aggregator reads
    ``agents.*.listeners.mass.cell_mass`` and writes the reactor-scale
    observables; it is appended to ``flow_order`` so it runs after every per-cell
    step has emitted.

    ``single_daughters`` (default False): when True, also add a
    :class:`~v2ecoli.steps.lineage_bookkeeper.LineageBookkeeper` Step *before*
    the aggregator in ``flow_order``. It prunes the un-followed sibling and
    advances ``lineage.doublings`` ON the division tick (not the chunk boundary),
    so the aggregator/coupler see the correct represented population every tick
    regardless of the runner's ``chunk`` (fixes #588). When False the Step is
    absent and behavior is byte-identical to before. Must match the multigen
    runner's ``single_daughters`` so the in-composite pruned lineage and the
    runner's emitted lineage agree.
    """
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    state = document["state"]

    # Top-level data store. The aggregator owns the write path; pre-seed
    # with zeros so the document is structurally complete before the first tick.
    state.setdefault("population", _empty_population_store())
    # Top-level lineage store (doubling count for representative-growing mode,
    # #225 item #1). Always present so the aggregator's lineage input wire
    # resolves; harmless in the default fixed mode.
    state.setdefault(LINEAGE_STORE_NAME, _initial_lineage_store())

    flow_order = document.setdefault("flow_order", [])

    # In-composite division bookkeeping (#588). Added BEFORE the aggregator so
    # that on the division tick the sibling is pruned and lineage.doublings is
    # advanced before the aggregator (and the downstream coupler) read the
    # agents/lineage state. Only active when single_daughters=True; otherwise a
    # pure no-op, so non-single-lineage composites are byte-identical.
    if single_daughters:
        bookkeeper_config = {"single_daughters": True}
        bookkeeper = _make_instance(LineageBookkeeper, bookkeeper_config, core)
        bookkeeper_edge = make_edge(
            bookkeeper, LineageBookkeeper.topology, edge_type="step",
            config=bookkeeper_config,
        )
        # The lineage store's float leaves otherwise infer ACCUMULATE apply
        # semantics (bigraph-schema default for a bare float), so a Step writing
        # doublings=2 each tick would sum to 2,4,6,... The runner's
        # set_lineage_doublings sidesteps this by mutating the dict directly; a
        # Step must go through apply, so pin the leaves to overwrite (same device
        # the ReactorCellCoupler uses for its reactor leaves) — the bookkeeper
        # writes ABSOLUTE doublings/generation and they must replace, not add.
        bookkeeper_edge["_outputs"]["lineage"] = {
            LINEAGE_DOUBLINGS_KEY:  "overwrite[float]",
            LINEAGE_GENERATION_KEY: "overwrite[float]",
        }
        state[LINEAGE_BOOKKEEPER_STEP_NAME] = bookkeeper_edge
        flow_order.append(LINEAGE_BOOKKEEPER_STEP_NAME)

    aggregator_config = {
        "cells_per_agent":         cells_per_agent,
        "od_to_gdw":               od_to_gdw,
        "reactor_volume_L":        reactor_volume_L,
        "population_growth_mode":  population_growth_mode,
    }
    aggregator = _make_instance(PopulationAggregator, aggregator_config, core)
    state[POPULATION_AGGREGATOR_STEP_NAME] = make_edge(
        aggregator, PopulationAggregator.topology, edge_type="step",
        config=aggregator_config,
    )

    # Register in flow_order. Appended after the bookkeeper (if any) so it runs
    # after every per-cell step has emitted (the aggregator reads the post-step
    # agent state, not pre-step).
    flow_order.append(POPULATION_AGGREGATOR_STEP_NAME)

    return document


@composite_generator(
    name="ecoli_population",
    description=(
        "v2ecoli baseline + PopulationAggregator Step. Adds top-level "
        "population.* store with total_biomass_gDW, cell_count, "
        "biomass_concentration_gL, and OD600. Default cells_per_agent=1.0 "
        "preserves literal-sum aggregation so regression tests against the "
        "unaggregated baseline remain byte-identical."
    ),
    parameters={
        "seed":              {"type": "int",    "default": 0},
        "cache_dir":         {"type": "string", "default": "out/cache"},
        # Load-bearing architectural knob (chris_feedback_2026_05_26 §4):
        # representative-sampling scaling factor. Default 1.0 = literal-sum.
        "cells_per_agent":   {"type": "number", "default": DEFAULT_CELLS_PER_AGENT},
        "od_to_gdw":         {"type": "number", "default": DEFAULT_OD_TO_GDW},
        "reactor_volume_L":  {"type": "number", "default": DEFAULT_REACTOR_VOLUME_L},
        # "fixed" (default) | "representative_doubling" (#225 item #1). In
        # doubling mode the multigen runner advances lineage.doublings so the
        # represented population grows 2x per generation (breaks the plateau).
        "population_growth_mode": {"type": "string", "default": DEFAULT_POPULATION_GROWTH_MODE},
        # Opt into the #572 substrate-exhaustion growth arrest (default off ->
        # metabolism unchanged). Needed for batch-to-exhaustion runs so the cell
        # arrests instead of growing on phantom internal carbon once glucose is
        # gone. carbon_source_ids defaults to ["GLC[p]"] (M9 glucose).
        "carbon_exhaustion_arrest": {"type": "boolean", "default": False},
        # Follow a single lineage past divisions (#588). Must match the multigen
        # runner's single_daughters. Adds the in-composite LineageBookkeeper so
        # sibling-prune + doublings advance ON the division tick, making the
        # aggregator/coupler chunk-independent. Default False = no-op.
        "single_daughters": {"type": "boolean", "default": False},
        # Per-cell biological build kwarg, forwarded verbatim to baseline().
        # Without it a population run silently builds whatever metabolism the
        # cache defaults to, so a pathway cache produces no product and the run
        # looks healthy -- the same shape #640 fixed one call site over, on the
        # batch path. Empty = none, so the default is byte-identical to before.
        "injected_processes": {
            "type": "map",
            "default": {},
            "description": "Process-injection spec {fork_repo, add_processes, "
                           "swap_processes, process_configs, topology, "
                           "time_step}; empty = none. Passed through to "
                           "baseline(), which enforces the native-vs-fork "
                           "sourcing policy via assert_injection_sourcing.",
        },
    },
)
def baseline_population(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    cells_per_agent: float = DEFAULT_CELLS_PER_AGENT,
    od_to_gdw: float = DEFAULT_OD_TO_GDW,
    reactor_volume_L: float = DEFAULT_REACTOR_VOLUME_L,
    population_growth_mode: str = DEFAULT_POPULATION_GROWTH_MODE,
    carbon_exhaustion_arrest: bool = False,
    carbon_source_ids: list | None = None,
    single_daughters: bool = False,
    injected_processes: dict | None = None,
) -> dict:
    """Build the baseline_population document.

    Returns a process-bigraph document dict with the same shape as
    ``v2ecoli.composites.ecoli_baseline.ecoli_baseline`` plus an added top-level
    ``population`` store and ``population_aggregator`` Step.

    ``carbon_exhaustion_arrest`` (default False): opt into the substrate-exhaustion
    growth arrest (#572). When True the cell stops building biomass once none of
    ``carbon_source_ids`` (default ``["GLC[p]"]``) is importable — needed for
    batch-to-exhaustion runs so the cell arrests instead of growing on phantom
    internal carbon. Threaded onto the metabolism process via ``config_overrides``.
    """
    config_overrides = _carbon_arrest_overrides(
        carbon_exhaustion_arrest, carbon_source_ids)
    document = _baseline_builder(
        core, seed=seed, cache_dir=cache_dir, config_overrides=config_overrides,
        injected_processes=injected_processes)

    # Add the top-level population store + aggregator Step (shared helper, also
    # used by the Millard cell base in reactor_bird_coupled_millard).
    return add_population_aggregator(
        document, core,
        cells_per_agent=cells_per_agent,
        od_to_gdw=od_to_gdw,
        reactor_volume_L=reactor_volume_L,
        population_growth_mode=population_growth_mode,
        single_daughters=single_daughters,
    )
