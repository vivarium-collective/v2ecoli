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
from v2ecoli.composites.baseline import baseline as _baseline_builder
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


# Step name used in the top-level state document and in flow_order.
POPULATION_AGGREGATOR_STEP_NAME = "population_aggregator"


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
) -> dict:
    """Add the top-level ``population`` store + ``PopulationAggregator`` Step.

    Shared helper so any cell-base document (the WCM ``baseline`` or the Millard
    ``baseline_millard``) gets the SAME cell-engine-agnostic aggregator. Mutates
    and returns ``document`` in place. The aggregator reads
    ``agents.*.listeners.mass.cell_mass`` and writes the reactor-scale
    observables; it is appended to ``flow_order`` so it runs after every per-cell
    step has emitted.
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

    # Register in flow_order. Appended at the end so it runs after every
    # per-cell step has emitted (the aggregator reads the post-step agent
    # state, not pre-step).
    document.setdefault("flow_order", []).append(POPULATION_AGGREGATOR_STEP_NAME)

    return document


@composite_generator(
    name="baseline_population",
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
) -> dict:
    """Build the baseline_population document.

    Returns a process-bigraph document dict with the same shape as
    ``v2ecoli.composites.baseline.baseline`` plus an added top-level
    ``population`` store and ``population_aggregator`` Step.
    """
    document = _baseline_builder(core, seed=seed, cache_dir=cache_dir)

    # Add the top-level population store + aggregator Step (shared helper, also
    # used by the Millard cell base in reactor_bird_coupled_millard).
    return add_population_aggregator(
        document, core,
        cells_per_agent=cells_per_agent,
        od_to_gdw=od_to_gdw,
        reactor_volume_L=reactor_volume_L,
        population_growth_mode=population_growth_mode,
    )
