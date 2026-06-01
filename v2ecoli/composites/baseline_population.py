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

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.composites.baseline import baseline as _baseline_builder
from v2ecoli.steps.population_aggregator import (
    DEFAULT_CELLS_PER_AGENT,
    DEFAULT_OD_TO_GDW,
    DEFAULT_REACTOR_VOLUME_L,
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
) -> dict:
    """Build the baseline_population document.

    Returns a process-bigraph document dict with the same shape as
    ``v2ecoli.composites.baseline.baseline`` plus an added top-level
    ``population`` store and ``population_aggregator`` Step.
    """
    document = _baseline_builder(core, seed=seed, cache_dir=cache_dir)

    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    state = document["state"]

    # Top-level data store. The aggregator owns the write path; pre-seed
    # with zeros so the document is structurally complete before the first
    # tick.
    state.setdefault("population", _empty_population_store())

    # Instantiate + wire the Step at the TOP LEVEL of state (alongside
    # `agents` / `global_time`). The Step's topology declares
    # `agents: ("agents",)` + `population: ("population",)` — wires resolve
    # against the top-level state dict because the edge sits at that level.
    aggregator_config = {
        "cells_per_agent":  cells_per_agent,
        "od_to_gdw":        od_to_gdw,
        "reactor_volume_L": reactor_volume_L,
    }
    aggregator = _make_instance(PopulationAggregator, aggregator_config, core)
    state[POPULATION_AGGREGATOR_STEP_NAME] = make_edge(
        aggregator, PopulationAggregator.topology, edge_type="step",
        config=aggregator_config,
    )

    # Register in flow_order so the executor knows about the new step.
    # Appended at the end so it runs after every per-cell step has emitted
    # (the aggregator reads the post-step agent state, not pre-step).
    document.setdefault("flow_order", []).append(POPULATION_AGGREGATOR_STEP_NAME)

    return document
