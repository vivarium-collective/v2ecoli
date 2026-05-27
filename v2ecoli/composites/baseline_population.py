"""baseline_population — baseline composite with PopulationAggregator.

Build-phase scaffold for mbp-02-population-aggregation
(req-2-population-composite). Thin wrapper around ``v2ecoli.composites.baseline``
that adds the ``population`` store + ``PopulationAggregator`` Step so
reactor-scale biomass / OD / cell-count observables exist.

Default ``reactor_volume_L = 1.0``; overridden by mbp-03's
``v2ecoli.composites.reactor_bird_coupled`` once that composite reads
``reactor.volume_L`` from the BiRD store. Default ``cells_per_agent = 1.0``
preserves literal-sum aggregation so regression tests against the
unaggregated baseline remain byte-identical.

See ``studies/mbp-02-population-aggregation/study.yaml`` for the full spec
and ``references/expert/chris_feedback_2026_05_26.md`` §4 for the
cells_per_agent rationale.

TODO (Build phase):
  - Inject PopulationAggregator into document["composition"][...] at the
    emit-cadence Step layer. The aggregator reads agents/ and writes
    population/; both stores need to be declared in the document state
    schema (population.* doesn't exist in the unmodified baseline document).
  - Verify per-cell observables are byte-identical to the unaggregated
    baseline (regression guard:
    per-cell-observables-unchanged-vs-baseline).
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites.baseline import baseline as _baseline_builder
from v2ecoli.steps.population_aggregator import (
    DEFAULT_CELLS_PER_AGENT,
    DEFAULT_OD_TO_GDW,
    DEFAULT_REACTOR_VOLUME_L,
)


@composite_generator(
    name="baseline_population",
    description=(
        "v2ecoli baseline + PopulationAggregator Step. Adds population.* "
        "store with total_biomass_gDW, cell_count, biomass_concentration_gL, "
        "and OD600. Default cells_per_agent=1.0 preserves literal-sum "
        "aggregation so regression tests against the unaggregated baseline "
        "remain byte-identical."
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
    core: Any,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    cells_per_agent: float = DEFAULT_CELLS_PER_AGENT,
    od_to_gdw: float = DEFAULT_OD_TO_GDW,
    reactor_volume_L: float = DEFAULT_REACTOR_VOLUME_L,
) -> dict:
    """Build the baseline_population document.

    Build-phase scaffold: today this just delegates to the unmodified
    baseline. The TODO above tracks the actual wire-up work.
    """
    document = _baseline_builder(core, seed=seed, cache_dir=cache_dir)

    # TODO: inject PopulationAggregator Step into document["composition"][...]
    # and declare the population.* store in the document schema.

    # Stash config on the document for downstream wiring once the hook lands.
    meta = document.setdefault("_v2ecoli_meta", {})
    meta["population_aggregator"] = {
        "cells_per_agent":   cells_per_agent,
        "od_to_gdw":         od_to_gdw,
        "reactor_volume_L":  reactor_volume_L,
    }

    return document
