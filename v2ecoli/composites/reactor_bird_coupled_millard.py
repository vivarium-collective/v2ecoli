"""reactor_bird_coupled_millard — Millard cell + BiRD reactor + cell<->reactor coupler.

Build-phase wire-up for mbp-09 (req-2). The Millard analogue of
``reactor_bird_coupled``: it swaps the WCM FBA-metabolism cell for the Millard
2017 kinetic-ODE cell (``baseline_millard``) but layers on the *identical*
population + reactor coupling.

Structure (sibling composite, chosen over a ``cell_engine`` param on
``reactor_bird_coupled`` because the two cell-base builders have different
signatures — ``baseline_population`` bakes in the aggregator and takes
od_to_gdw/reactor_volume_L, while ``baseline_millard`` takes tick_s/features and
has no aggregator — and a distinct registered spec id is cleaner for the
dashboard/studies)::

    baseline_millard (cell side)
      agents.0.*            single-cell Millard kinetic-metabolism WCM
    + add_population_aggregator   -> population.* (cell-engine-agnostic)
    + add_reactor_coupling        -> environment.* + reactor.* + transport + coupler

Both ``add_population_aggregator`` (baseline_population) and
``add_reactor_coupling`` (reactor_bird_coupled) are the SAME shared helpers the
WCM arm uses — no reactor/population/coupler code is duplicated here. The coupler
is cell-engine-agnostic: it reads ``population.biomass_concentration_gL`` +
``agents.*.environment.exchange`` (O2/CO2 counts). The Millard cell now emits
those medium-exchange counts (O2 via CYTBO, glucose via XCH_GLC; mbp-09 req-1),
so the Millard cell's respiration drives reactor O2 depletion end-to-end exactly
as the WCM cell does.
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites.baseline_millard import baseline_millard
from v2ecoli.composites.baseline_population import add_population_aggregator
from v2ecoli.composites._millard_helpers import _register_millard_pdmp_links
from v2ecoli.composites.reactor_bird_coupled import (
    DEFAULT_BIRD_REACTOR_CONFIG,
    add_reactor_coupling,
)
from v2ecoli.steps.population_aggregator import (
    DEFAULT_CELLS_PER_AGENT,
    DEFAULT_OD_TO_GDW,
)


@composite_generator(
    name="reactor_bird_coupled_millard",
    description=(
        "v2ecoli Millard kinetic-metabolism cell (baseline_millard) + "
        "PopulationAggregator coupled to the BiRD bioreactor "
        "(BiRDTransportHours) via ReactorCellCoupler. The Millard arm of "
        "reactor_bird_coupled: identical population + reactor coupling layered "
        "on the Millard 2017 ODE cell instead of the WCM FBA-metabolism cell. "
        "The Millard cell's respiration (CYTBO O2 uptake, emitted as "
        "environment.exchange counts) nets additively against the reactor's "
        "gas-liquid transport at shared reactor.dissolved_o2/dissolved_co2 "
        "stores, so the kinetic cell consumes reactor O2 end-to-end."
    ),
    parameters={
        "seed":      {"type": "int",    "default": 0},
        "cache_dir": {"type": "string", "default": "out/cache"},
        "tick_s":    {"type": "float",  "default": 1.0},
        "features":  {"type": "list",   "default": []},
        # BiRDTransportProcess config (reactor_type / volume_L /
        # gas_flow_rate_Lpm / temperature_K). Time base hours.
        "bird_reactor_config": {
            "type": "object", "default": dict(DEFAULT_BIRD_REACTOR_CONFIG)},
        # Population-aggregator knobs.
        "cells_per_agent": {"type": "number", "default": DEFAULT_CELLS_PER_AGENT},
        "od_to_gdw":       {"type": "number", "default": DEFAULT_OD_TO_GDW},
        # "fixed" (default) | "representative_doubling" (#225 item #1).
        "population_growth_mode": {"type": "string", "default": "fixed"},
    },
    core_extensions=[_register_millard_pdmp_links],
)
def reactor_bird_coupled_millard(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    tick_s: float = 1.0,
    features: list | None = None,
    bird_reactor_config: dict | None = None,
    cells_per_agent: float = DEFAULT_CELLS_PER_AGENT,
    od_to_gdw: float = DEFAULT_OD_TO_GDW,
    population_growth_mode: str = "fixed",
) -> dict:
    """Build the reactor_bird_coupled_millard document.

    Takes ``baseline_millard`` (the Millard kinetic cell) as the cell base, adds
    the cell-engine-agnostic PopulationAggregator, then layers on the SAME BiRD
    reactor + cell<->reactor coupling helper that ``reactor_bird_coupled`` uses.
    """
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    bird_config = dict(DEFAULT_BIRD_REACTOR_CONFIG)
    if bird_reactor_config:
        bird_config.update(bird_reactor_config)

    # --- cell side: Millard kinetic-metabolism cell -----------------------
    document = baseline_millard(
        core, seed=seed, cache_dir=cache_dir, tick_s=tick_s, features=features,
    )

    # --- population: cell-engine-agnostic aggregator ----------------------
    add_population_aggregator(
        document, core,
        cells_per_agent=cells_per_agent,
        od_to_gdw=od_to_gdw,
        reactor_volume_L=float(bird_config.get("volume_L", 1.0)),
        population_growth_mode=population_growth_mode,
    )

    # --- env hook + reactor + coupler (shared with reactor_bird_coupled) ---
    return add_reactor_coupling(
        document, core,
        bird_config=bird_config, cells_per_agent=cells_per_agent,
    )
