"""PopulationAggregator — reactor-scale biomass / cell-count aggregator.

Build-phase scaffold for mbp-02-population-aggregation (see
``studies/mbp-02-population-aggregation/study.yaml`` req-1-population-aggregator
+ chris_feedback_2026_05_26 §4).

PBG Step that, each emit cycle, walks ``agents.*.listeners.mass.cell_mass``,
sums, applies the ``cells_per_agent`` scaling factor, and writes:

  population.total_biomass_gDW =
      sum(agents.*.cell_mass) × cells_per_agent × 1e-15 g/fg
  population.cell_count        = len(agents) × cells_per_agent
                                 (float — cells_per_agent may be non-integer)
  population.biomass_concentration_gL =
      total_biomass_gDW / reactor_volume_L
  population.OD600 =
      biomass_concentration_gL / od_to_gdw   # COSMETIC

NO mutation of agent state (regression-guarded by
``per-cell-observables-unchanged-vs-baseline`` +
``per-cell-observables-invariant-under-scaling`` tests). The
``cells_per_agent`` factor is applied ONLY to the population.* outputs —
NEVER to per-cell stores.

``cells_per_agent`` is the load-bearing architectural decision (Eran's
adoption 2026-05-26 of representative-sampling over literal-sum). Default
1.0 preserves literal-sum so existing single-cell sims and the
unaggregated baseline regression test remain byte-identical. Production
runs override to scale the simulated lineage to a target population size.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


# Default conversion / scaling constants (override in composite config).
DEFAULT_CELLS_PER_AGENT: float = 1.0           # literal-sum preserves baseline
DEFAULT_OD_TO_GDW: float = 0.34                # Beulig 2025; was textbook 0.33
DEFAULT_REACTOR_VOLUME_L: float = 1.0
FG_PER_GRAM: float = 1.0e-15                   # cell_mass listener is in fg


class PopulationAggregator(Step):
    """Aggregate per-cell mass + agent count into reactor-scale observables.

    See module docstring; spec in
    ``studies/mbp-02-population-aggregation/study.yaml`` req-1-population-aggregator.
    """

    name = "population_aggregator"
    # NOTE on schema syntax: bigraph-schema's `core.fill` resolves
    # `{"_default": <val>}` as an opaque scalar (no type → user overrides
    # are discarded, only the default flows through). Use bare type names
    # ("float", "integer") so user overrides actually take effect. See
    # `_make_instance` in composites/_helpers.py for the v2ecoli init path.
    config_schema = {
        "cells_per_agent":   "float",
        "od_to_gdw":         "float",
        "reactor_volume_L":  "float",
        "time_step":         "float",
    }
    topology = {
        "agents":     ("agents",),
        "population": ("population",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.cells_per_agent = float(cfg.get("cells_per_agent") or DEFAULT_CELLS_PER_AGENT)
        self.od_to_gdw = float(cfg.get("od_to_gdw") or DEFAULT_OD_TO_GDW)
        self.reactor_volume_L = float(cfg.get("reactor_volume_L") or DEFAULT_REACTOR_VOLUME_L)

    def inputs(self) -> dict[str, Any]:
        return {"agents": InPlaceDict()}

    def outputs(self) -> dict[str, Any]:
        return {"population": InPlaceDict()}

    # --- main update -------------------------------------------------------

    def next_update(self, timestep, states):
        agents = states.get("agents", {}) or {}
        n_simulated = len(agents)
        if n_simulated == 0:
            # Empty population — emit zeros (used by mbp-03's
            # no-cells-henry-equilibrium sim where force_zero_population=true).
            return {"population": _build_population_dict(
                total_biomass_gDW=0.0,
                cell_count=0.0,
                biomass_concentration_gL=0.0,
                od600=0.0,
            )}

        sum_cell_mass_fg = 0.0
        for _agent_id, agent_state in agents.items():
            cell_mass = _extract_cell_mass_fg(agent_state)
            if cell_mass is not None:
                sum_cell_mass_fg += cell_mass

        total_biomass_gDW = sum_cell_mass_fg * self.cells_per_agent * FG_PER_GRAM
        cell_count = float(n_simulated) * self.cells_per_agent
        biomass_concentration_gL = total_biomass_gDW / self.reactor_volume_L
        # OD600 is COSMETIC (per chris_feedback_2026_05_26 §4) — derive from
        # biomass_concentration_gL strictly for plotting / OD-only comparisons.
        od600 = biomass_concentration_gL / self.od_to_gdw if self.od_to_gdw else 0.0

        return {"population": _build_population_dict(
            total_biomass_gDW=total_biomass_gDW,
            cell_count=cell_count,
            biomass_concentration_gL=biomass_concentration_gL,
            od600=od600,
        )}

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)


# --- helpers ----------------------------------------------------------------

def _extract_cell_mass_fg(agent_state: dict | Any) -> float | None:
    """Walk agent state to ``listeners.mass.cell_mass``; return as float in fg.

    Defensive against missing intermediate dict keys — emit cadence may
    snapshot agents mid-init. Returns None when the cell_mass key is missing
    so the aggregator skips that agent rather than crashing.
    """
    try:
        listeners = agent_state.get("listeners", {}) if hasattr(agent_state, "get") else {}
        mass = listeners.get("mass", {})
        cell_mass = mass.get("cell_mass")
        if cell_mass is None:
            return None
        # pint Quantity? strip units to fg.
        if hasattr(cell_mass, "to") and hasattr(cell_mass, "magnitude"):
            return float(cell_mass.to("femtogram").magnitude)
        return float(cell_mass)
    except (AttributeError, KeyError, TypeError):
        return None


def _build_population_dict(
    *,
    total_biomass_gDW: float,
    cell_count: float,
    biomass_concentration_gL: float,
    od600: float,
) -> dict[str, float]:
    return {
        "total_biomass_gDW":          total_biomass_gDW,
        "cell_count":                 cell_count,
        "biomass_concentration_gL":   biomass_concentration_gL,
        "OD600":                      od600,
    }
