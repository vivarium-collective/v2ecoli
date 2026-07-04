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

Representative-growing-population mode (#225 item #1)
----------------------------------------------------
When ``population_growth_mode='representative_doubling'`` the represented
``cell_count`` / biomass are additionally multiplied by ``2**doublings`` where
``doublings`` is read from a top-level ``lineage`` store (``lineage.doublings``,
set by the multigen runner = ``generation - 1``). This breaks the single-lineage
biomass PLATEAU: the followed lineage stands in for a population that doubles
each generation, so population biomass ACCUMULATES exponentially across
generations instead of just oscillating with one cell's mass. Because a daughter
halves its mass at division while the represented count doubles, total biomass
is CONTINUOUS across the division (no artificial jump). The default mode
(``'fixed'``, factor 1.0) is byte-identical to mbp-02's cells_per_agent
aggregation, so existing studies/tests are unaffected.

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

# --- Representative-growing-population mode (#225 item #1) -------------------
# The single-lineage runner (single_daughters=True) follows ONE cell, so the
# represented population PLATEAUS — biomass just oscillates with one cell's mass
# (grows, halves at division) and never accumulates toward a batch density. The
# `representative_doubling` mode makes the REPRESENTED population grow 2x per
# generation: at generation g, the represented cell_count is
#   N0 x cells_per_agent x 2^(g-1)
# i.e. one followed lineage stands in for a population that DOUBLES each division
# (the sibling we drop is represented by the doubling factor instead). This is
# the same 0D well-mixed representative-sampling assumption that licenses
# cells_per_agent (mbp-02): one representative lineage stands in for N0 x 2^g
# cells. The runner injects the doubling count (`lineage.doublings` = g-1) into
# a top-level `lineage` store each generation; the aggregator reads it and
# applies the 2^doublings factor.
#
# Continuity across division (the load-bearing invariant): at a division a
# daughter HALVES its mass while the represented count DOUBLES, so total
# represented biomass is CONTINUOUS across the division (no artificial jump);
# the daughter then grows the next generation's mass back up, so the population
# biomass grows ~2x per generation instead of plateauing. See the unit test
# `test_representative_doubling_continuous_then_grows`.
GROWTH_MODE_FIXED: str = "fixed"
GROWTH_MODE_DOUBLING: str = "representative_doubling"
DEFAULT_POPULATION_GROWTH_MODE: str = GROWTH_MODE_FIXED
# Store + key the runner writes / the aggregator reads.
LINEAGE_STORE_NAME: str = "lineage"
LINEAGE_DOUBLINGS_KEY: str = "doublings"
LINEAGE_GENERATION_KEY: str = "generation"


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
        "cells_per_agent":         "float",
        "od_to_gdw":               "float",
        "reactor_volume_L":        "float",
        "time_step":               "float",
        # "fixed" (default) | "representative_doubling" (#225 item #1).
        "population_growth_mode":  "string",
    }
    topology = {
        "agents":     ("agents",),
        "population": ("population",),
        # Read-only doubling-count store written by the multigen runner; absent
        # for single-generation / non-runner builds (defensively defaulted to 0).
        "lineage":    ("lineage",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.cells_per_agent = float(cfg.get("cells_per_agent") or DEFAULT_CELLS_PER_AGENT)
        self.od_to_gdw = float(cfg.get("od_to_gdw") or DEFAULT_OD_TO_GDW)
        self.reactor_volume_L = float(cfg.get("reactor_volume_L") or DEFAULT_REACTOR_VOLUME_L)
        self.population_growth_mode = (
            cfg.get("population_growth_mode") or DEFAULT_POPULATION_GROWTH_MODE
        )

    def inputs(self) -> dict[str, Any]:
        return {"agents": InPlaceDict(), "lineage": InPlaceDict()}

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

        # Representative-growing-population factor (#225 item #1). In the default
        # "fixed" mode this is 1.0 and behavior is identical to mbp-02's
        # cells_per_agent aggregation. In "representative_doubling" mode the
        # followed lineage stands in for a population that doubles each
        # generation, so apply 2^doublings (doublings = generation - 1, injected
        # by the multigen runner into the `lineage` store). The daughter's mass
        # halving at division cancels this doubling, so total biomass is
        # continuous across the division and then grows with the next
        # generation — see module docstring + the unit test.
        growth_factor = self._growth_factor(states)

        total_biomass_gDW = (
            sum_cell_mass_fg * self.cells_per_agent * growth_factor * FG_PER_GRAM
        )
        cell_count = float(n_simulated) * self.cells_per_agent * growth_factor
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

    # --- representative-growing-population factor ---------------------------

    def _growth_factor(self, states) -> float:
        """Population-growth multiplier from the current doubling count.

        Returns 1.0 in the default "fixed" mode (mbp-02 behavior). In
        "representative_doubling" mode, reads ``lineage.doublings`` (set by the
        multigen runner = generation - 1) and returns ``2 ** doublings``.
        Defensive: a missing/empty ``lineage`` store -> 0 doublings -> 1.0, so
        the first generation never scales and non-runner builds are unaffected.
        """
        if self.population_growth_mode != GROWTH_MODE_DOUBLING:
            return 1.0
        lineage = (states or {}).get(LINEAGE_STORE_NAME) or {}
        try:
            doublings = float(lineage.get(LINEAGE_DOUBLINGS_KEY, 0) or 0)
        except (AttributeError, TypeError):
            doublings = 0.0
        if doublings < 0:
            doublings = 0.0
        return 2.0 ** doublings


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
