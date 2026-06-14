"""ReactorCellCoupler — bridge v2ecoli's cell population to the BiRD reactor.

Build-phase scaffold for mbp-03 (req-2). PBG Step that runs each emit cycle,
implementing **Option B** of the v2ecoli <-> pbg-bioreactordesign coupling:

  * BiRD owns transport physics + reads biomass as an input.
  * v2ecoli owns biomass + the metabolic exchange demand.

This Step is the thin translator between the two halves. Each cycle it:

1. **Biomass passthrough.** Read ``population.biomass_concentration_gL`` (g/L,
   produced by :class:`PopulationAggregator`) and write it to
   ``reactor.biomass`` (g/L) for ``BiRDTransportProcess`` to consume. This is a
   passthrough/overwrite of the absolute value (the ``reactor.biomass`` port
   must be wired as an overwrite type in the composite).

2. **Dissolved-gas -> environment concentration.** Read ``reactor.dissolved_o2``
   and ``reactor.dissolved_co2`` (mg/L) and write the cell environment's
   ``external_concentrations`` (mM) for the two gases.

   Conversion (the load-bearing one):

       mM = (mg/L) / MW[g/mol]

   because  (mg/L) / (g/mol) = (1e-3 g/L) / (g/mol) = 1e-3 mol/L = 1 mM.
   e.g. 8 mg/L O2 / 31.999 ~= 0.25 mM. These are absolute concentrations
   (overwrite the environment port, not additive).

3. **Metabolic demand -> dissolved-gas delta.** Aggregate each agent's
   metabolic exchange flux into an *additive* dissolved-gas delta (mg/L) the
   reactor accumulates this interval:

       biomass_gDW_agent = cell_mass_fg * cells_per_agent * 1e-15
       rate[mmol/h]      = sum_agents( flux[mmol/(gDW*h)] * biomass_gDW_agent )
       delta[mg/L]       = rate[mmol/h] * interval_h * MW[mg/mmol] / volume_L

   MW in g/mol numerically equals mg/mmol, so the same constant serves both
   the mM conversion and this mg/L delta.

Sign convention (verified against ``v2ecoli/processes/metabolism.py``
:func:`_fba_output_to_deltas`, lines 469-492): ``external_exchange_fluxes`` is
reported on a gDCW basis where the FBA exchange flux feeds
``delta_nutrients = (1/counts_to_molar) * exchange_fluxes`` — described there as
"the cumulative count added to the environment store". So a POSITIVE flux means
the molecule is *secreted* (added to the environment) and a NEGATIVE flux means
*uptake* (removed). Therefore O2 (consumed) carries a negative flux and yields a
NEGATIVE ``reactor.dissolved_o2`` delta (uptake depletes dissolved O2), while
CO2 (produced) carries a positive flux and yields a POSITIVE
``reactor.dissolved_co2`` delta. The Step passes the signed flux straight
through the conversion chain, so the sign is preserved without special-casing.

Time base: ``BiRDTransportProcess`` works in HOURS and the exchange fluxes are
per hour (mmol/(gDW*h)), while v2ecoli's cell sim steps in SECONDS. The Step
therefore converts its timestep to hours (``/ SECONDS_PER_HOUR``) before scaling
the per-hour flux into a per-interval delta.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


# --- Constants --------------------------------------------------------------

# Molecular weights (g/mol == mg/mmol).
MW_O2: float = 31.999
MW_CO2: float = 44.010

# Metabolism exchange-flux map keys (periplasmic compartment).
O2_ID: str = "OXYGEN-MOLECULE[p]"
CO2_ID: str = "CARBON-DIOXIDE[p]"

SECONDS_PER_HOUR: float = 3600.0
FG_PER_GRAM: float = 1.0e-15

DEFAULT_CELLS_PER_AGENT: float = 1.0
DEFAULT_REACTOR_VOLUME_L: float = 1.0


class ReactorCellCoupler(Step):
    """Translate between v2ecoli's cell population and the BiRD reactor stores.

    See module docstring; spec in
    ``studies/mbp-03-*/study.yaml`` req-2 (reactor<->cell coupler).
    """

    name = "reactor_cell_coupler"
    # Bare type names (not {"_default": ...}) so config overrides propagate —
    # see PopulationAggregator's schema comment for the bigraph-schema rationale.
    config_schema = {
        "cells_per_agent":  "float",
        "reactor_volume_L": "float",
        "time_step":        "float",
    }
    topology = {
        "population":  ("population",),
        "reactor":     ("reactor",),
        "environment": ("environment",),
        "agents":      ("agents",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.cells_per_agent = float(
            cfg.get("cells_per_agent") or DEFAULT_CELLS_PER_AGENT)
        self.reactor_volume_L = float(
            cfg.get("reactor_volume_L") or DEFAULT_REACTOR_VOLUME_L)

    def inputs(self) -> dict[str, Any]:
        return {
            "population": InPlaceDict(),
            "reactor":    InPlaceDict(),
            "agents":     InPlaceDict(),
        }

    def outputs(self) -> dict[str, Any]:
        return {
            "reactor":     InPlaceDict(),
            "environment": InPlaceDict(),
        }

    # --- main update -------------------------------------------------------

    def next_update(self, timestep, states):
        update: dict[str, Any] = {}

        reactor_out: dict[str, float] = {}
        env_concs: dict[str, float] = {}

        reactor = states.get("reactor") or {}
        # Reactor volume: prefer the live store, fall back to config default.
        volume_L = _as_float(reactor.get("volume_L"))
        if not volume_L or volume_L <= 0.0:
            volume_L = self.reactor_volume_L

        # 1. Biomass passthrough (g/L) — overwrite the reactor biomass input.
        population = states.get("population") or {}
        biomass_gL = population.get("biomass_concentration_gL")
        if biomass_gL is not None:
            reactor_out["biomass"] = _as_float(biomass_gL)

        # 2. Dissolved gas (mg/L) -> environment concentration (mM).
        do2 = reactor.get("dissolved_o2")
        dco2 = reactor.get("dissolved_co2")
        if do2 is not None:
            env_concs[O2_ID] = _as_float(do2) / MW_O2
        if dco2 is not None:
            env_concs[CO2_ID] = _as_float(dco2) / MW_CO2

        # 3. Metabolic exchange demand -> additive dissolved-gas delta (mg/L).
        agents = states.get("agents") or {}
        interval_h = float(timestep) / SECONDS_PER_HOUR
        o2_rate = 0.0   # mmol/h, signed (negative == uptake)
        co2_rate = 0.0  # mmol/h, signed (positive == evolution)
        for _agent_id, agent_state in agents.items():
            cell_mass_fg = _extract_cell_mass_fg(agent_state)
            if cell_mass_fg is None:
                continue
            biomass_gDW = cell_mass_fg * self.cells_per_agent * FG_PER_GRAM
            fluxes = _extract_exchange_fluxes(agent_state)
            o2_rate += _as_float(fluxes.get(O2_ID, 0.0)) * biomass_gDW
            co2_rate += _as_float(fluxes.get(CO2_ID, 0.0)) * biomass_gDW

        if agents:
            # Signs flow straight through: negative O2 flux -> negative delta.
            reactor_out["dissolved_o2"] = o2_rate * interval_h * MW_O2 / volume_L
            reactor_out["dissolved_co2"] = co2_rate * interval_h * MW_CO2 / volume_L

        if reactor_out:
            update["reactor"] = reactor_out
        if env_concs:
            update["environment"] = {"external_concentrations": env_concs}
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)


# --- helpers ----------------------------------------------------------------

def _as_float(value: Any) -> float:
    """Coerce a possibly pint-Quantity scalar to a bare float."""
    if value is None:
        return 0.0
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    return float(value)


def _extract_cell_mass_fg(agent_state: dict | Any) -> float | None:
    """Walk agent state to ``listeners.mass.cell_mass``; return float in fg.

    Defensive against missing intermediate keys (emit cadence may snapshot an
    agent mid-init). Returns None when cell_mass is absent so the coupler skips
    that agent rather than crashing.
    """
    try:
        listeners = agent_state.get("listeners", {}) if hasattr(agent_state, "get") else {}
        mass = listeners.get("mass", {})
        cell_mass = mass.get("cell_mass")
        if cell_mass is None:
            return None
        if hasattr(cell_mass, "to") and hasattr(cell_mass, "magnitude"):
            return float(cell_mass.to("femtogram").magnitude)
        return float(cell_mass)
    except (AttributeError, KeyError, TypeError):
        return None


def _extract_exchange_fluxes(agent_state: dict | Any) -> dict[str, Any]:
    """Return ``metabolism.external_exchange_fluxes`` map, or {} if absent."""
    try:
        metabolism = agent_state.get("metabolism", {}) if hasattr(agent_state, "get") else {}
        fluxes = metabolism.get("external_exchange_fluxes")
        if isinstance(fluxes, dict):
            return fluxes
        return {}
    except (AttributeError, KeyError, TypeError):
        return {}
