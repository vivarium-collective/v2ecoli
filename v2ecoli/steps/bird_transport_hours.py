"""BiRDTransportHours — seconds->hours time-base adapter for BiRDTransportProcess.

v2ecoli's whole-cell model steps in SECONDS. pbg-bioreactordesign's
:class:`BiRDTransportProcess` does its explicit-Euler gas-liquid transport math
in HOURS: ``dC = kLa[1/h] * (C* - C) * interval`` with ``interval`` treated as
hours (see ``pbg_bioreactordesign/processes.py::BiRDTransportProcess.update``).
process-bigraph passes a process its ``interval`` in global-time units — here
SECONDS — straight into that hours-based math, so the raw process transports
~3600x too fast and the dissolved-gas stores diverge/oscillate (which the
composite previously masked with a ``DEFAULT_TRANSPORT_INTERVAL = 0.01`` damping
hack, not a fix).

This thin v2ecoli-side subclass is the proper time-base bridge: it intercepts
``update`` and converts the incoming per-second interval to hours (``/ 3600``)
before delegating to the upstream transport update. Every input/output port,
config field, and the transport physics itself are inherited unchanged — only
the time unit of the interval is corrected.

This mirrors :class:`ReactorCellCoupler`'s own ``timestep / SECONDS_PER_HOUR``
conversion (it scales per-hour metabolic fluxes into per-step deltas), so after
this adapter BOTH the reactor transport deltas and the cell-consumption deltas
are accumulated on the same per-second basis — they net additively at the shared
``reactor.dissolved_o2`` / ``reactor.dissolved_co2`` stores correctly.
"""

from __future__ import annotations

from viva_bioreactordesign import BiRDTransportProcess

# Convention shared with v2ecoli.steps.reactor_cell_coupler.SECONDS_PER_HOUR.
SECONDS_PER_HOUR: float = 3600.0


class BiRDTransportHours(BiRDTransportProcess):
    """``BiRDTransportProcess`` with its update interval read in SECONDS.

    The only behavioral change vs the upstream process: ``update`` receives a
    per-second ``interval`` (process-bigraph global-time units in v2ecoli) and
    converts it to hours before the upstream hours-based transport math runs.
    All ports and physics are inherited verbatim.
    """

    name = "BiRDTransportHours"

    description = (
        "BiRD reactor gas-liquid mass-transport node with a seconds->hours "
        "time-base adapter. Each step it computes the volumetric mass-transfer "
        "coefficient kLa and the Henry's-law saturation concentrations for O2 "
        "and CO2 from the reactor geometry, aeration, and hydrodynamics, then "
        "applies an explicit-Euler transfer delta kLa*(C*-C)*dt to the dissolved "
        "gas stores. All ports and transport physics are inherited verbatim from "
        "BiRDTransportProcess; only the interval unit is corrected (s -> h)."
    )

    # Structured contract surfaced by the workbench loom viewer (card contract
    # band + Inspector) and by bigraph_schema.contract.resolve_contract. Ports
    # are inherited from BiRDTransportProcess; documented here with units.
    contract = {
        "summary": (
            "Gas-liquid mass transport for the BiRD reactor: kLa + Henry's-law "
            "saturation drive an explicit-Euler transfer of O2/CO2 between the "
            "sparged gas phase and the dissolved liquid stores, on a per-second "
            "time base."
        ),
        "inputs": {
            "dissolved_o2": "Current dissolved O2 concentration in the liquid (mg/L).",
            "dissolved_co2": "Current dissolved CO2 concentration in the liquid (mg/L).",
            "biomass": "Biomass concentration (g/L) driving O2/glucose uptake and CO2 evolution.",
            "glucose": "Dissolved glucose concentration the reactor tracks (mM).",
            "gas_flow_rate_Lpm": "Sparged aeration flow rate (L/min).",
        },
        "outputs": {
            "dissolved_o2": "Updated dissolved O2 concentration after this step's transfer (mg/L).",
            "dissolved_co2": "Updated dissolved CO2 concentration after this step's transfer (mg/L).",
            "o2_transport_delta": "Change in dissolved O2 this step from gas-liquid transfer (mg/L), kLa*(C*-C)*dt.",
            "co2_transport_delta": "Change in dissolved CO2 this step from gas-liquid transfer (mg/L), kLa*(C*-C)*dt.",
            "glucose_transport": "Update to the dissolved glucose the reactor tracks (mM).",
            "biomass_transport_delta": "Change in biomass concentration this step (g/L).",
            "kla_o2": "Volumetric mass-transfer coefficient kLa for O2 (1/h), from the correlation + hydrodynamics.",
            "kla_co2": "Volumetric mass-transfer coefficient kLa for CO2 (1/h), from the correlation + hydrodynamics.",
            "o2_saturation": "Equilibrium/saturation dissolved O2 concentration C* at the gas-phase partial pressure via Henry's law (mg/L).",
            "co2_saturation": "Equilibrium/saturation dissolved CO2 concentration C* at the gas-phase partial pressure via Henry's law (mg/L).",
            "gas_holdup": "Fraction of reactor volume occupied by gas bubbles (dimensionless).",
            "superficial_gas_velocity": "Gas volumetric flow divided by vessel cross-section (m/s).",
        },
        "config": {
            "reactor_type": "Reactor geometry model, e.g. 'bubble_column' or 'stirred_tank'.",
            "volume_L": "Liquid working volume (L).",
            "diameter_m": "Vessel inner diameter (m).",
            "liquid_height_m": "Liquid column height (m).",
            "gas_flow_rate_Lpm": "Default sparged aeration flow (L/min).",
            "temperature_K": "Operating temperature (K).",
            "pressure_atm": "Operating pressure (atm).",
            "o2_fraction_inlet": "O2 mole fraction of the inlet gas (dimensionless).",
            "co2_fraction_inlet": "CO2 mole fraction of the inlet gas (dimensionless).",
            "mean_bubble_diameter_mm": "Sauter-mean bubble diameter (mm).",
            "impeller_power_W": "Stirred-tank impeller power input (W).",
            "kla_correlation": "Name of the kLa correlation to use, e.g. 'auto' or 'vant_riet'.",
        },
        "assumptions": [
            "Transport uses explicit-Euler integration dC = kLa*(C*-C)*dt over the step.",
            "The interval arrives in SECONDS (v2ecoli global time) and is converted to HOURS before the upstream hours-based math runs.",
            "Henry's-law saturation assumes local gas-liquid equilibrium at the operating temperature and pressure.",
        ],
    }

    def update(self, state, interval):
        # interval arrives in SECONDS (v2ecoli global time); the upstream
        # transport update expects HOURS.
        return super().update(state, interval / SECONDS_PER_HOUR)
