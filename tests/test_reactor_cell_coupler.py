"""Unit tests for the reactor <-> cell coupling (mbp-03).

Two concerns:

* TASK A — ``BiRDTransportProcess`` (pbg-bioreactordesign) is registered as a
  ``local:`` link in :func:`v2ecoli.core.build_core`, so the coupled reactor
  composite can address it (mirrors the KetchupEstimator registration).
* TASK B — :class:`v2ecoli.steps.reactor_cell_coupler.ReactorCellCoupler`
  bridges the two halves each emit cycle with the exact unit conversions
  documented in its module docstring.

These exercise the Step's ``next_update`` math directly against hand-built
``states`` dicts — no composite, no ParCa cache (that wiring is a later task).
"""

from __future__ import annotations

import pytest

from process_bigraph import Composite

from v2ecoli.core import build_core
from v2ecoli.steps.reactor_cell_coupler import (
    CO2_ID,
    MW_CO2,
    MW_O2,
    O2_ID,
    SECONDS_PER_HOUR,
    ReactorCellCoupler,
)


# Shared core (build_core is expensive).
@pytest.fixture(scope="module")
def core():
    return build_core()


# --- TASK A: BiRDTransportProcess registration ------------------------------

def test_bird_transport_registered(core):
    """build_core() must register BiRDTransportProcess as a local link so a
    composite doc addressing ``local:BiRDTransportProcess`` instantiates.
    Mirrors the KetchupEstimator registration precedent in core.py."""
    from pbg_bioreactordesign import BiRDTransportProcess  # import guard

    doc = {
        "reactor_transport": {
            "_type": "process",
            "address": "local:BiRDTransportProcess",
            "config": {},
            "inputs": {},
            "outputs": {},
        }
    }
    # Resolution-only check: if the link is registered, Composite build does
    # not raise on the unknown address.
    sim = Composite({"state": doc}, core=core)
    assert sim is not None


# --- TASK B: ReactorCellCoupler -------------------------------------------

def _agent(*, cell_mass_fg: float, o2_flux: float = 0.0, co2_flux: float = 0.0) -> dict:
    """Minimal agent state: cell_mass listener + metabolic exchange flux map."""
    return {
        "listeners": {"mass": {"cell_mass": cell_mass_fg}},
        "metabolism": {
            "external_exchange_fluxes": {
                O2_ID: o2_flux,
                CO2_ID: co2_flux,
            }
        },
    }


def test_biomass_passthrough(core):
    """population.biomass_concentration_gL flows straight to reactor.biomass."""
    c = ReactorCellCoupler(config={}, core=core)
    out = c.next_update(60.0, {"population": {"biomass_concentration_gL": 5.0}})
    assert out["reactor"]["biomass"] == pytest.approx(5.0)


def test_mgL_to_mM_conversion(core):
    """reactor.dissolved_o2 = 8.0 mg/L -> external O2 conc = 8.0 / MW_O2 mM."""
    c = ReactorCellCoupler(config={}, core=core)
    out = c.next_update(
        60.0,
        {"reactor": {"dissolved_o2": 8.0, "dissolved_co2": 0.0, "volume_L": 1.0}},
    )
    ext = out["environment"]["external_concentrations"]
    assert ext[O2_ID] == pytest.approx(8.0 / 31.999, abs=1e-6)
    assert ext[CO2_ID] == pytest.approx(0.0, abs=1e-6)


def test_o2_uptake_decreases_reactor_o2(core):
    """Negative O2 flux (uptake) drives a NEGATIVE reactor.dissolved_o2 delta,
    with the magnitude the conversion chain predicts (computed explicitly)."""
    cells_per_agent = 1.0e9
    cell_mass_fg = 1.0e3          # 1000 fg / cell
    o2_flux = -10.0              # mmol / (gDW * h), uptake -> negative
    volume_L = 2.0
    timestep_s = 3600.0          # 1 hour

    c = ReactorCellCoupler(
        config={"cells_per_agent": cells_per_agent, "reactor_volume_L": volume_L},
        core=core,
    )
    states = {
        "reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0, "volume_L": volume_L},
        "agents": {
            "0": _agent(cell_mass_fg=cell_mass_fg, o2_flux=o2_flux),
            "1": _agent(cell_mass_fg=cell_mass_fg, o2_flux=o2_flux),
        },
    }
    out = c.next_update(timestep_s, states)

    # Expected, computed explicitly:
    biomass_gDW = cell_mass_fg * cells_per_agent * 1.0e-15          # per agent
    interval_h = timestep_s / SECONDS_PER_HOUR
    sum_o2_mmol_per_h = 2 * (o2_flux * biomass_gDW)                 # 2 agents
    expected_delta = sum_o2_mmol_per_h * interval_h * MW_O2 / volume_L

    assert expected_delta < 0.0
    assert out["reactor"]["dissolved_o2"] == pytest.approx(expected_delta, rel=1e-9)


def test_co2_evolution_positive(core):
    """Positive CO2 flux (evolution) drives a POSITIVE reactor.dissolved_co2
    delta."""
    cells_per_agent = 1.0e9
    cell_mass_fg = 1.0e3
    co2_flux = 12.0             # mmol / (gDW * h), secretion -> positive
    volume_L = 2.0
    timestep_s = 3600.0

    c = ReactorCellCoupler(
        config={"cells_per_agent": cells_per_agent, "reactor_volume_L": volume_L},
        core=core,
    )
    states = {
        "reactor": {"dissolved_o2": 0.0, "dissolved_co2": 0.0, "volume_L": volume_L},
        "agents": {"0": _agent(cell_mass_fg=cell_mass_fg, co2_flux=co2_flux)},
    }
    out = c.next_update(timestep_s, states)

    biomass_gDW = cell_mass_fg * cells_per_agent * 1.0e-15
    interval_h = timestep_s / SECONDS_PER_HOUR
    expected_delta = (co2_flux * biomass_gDW) * interval_h * MW_CO2 / volume_L

    assert expected_delta > 0.0
    assert out["reactor"]["dissolved_co2"] == pytest.approx(expected_delta, rel=1e-9)
