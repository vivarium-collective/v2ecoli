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
    AVOGADRO,
    CO2_EXCHANGE_KEY,
    CO2_ID,
    MW_CO2,
    MW_O2,
    O2_EXCHANGE_KEY,
    O2_ID,
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

def _agent(*, cell_mass_fg: float = 1.0e3, o2_counts: float = 0.0,
           co2_counts: float = 0.0) -> dict:
    """Minimal agent state.

    The coupler reads per-step environmental exchange as molecule COUNTS at
    ``environment.exchange`` (keyed by the bare molecule name; negative ==
    uptake) — v2ecoli's real exchange store. (The cell_mass listener is kept
    for shape realism; the counts-based path does not use it.)
    """
    return {
        "listeners": {"mass": {"cell_mass": cell_mass_fg}},
        "environment": {
            "exchange": {
                O2_EXCHANGE_KEY: o2_counts,
                CO2_EXCHANGE_KEY: co2_counts,
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
    """Negative O2 exchange counts (uptake) drive a NEGATIVE reactor.dissolved_o2
    delta, with the magnitude the counts->mg/L conversion predicts (computed
    explicitly). DO is seeded high enough that the safety clamp does not bind."""
    cells_per_agent = 1.0e9
    o2_counts = -5.0e10          # molecules / step / agent, uptake -> negative
    volume_L = 2.0
    timestep_s = 3600.0          # ignored by the counts path (counts are per-step)

    c = ReactorCellCoupler(
        config={"cells_per_agent": cells_per_agent, "reactor_volume_L": volume_L},
        core=core,
    )
    states = {
        "reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0, "volume_L": volume_L},
        "agents": {
            "0": _agent(o2_counts=o2_counts),
            "1": _agent(o2_counts=o2_counts),
        },
    }
    out = c.next_update(timestep_s, states)

    # Expected, computed explicitly from the counts->mg/L conversion (2 agents):
    counts_to_mgL = cells_per_agent / AVOGADRO * 1000.0 / volume_L * MW_O2
    expected_delta = 2 * o2_counts * counts_to_mgL

    assert expected_delta < 0.0
    assert 100.0 + expected_delta > 0.0, "test design: clamp must not bind"
    assert out["reactor"]["dissolved_o2"] == pytest.approx(expected_delta, rel=1e-9)


def test_co2_evolution_positive(core):
    """Positive CO2 exchange counts (secretion) drive a POSITIVE
    reactor.dissolved_co2 delta."""
    cells_per_agent = 1.0e9
    co2_counts = 6.0e10          # molecules / step / agent, secretion -> positive
    volume_L = 2.0
    timestep_s = 3600.0

    c = ReactorCellCoupler(
        config={"cells_per_agent": cells_per_agent, "reactor_volume_L": volume_L},
        core=core,
    )
    states = {
        "reactor": {"dissolved_o2": 0.0, "dissolved_co2": 0.0, "volume_L": volume_L},
        "agents": {"0": _agent(co2_counts=co2_counts)},
    }
    out = c.next_update(timestep_s, states)

    counts_to_mgL = cells_per_agent / AVOGADRO * 1000.0 / volume_L * MW_CO2
    expected_delta = co2_counts * counts_to_mgL

    assert expected_delta > 0.0
    assert out["reactor"]["dissolved_co2"] == pytest.approx(expected_delta, rel=1e-9)
