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

    The coupler reads CUMULATIVE environmental exchange as molecule COUNTS at
    ``environment.exchange`` (keyed by the bare molecule name; negative ==
    uptake) — v2ecoli's real exchange store, a lineage running total that the
    coupler differences per agent. (The cell_mass listener is kept
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
    # environment.exchange is a lineage RUNNING TOTAL, so drive it as one: tick 1
    # establishes the baseline, tick 2 advances it by `o2_counts`. The Step must
    # consume the DIFFERENCE, not the total.
    c.next_update(timestep_s, states)
    states["agents"] = {
        "0": _agent(o2_counts=2 * o2_counts),
        "1": _agent(o2_counts=2 * o2_counts),
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
    # Running total: baseline tick, then advance by `co2_counts`.
    c.next_update(timestep_s, states)
    states["agents"] = {"0": _agent(co2_counts=2 * co2_counts)}
    out = c.next_update(timestep_s, states)

    counts_to_mgL = cells_per_agent / AVOGADRO * 1000.0 / volume_L * MW_CO2
    expected_delta = co2_counts * counts_to_mgL

    assert expected_delta > 0.0
    assert out["reactor"]["dissolved_co2"] == pytest.approx(expected_delta, rel=1e-9)


# ---------------------------------------------------------------------------
# environment.exchange is a LINEAGE-CUMULATIVE running total, not a per-step
# delta. The Step must difference it. (Before 2026-08-29 it consumed the total
# directly: an N-tick run over-consumed by ~N/2 -- a 22.2 mM glucose pool fully
# drained while the population formed 2.3 mg of biomass.)
# ---------------------------------------------------------------------------

def _glc_agent(total: float) -> dict:
    """Agent whose exchange store carries a cumulative GLC total."""
    return {
        "listeners": {"mass": {"cell_mass": 1.0e3}},
        "environment": {"exchange": {"GLC": total, O2_EXCHANGE_KEY: 0.0,
                                     CO2_EXCHANGE_KEY: 0.0}},
    }


def _medium(c, totals, *, glucose_mM=22.2):
    """Feed a sequence of cumulative GLC totals; return medium mM after each."""
    out_mM = [glucose_mM]
    for total in totals:
        states = {
            "reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                        "volume_L": 1.0, "glucose_medium_mM": out_mM[-1]},
            "population": {"cell_count": 1.0e9},
            "agents": {"0": _glc_agent(total)},
        }
        out = c.next_update(1.0, states)
        out_mM.append(out_mM[-1] + out["reactor"].get("glucose_medium_mM", 0.0))
    return out_mM


def test_consumes_the_tick_delta_not_the_running_total(core):
    """A CONSTANT per-tick uptake must draw the medium down LINEARLY.

    Consuming the running total instead makes the drawdown grow with elapsed
    time -- quadratic, and ~N/2 too large over N ticks. That is the defect:
    the reactor drained by time rather than by cell demand.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core
    )
    # Sized so the medium's zero-clamp cannot bind over the window: at
    # cells_per_agent=1e9 in 1 L, 1 count -> 1.66e-12 mM, so 6e10 counts/tick
    # is ~0.1 mM/tick and five ticks draw down ~0.5 of 22.2 mM. A clamped tick
    # reports a drop of zero and would mask exactly the defect under test.
    step = -6.0e10                       # constant uptake per tick
    totals = [step * i for i in range(1, 6)]   # cumulative: -6e10, -1.2e11, ...
    mM = _medium(c, totals)

    assert mM[-1] > 0.5 * 22.2, (
        f"test design: medium fell to {mM[-1]} -- the zero-clamp bound and "
        "every drop after it reads as zero, masking the defect"
    )
    drops = [mM[i] - mM[i + 1] for i in range(1, len(mM) - 1)]
    assert all(d > 0 for d in drops), "uptake must reduce the medium"
    # Linear drawdown => every per-tick drop is equal.
    assert max(drops) == pytest.approx(min(drops), rel=1e-9), (
        f"drawdown is not constant under constant uptake: {drops} -- the Step "
        "is consuming the cumulative total, not this tick's delta"
    )


def test_first_observation_of_a_new_agent_yields_no_exchange(core):
    """A daughter inherits the parent's cumulative total under a NEW id.

    Differencing that against an assumed zero would dump a whole generation's
    accumulation into one tick -- a spike at every division that reads as a
    huge uptake. The first observation must yield 0.0 instead.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core
    )
    states = {
        "reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                    "volume_L": 1.0, "glucose_medium_mM": 22.2},
        "population": {"cell_count": 1.0e9},
        "agents": {"00": _glc_agent(-9.9e15)},   # inherited, large, never seen
    }
    assert c.track_medium, "test would be vacuous with the medium path off"
    out = c.next_update(1.0, states)
    assert "glucose_medium_mM" in out["reactor"], (
        "medium path did not run -- assertion below would pass vacuously"
    )
    assert out["reactor"]["glucose_medium_mM"] == pytest.approx(0.0)
    assert c.first_observation_ticks == 1


def test_absent_exchange_key_does_not_reset_the_baseline(core):
    """A tick whose snapshot is MISSING the exchange key must draw nothing and
    must leave the baseline untouched.

    ``_extract_environment_exchange``'s own docstring says the emit cadence can
    snapshot mid-init, so a key present last tick can be absent this tick.
    Treating absent as 0.0 would overwrite the baseline with 0.0, emitting a
    spurious SECRETION of the whole running total on that tick and then a
    spurious UPTAKE of the whole total on the next -- reintroducing exactly the
    spike the first-observation rule exists to prevent. The two nominally
    cancel, but the medium and dissolved-gas paths are zero-clamped, and a
    clamped spike does not cancel: it leaves permanent mass error.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core
    )
    base = {"reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                        "volume_L": 1.0, "glucose_medium_mM": 22.2},
            "population": {"cell_count": 1.0e9}}
    c.next_update(1.0, {**base, "agents": {"0": _glc_agent(-1.0e11)}})
    normal = c.next_update(
        1.0, {**base, "agents": {"0": _glc_agent(-2.0e11)}}
    )["reactor"]["glucose_medium_mM"]
    assert normal < 0.0, "no uptake on a normal tick -- assertions below vacuous"

    # Tick 3: the agent is present but its exchange dict has no GLC key.
    gapped = _glc_agent(-2.0e11)
    del gapped["environment"]["exchange"]["GLC"]
    out = c.next_update(1.0, {**base, "agents": {"0": gapped}})
    assert out["reactor"].get("glucose_medium_mM", 0.0) == pytest.approx(0.0), (
        "a tick missing the exchange key must draw nothing"
    )

    # Tick 4: the key is back, one tick further along. The draw must be ONE
    # tick's worth -- not the whole running total re-differenced from zero.
    after = c.next_update(1.0, {**base, "agents": {"0": _glc_agent(-3.0e11)}})
    assert after["reactor"]["glucose_medium_mM"] == pytest.approx(normal), (
        "baseline was reset by the absent key: the tick after a gap drew "
        "the accumulated total instead of one tick's exchange"
    )


def test_division_does_not_double_count_the_inherited_total(core):
    """At division each daughter inherits the parent's cumulative total under a
    NEW id, so the SUMMED total across agents doubles. Differencing the sum
    would charge the reactor one full lineage total on the division tick --
    with the negative-uptake convention that is a spurious extra UPTAKE, not a
    sign flip and not secretion.

    ⚠ This test previously asserted only ``delta <= 0.0``, which is satisfied by
    BOTH the correct 0.0 and the summed mutant's spurious uptake -- so the
    mutation it is named for survived it. It now asserts on MAGNITUDE, against
    a measured normal tick, with that tick doubling as the positive control so
    a coupler that emitted 0.0 unconditionally could not pass either.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core
    )
    base = {"reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                        "volume_L": 1.0, "glucose_medium_mM": 22.2},
            "population": {"cell_count": 1.0e9}}
    # Counts sized so one tick draws ~0.17 mM of the 22.2 mM pool: the medium
    # path is clamped at the remaining pool, and at larger counts every draw
    # below saturates at -22.2 and the magnitude comparisons compare two clamps.
    c.next_update(1.0, {**base, "agents": {"0": _glc_agent(-1.0e11)}})
    normal = c.next_update(
        1.0, {**base, "agents": {"0": _glc_agent(-2.0e11)}}
    )["reactor"]["glucose_medium_mM"]
    # POSITIVE CONTROL: a normal tick must register real uptake, otherwise every
    # magnitude assertion below is vacuous.
    assert normal < 0.0, (
        f"no uptake on a normal tick (got {normal}) -- the assertions below "
        f"would pass vacuously"
    )
    assert normal > -22.2, (
        f"normal tick ({normal} mM) saturated the pool clamp; the magnitude "
        f"assertions below would compare two clamped values, not two draws"
    )

    # Division: parent "0" retires, daughters "00"/"01" each inherit its total.
    out = c.next_update(1.0, {**base, "agents": {"00": _glc_agent(-2.0e11),
                                                 "01": _glc_agent(-2.0e11)}})
    delta = out["reactor"].get("glucose_medium_mM", 0.0)
    # Both daughters are first observations, so the division tick draws NOTHING.
    # Differencing the summed total instead would draw ~2x a normal tick here.
    assert delta == pytest.approx(0.0), (
        f"division tick drew {delta} mM; expected 0.0 (both daughters are first "
        f"observations). A summed-total difference draws ~{2 * normal:.4g} mM here."
    )
    assert abs(delta) < abs(normal), (
        f"division-tick draw {abs(delta)} is not smaller than a normal tick "
        f"{abs(normal)} -- the inherited total is being double-counted"
    )
    assert "0" not in c._prev_exchange, "retired agent id must be dropped"

    # And the tick AFTER division must resume at a normal draw, not carry the
    # doubling forward. It equals ONE normal tick, not two: `cell_count` is the
    # REPRESENTED population and covers all agents, so the per-agent scale is
    # cell_count/n_agents (see test_scale_is_per_agent_not_per_population).
    # Two daughters at an unchanged cell_count therefore draw between them
    # exactly what the single parent drew -- the represented population has not
    # grown just because the agent count did.
    after = c.next_update(1.0, {**base, "agents": {"00": _glc_agent(-3.0e11),
                                                   "01": _glc_agent(-3.0e11)}})
    assert after["reactor"]["glucose_medium_mM"] == pytest.approx(normal), (
        "post-division tick should resume at a normal per-population draw"
    )
