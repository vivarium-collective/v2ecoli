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
    AMMONIUM_MEDIUM_LEAF,
    BIOMASS_C_FRACTION,
    BIOMASS_N_FRACTION,
    DIAGNOSTIC_LEAVES,
    MW_C,
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


def _byproduct_agent(acet: float, suc: float = 0.0) -> dict:
    """Agent whose exchange store carries cumulative BYPRODUCT totals
    (positive == secreted to the environment)."""
    return {
        "listeners": {"mass": {"cell_mass": 1.0e3}},
        "environment": {"exchange": {"ACET": acet, "SUC": suc,
                                     O2_EXCHANGE_KEY: 0.0,
                                     CO2_EXCHANGE_KEY: 0.0}},
    }


def _ledger_coupler(core):
    return ReactorCellCoupler(
        config={"cells_per_agent": 1.0e12, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core)


def _ledger_states(*, glc, nh4, biomass_gL, glc_total, nh4_total, co2_total=0.0):
    """One tick of states for the elemental ledger.

    ``*_total`` are CUMULATIVE exchange counts (the store is a running total).
    """
    return {
        "reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0, "volume_L": 1.0,
                    "glucose_medium_mM": glc, "ammonium_medium_mM": nh4,
                    "acetate_mM": 0.0, "lactate_mM": 0.0, "formate_mM": 0.0,
                    "ethanol_mM": 0.0, "pyruvate_mM": 0.0, "succinate_mM": 0.0},
        "population": {"cell_count": 1.0e12,
                       "biomass_concentration_gL": biomass_gL},
        "agents": {"0": {
            "listeners": {"mass": {"cell_mass": 1.0e3}},
            "environment": {"exchange": {
                "GLC": glc_total, "AMMONIUM": nh4_total,
                O2_EXCHANGE_KEY: 0.0, CO2_EXCHANGE_KEY: co2_total}}}},
    }


def test_ledger_baseline_waits_for_a_populated_population_store(core):
    """The ledger must not latch its baseline on a pre-population tick.

    ⚠ The Step flow runs once at CYCLE START, before the PopulationAggregator
    has written anything, so the first invocation sees
    ``biomass_concentration_gL == 0.0``. Latching there sets b0 = 0 and every
    later tick charges the INOCULUM against consumed glucose. Measured before
    the fix: carbon_residual -8.48 with carbon_biomass_mM (16.3) an order of
    magnitude above carbon_in_mM (1.7) -- i.e. more carbon in cells than was
    ever consumed, which is the shape of the bug.
    """
    c = _ledger_coupler(core)
    # Tick 1: population store still empty (the pre-population tick).
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.0,
                                      glc_total=0.0, nh4_total=0.0))
    assert c._c_ledger_glc0 is None, (
        "baseline latched on a tick where biomass_concentration_gL == 0.0")

    # Tick 2: aggregator has run. Baselines latch together, at this instant.
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    assert c._c_ledger_glc0 == pytest.approx(40.0)
    assert c._c_ledger_b0 == pytest.approx(0.5), (
        "biomass baseline must be the standing inoculum, not 0.0")


def test_a_tick_with_no_ledger_emits_NaN_not_zero(core):
    """0.0 must mean "balanced", never "no data".

    ⚠ THE DEFECT THIS PINS: the residual was `((c_in-c_out)/c_in) if c_in > 0
    else 0.0`, so a tick that carried NO ledger emitted the exact value that
    means PERFECT CLOSURE. The three partition fractions did the same. An
    instrument built to catch "a passing gate that cannot see the failure"
    contained one.
    """
    import math
    c = _ledger_coupler(core)
    # First tick: baseline latches here, nothing consumed yet -> no ledger.
    out = c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                            glc_total=0.0, nh4_total=0.0))
    d = out["reactor"]["diagnostics"]
    assert d["ledger_valid"] == 0.0, f"tick with nothing consumed must be invalid: {d}"
    for leaf in ("carbon_residual", "nitrogen_residual", "carbon_to_biomass_frac",
                 "carbon_to_co2_frac", "carbon_to_byproducts_frac"):
        assert math.isnan(d[leaf]), (
            f"{leaf} must be NaN on an invalid tick, got {d[leaf]!r} -- a real "
            f"number here is indistinguishable from a balanced ledger")
    # The COMPONENTS still come through: they are what diagnose why it is invalid.
    assert d["carbon_in_mM"] == pytest.approx(0.0, abs=1e-12)


def test_biomass_below_its_own_baseline_invalidates_the_tick(core):
    """A negative biomass delta must not produce a confident partition.

    ⚠ `[m@31Aug]` the population store DIPS below its latched baseline early
    (0.38062 -> 0.37981, ~11 ticks to recover), driving carbon_biomass_mM
    negative -- 11 of 120 ticks, at t=12 bio=-0.175 against byp=+1.175. Those
    fractions summed to 1.0 the whole time, so a sum-only assertion passed on
    a physically impossible split.
    """
    import math
    counts_per_mM = AVOGADRO / 1000.0 / 1.0e12
    c = _ledger_coupler(core)
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    # Consume glucose, but biomass DROPS below the latched 0.5 baseline.
    out = c.next_update(1.0, _ledger_states(
        glc=40.0, nh4=30.0, biomass_gL=0.49,
        glc_total=-1.0 * counts_per_mM, nh4_total=0.0))
    d = out["reactor"]["diagnostics"]
    assert d["carbon_biomass_mM"] < 0.0, (
        "precondition: this tick must actually have negative biomass carbon, "
        f"else the test is not exercising the defect: {d}")
    assert d["ledger_valid"] == 0.0, (
        f"a tick whose biomass is below its own baseline must be invalid: {d}")
    assert math.isnan(d["carbon_residual"])
    assert math.isnan(d["carbon_to_biomass_frac"])


def test_the_writer_emits_exactly_the_enumerated_leaves(core):
    """DIAGNOSTIC_LEAVES exists to stop drift; it has to actually be checked.

    A leaf written but not enumerated is SILENTLY DROPPED by the InPlaceDict
    port; a leaf enumerated but never written keeps a stale value forever.
    The writer previously used literal string keys, so the constant introduced
    to prevent that drift did not prevent it.
    """
    c = _ledger_coupler(core)
    out = c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                            glc_total=0.0, nh4_total=0.0))
    assert tuple(out["reactor"]["diagnostics"]) == DIAGNOSTIC_LEAVES


def test_carbon_ledger_closes_on_a_stoichiometric_tick(core):
    """A hand-built tick whose carbon balances must give residual ~0.

    Glucose consumed is converted to biomass at exactly BIOMASS_C_FRACTION,
    so in == out by construction and the residual must vanish. This is the
    positive control: without it every assertion about a FAILING residual
    could be satisfied by a ledger that always reports failure.
    """
    c = _ledger_coupler(core)
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    # Consume 1 mM glucose == 6 mM carbon; convert all of it to biomass.
    counts_per_mM = AVOGADRO / 1000.0 / 1.0e12          # counts per mM at cpa=1e12
    # ⚠⚠ THIS LITERAL IS THE POINT OF THE TEST. An earlier version computed it as
    # `6.0 * MW_C / 1000.0 / BIOMASS_C_FRACTION` -- i.e. it derived the expected
    # biomass FROM the very constant the ledger uses, so both sides of the
    # balance moved together and the residual closed for ANY value of that
    # constant. Mutating BIOMASS_C_FRACTION survived it. A positive control that
    # cannot detect a wrong constant is not a control.
    # 6 mM C x 12.011 mg/mmol / 1000 = 0.0720660 gC/L; / 0.46 gC/gDW = 0.15666522.
    d_biomass_gL = 0.15666521739130435
    out = c.next_update(1.0, _ledger_states(
        glc=40.0, nh4=30.0, biomass_gL=0.5 + d_biomass_gL,
        glc_total=-1.0 * counts_per_mM, nh4_total=0.0))
    d = out["reactor"]["diagnostics"]
    assert d["ledger_valid"] == 1.0, f"stoichiometric tick should be valid: {d}"
    assert d["carbon_in_mM"] == pytest.approx(6.0, rel=1e-6)
    assert d["carbon_residual"] == pytest.approx(0.0, abs=1e-6), (
        f"stoichiometric tick did not close: {d}")
    # Pin the constant itself, independently of the balance above.
    assert BIOMASS_C_FRACTION == pytest.approx(0.46, abs=1e-9), (
        "BIOMASS_C_FRACTION moved; the literal above was derived from 0.46 and "
        "must be recomputed, or the ledger is being graded against a constant "
        "no test pins")


def test_nitrogen_ledger_closes_on_a_stoichiometric_tick(core):
    """The nitrogen twin of the carbon positive control, and it was MISSING.

    ⚠⚠ WHY THIS EXISTS: mutating BIOMASS_N_FRACTION from 0.135 back to the old
    0.12 passed the ENTIRE suite (measured 2026-09-01). Nothing anywhere pinned
    the nitrogen constant -- the same vacuous-control defect the carbon test had,
    sitting on the constant this PR CHANGES. A value nothing tests is a value
    nobody can trust, least of all the person who just moved it.

    Consume 1 mM ammonium == 1 mM N and convert exactly that much to biomass, so
    in == out by construction and the nitrogen residual must vanish.
    """
    counts_per_mM = AVOGADRO / 1000.0 / 1.0e12
    c = _ledger_coupler(core)
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    # ⚠ LITERAL ON PURPOSE -- deriving it from BIOMASS_N_FRACTION would move both
    # sides of the balance together and could not detect a wrong constant.
    # 1 mM N x 14.007 mg/mmol / 1000 = 0.014007 gN/L; / 0.135 gN/gDW = 0.10375556.
    d_biomass_gL = 0.10375555555555555
    out = c.next_update(1.0, _ledger_states(
        glc=40.0, nh4=30.0, biomass_gL=0.5 + d_biomass_gL,
        # glucose is consumed only so the tick is VALID (validity requires
        # c_in > 0); the carbon balance is deliberately not asserted here.
        glc_total=-1.0 * counts_per_mM, nh4_total=-1.0 * counts_per_mM))
    d = out["reactor"]["diagnostics"]
    assert d["ledger_valid"] == 1.0, f"tick should be valid: {d}"
    assert d["nitrogen_in_mM"] == pytest.approx(1.0, rel=1e-6)
    assert d["nitrogen_residual"] == pytest.approx(0.0, abs=1e-6), (
        f"stoichiometric nitrogen tick did not close: {d}")
    # Pin the constant independently of the balance above.
    assert BIOMASS_N_FRACTION == pytest.approx(0.135, abs=1e-9), (
        "BIOMASS_N_FRACTION moved; the literal above was derived from 0.135 and "
        "must be recomputed. sim_data implies ~0.135; the 0.12 this shipped with "
        "is ~11% low, and the literature range 0.11-0.14 brackets both, which is "
        "exactly why the error was invisible")


def test_carbon_partition_is_reported_beside_the_residual(core):
    """Closure alone does not validate the partition, so the split is reported.

    ⚠ Measured on a real run: carbon closes to within ~2% while ~95% of glucose
    carbon goes to BIOMASS and ~4% to CO2, against 40-50% to CO2 for real
    aerobic growth. A near-zero residual on a physically impossible split is a
    gate that cannot see the failure -- the fractions are what make it visible.
    """
    c = _ledger_coupler(core)
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    counts_per_mM = AVOGADRO / 1000.0 / 1.0e12
    # Split the tick's carbon deliberately: 4 mM C to biomass, 2 mM C to CO2.
    # ⚠ An earlier version of this test used a tick with NO CO2, so
    # carbon_to_co2_frac was 0.0 whether computed or hardcoded and the
    # assertion could not fail -- the mutation "co2 fraction := 0.0" survived
    # it. The fraction under test must be NON-ZERO for the test to discriminate.
    d_biomass_gL = 4.0 * MW_C / 1000.0 / BIOMASS_C_FRACTION
    out = c.next_update(1.0, _ledger_states(
        glc=40.0, nh4=30.0, biomass_gL=0.5 + d_biomass_gL,
        glc_total=-1.0 * counts_per_mM, nh4_total=0.0,
        co2_total=+2.0 * counts_per_mM))
    d = out["reactor"]["diagnostics"]
    fracs = (d["carbon_to_biomass_frac"], d["carbon_to_co2_frac"],
             d["carbon_to_byproducts_frac"])
    assert sum(fracs) == pytest.approx(1.0, abs=1e-9), (
        f"carbon-out fractions must partition to 1.0, got {fracs}")
    # ⚠⚠ SUMMING TO 1.0 IS THE WRONG INVARIANT ON ITS OWN, and asserting only it
    # is how this test passed while the split was nonsense. `[m@31Aug]` on a real
    # run, 11 of 120 ticks emitted a NEGATIVE carbon_to_biomass_frac against a
    # >1 byproducts frac (t=12: bio=-0.175, byp=+1.175) -- summing to exactly
    # 1.0 the whole time. Assert the VALUES and their RANGE, not the sum.
    assert all(0.0 <= f <= 1.0 for f in fracs), (
        f"a carbon-out fraction is outside [0,1] -- the partition is not "
        f"physical even if it sums to 1.0: {fracs}")
    assert d["carbon_to_biomass_frac"] == pytest.approx(4.0 / 6.0, rel=1e-6), (
        f"4 of 6 mM C went to biomass; got {d['carbon_to_biomass_frac']}")
    assert d["carbon_to_co2_frac"] == pytest.approx(2.0 / 6.0, rel=1e-6), (
        f"2 of 6 mM C went to CO2; got {d['carbon_to_co2_frac']}")
    # POSITIVE CONTROL on the axis under test: CO2 must be a real share here.
    assert d["carbon_to_co2_frac"] == pytest.approx(2.0 / 6.0, rel=1e-6), (
        f"CO2 share should be 2 of 6 mM carbon out, got {d['carbon_to_co2_frac']}")
    assert d["carbon_to_biomass_frac"] == pytest.approx(4.0 / 6.0, rel=1e-6)


def test_ammonium_is_drawn_down_like_glucose(core):
    """Ammonium must be tracked on the same footing as glucose.

    Without it the nitrogen ledger has no input term and mbp-04's declared
    nitrogen_residual criterion cannot be evaluated.
    """
    c = _ledger_coupler(core)
    c.next_update(1.0, _ledger_states(glc=40.0, nh4=30.0, biomass_gL=0.5,
                                      glc_total=0.0, nh4_total=0.0))
    counts_per_mM = AVOGADRO / 1000.0 / 1.0e12
    out = c.next_update(1.0, _ledger_states(
        glc=40.0, nh4=30.0, biomass_gL=0.5,
        glc_total=0.0, nh4_total=-2.0 * counts_per_mM))
    assert out["reactor"][AMMONIUM_MEDIUM_LEAF] == pytest.approx(-2.0, rel=1e-6), (
        "ammonium uptake did not draw down the medium pool")
    assert out["reactor"]["diagnostics"]["nitrogen_in_mM"] == pytest.approx(2.0, rel=1e-6)


def test_byproducts_are_differenced_not_consumed_as_a_running_total(core):
    """The byproduct leaves are a running total too, and must be differenced.

    ⚠ This test exists because the fix shipped WITHOUT it. Reverting
    ``byproduct_counts[leaf] += _tick_delta(key)`` to
    ``+= _as_float(exch.get(key, 0.0))`` -- the exact defect this change
    corrects, applied to the six byproduct leaves -- passed the entire suite.
    The O2/CO2/glucose arm was covered; this one was not.

    Secretion accumulates monotonically, so an undifferenced read reports the
    WHOLE lineage total as this tick's secretion: acetate would climb with
    elapsed time rather than with what the cells actually excreted, which is
    the same failure as the glucose side with the sign reversed.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0,
                "track_medium": True},
        core=core
    )
    base = {"reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                        "volume_L": 1.0, "glucose_medium_mM": 22.2},
            "population": {"cell_count": 1.0e9}}

    c.next_update(1.0, {**base, "agents": {"0": _byproduct_agent(1.0e11)}})
    out2 = c.next_update(1.0, {**base, "agents": {"0": _byproduct_agent(2.0e11)}})
    step = out2["reactor"]["acetate_mM"]
    # POSITIVE CONTROL: the leaf must actually be written and be a secretion.
    assert step > 0.0, (
        f"no acetate secretion registered (got {step}) -- the assertions below "
        f"would pass vacuously"
    )

    # Third tick advances the total by the SAME increment, so the per-tick
    # secretion must be unchanged. An undifferenced read would report 3.0e11
    # (the running total) instead of 1.0e11 -- 3x this value and climbing.
    out3 = c.next_update(1.0, {**base, "agents": {"0": _byproduct_agent(3.0e11)}})
    assert out3["reactor"]["acetate_mM"] == pytest.approx(step), (
        f"acetate secretion grew with elapsed time ({out3['reactor']['acetate_mM']} "
        f"vs {step}) -- the byproduct leaf is being read as a per-tick value when "
        f"it is a cumulative total"
    )

    # A tick with NO further secretion must report zero, not the standing total.
    out4 = c.next_update(1.0, {**base, "agents": {"0": _byproduct_agent(3.0e11)}})
    assert out4["reactor"]["acetate_mM"] == pytest.approx(0.0), (
        "a tick with no new secretion still reported acetate -- the running "
        "total is being consumed instead of differenced"
    )

    # A second byproduct leaf, to catch a fix applied to only one of the six.
    out5 = c.next_update(1.0, {**base, "agents": {"0": _byproduct_agent(3.0e11, suc=1.0e11)}})
    assert out5["reactor"]["succinate_mM"] == pytest.approx(step), (
        "succinate is not differenced the way acetate is"
    )


def test_first_observation_counter_counts_agents_not_ticks(core):
    """``first_observation_ticks`` must count NEW AGENTS, not ticks.

    ⚠ Regression for a defect introduced by the absent-key guard: because
    ``_tick_delta`` returns before writing ``prev[key]``, an agent whose
    snapshot carries none of the coupler's keys leaves ``prev`` empty, so a
    ``not prev`` test reads it as new on EVERY tick. The counter is diagnostic
    only -- which is exactly why a wrong value would be believed.
    """
    c = ReactorCellCoupler(
        config={"cells_per_agent": 1.0e9, "reactor_volume_L": 1.0},
        core=core
    )
    base = {"reactor": {"dissolved_o2": 100.0, "dissolved_co2": 0.0,
                        "volume_L": 1.0},
            "population": {"cell_count": 1.0e9}}
    # An agent present for three ticks with an EMPTY exchange store.
    empty = {"listeners": {"mass": {"cell_mass": 1.0e3}},
             "environment": {"exchange": {}}}
    for _ in range(3):
        c.next_update(1.0, {**base, "agents": {"0": empty}})
    assert c.first_observation_ticks == 1, (
        f"counter reached {c.first_observation_ticks} for ONE agent over three "
        f"ticks -- it is counting ticks, not new agents, so it can no longer "
        f"distinguish many divisions from one agent with no exchange data"
    )


@pytest.mark.parametrize("gap_kind", ["absent", "none_valued"])
def test_unusable_exchange_key_does_not_reset_the_baseline(core, gap_kind):
    """A tick whose exchange reading is unusable -- key MISSING, or present but
    ``None`` -- must draw nothing and must leave the baseline untouched.

    ⚠ ``none_valued`` is parametrized in separately because the first version of
    this guard tested ``key not in exch`` only, which lets a present ``None``
    through to ``_as_float`` (which maps it to 0.0) and reproduces the spike
    verbatim: measured +0.332 mM spurious secretion, then -0.498 mM spurious
    uptake, against a -0.166 mM normal tick. The fix was made without this case
    and the whole suite still passed.

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

    # Tick 3: the agent is present but its GLC reading is unusable.
    gapped = _glc_agent(-2.0e11)
    if gap_kind == "absent":
        del gapped["environment"]["exchange"]["GLC"]
    else:
        gapped["environment"]["exchange"]["GLC"] = None
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
        f"observations). A summed-total difference draws ~{normal:.4g} mM here "
        f"(1x a normal tick, not 2x: cells_per_agent_effective = cell_count / "
        f"n_agents, so the doubled counts and the halved scale cancel)."
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
