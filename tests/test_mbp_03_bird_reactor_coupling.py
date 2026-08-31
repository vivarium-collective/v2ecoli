"""Behavior tests for mbp-03-bird-reactor-coupling.

Each test maps to a ``behavior_test`` in
``workspace/studies/mbp-03-bird-reactor-coupling/study.yaml``. They exercise the
``reactor_bird_coupled`` composite — v2ecoli's single-cell WCM + population
aggregator coupled to pbg-bioreactordesign's gas-liquid transport via the
``BiRDTransportHours`` time-base adapter and the ``ReactorCellCoupler`` Step.

Time-base fix (mbp-03 T5, PART 1)
---------------------------------
v2ecoli steps in SECONDS; ``BiRDTransportProcess`` does its explicit-Euler
transport math in HOURS. process-bigraph passes the per-second interval straight
into the hours-based math, so the raw process over-transports ~3600x and the
dissolved-gas stores diverge. ``v2ecoli.steps.bird_transport_hours.BiRDTransportHours``
converts the per-second interval to hours (/3600) before delegating, mirroring
ReactorCellCoupler's own timestep/3600 conversion so both transport deltas and
cell-consumption deltas land on the same per-second basis.
``test_time_base_adapter_*`` is the unit-level proof.

Consumption-path fix (discovered during T5)
-------------------------------------------
The coupler originally read ``agents.*.metabolism.external_exchange_fluxes`` — a
path v2ecoli's live state never populates, so cell O2 consumption was a silent
no-op (DO pinned at saturation regardless of cell density). v2ecoli's real
per-step exchange is at ``agents.*.environment.exchange`` in molecule COUNTS
(negative == uptake). The coupler now reads that and converts counts -> mg/L via
Avogadro's number; ``test_o2_mass_balance_closes`` validates the conversion.

Runtime note: the coupled tests build + run the full WCM (~0.2-0.35 s/step), so
they are marked ``slow``/``sim`` and use module-scoped fixtures to share runs.
The per-test wall-clock cap is 120 s (pyproject ``timeout``); each fixture is
sized to stay well under it.
"""

from __future__ import annotations

import math
import statistics

import pytest

from v2ecoli import build_composite
from v2ecoli.core import build_core
from v2ecoli.steps.bird_transport_hours import BiRDTransportHours, SECONDS_PER_HOUR
from v2ecoli.steps.reactor_cell_coupler import AVOGADRO, MW_O2, O2_EXCHANGE_KEY


pytestmark = [pytest.mark.sim, pytest.mark.slow]


# --- helpers ---------------------------------------------------------------

def _f(x) -> float:
    """Coerce a possibly pint-Quantity scalar to a bare float."""
    return float(x.magnitude) if hasattr(x, "magnitude") else float(x)


def _run_coupled(cpa: float, gas_flow_Lpm: float, n_steps: int) -> dict:
    """Build + step the coupled composite, recording per-step reactor series.

    Returns a dict with the recorded trajectories + scalars needed by the
    behavior tests (DO series, transport-delta series, O2-uptake counts, the
    saturation references, biomass, and population cell count).
    """
    c = build_composite(
        "reactor_bird_coupled", seed=0, cache_dir="out/cache",
        cells_per_agent=cpa,
        bird_reactor_config={"gas_flow_rate_Lpm": gas_flow_Lpm},
    )
    do0 = _f(c.state["reactor"]["dissolved_o2"])
    do_series: list[float] = []
    co2_series: list[float] = []
    o2_transport_deltas: list[float] = []
    o2_counts: list[float] = []
    # ⚠ Anchor the exchange baseline BEFORE the loop, symmetrically with do0.
    # The store is a lineage-cumulative total, so the balance differences its
    # endpoints -- and if the first sample were taken AFTER tick 0 while do0 was
    # taken before it, tick 0's consumption would be differenced out of the
    # numerator while remaining in the denominator. That is not a rounding
    # concern: the WCM's first FBA step is a transient ~3500x a steady tick
    # (937,795 counts vs ~266), so the asymmetry alone read as a 5.3% mass-balance
    # deficit (ratio 0.9466) and was mistaken for a physical leak.
    o2_counts.append(float(
        c.state["agents"]["0"]["environment"]["exchange"].get(O2_EXCHANGE_KEY, 0.0)))
    for _ in range(n_steps):
        c.run(1)
        r = c.state["reactor"]
        do_series.append(_f(r["dissolved_o2"]))
        co2_series.append(_f(r["dissolved_co2"]))
        o2_transport_deltas.append(_f(r["o2_transport_delta"]))
        exch = c.state["agents"]["0"]["environment"]["exchange"]
        o2_counts.append(float(exch.get(O2_EXCHANGE_KEY, 0.0)))
    r = c.state["reactor"]
    return {
        "do0": do0,
        "do_final": _f(r["dissolved_o2"]),
        "do_series": do_series,
        "co2_series": co2_series,
        "o2_transport_deltas": o2_transport_deltas,
        "o2_counts": o2_counts,
        "cstar_o2": _f(r["o2_saturation"]),
        "cstar_co2": _f(r["co2_saturation"]),
        "kla_o2": _f(r["kla_o2"]),
        "biomass": _f(r["biomass"]),
        "cell_count": _f(c.state["population"]["cell_count"]),
        "cells_per_agent": cpa,
        "volume_L": _f(r.get("volume_L", 1.0)),
    }


# --- module-scoped runs (shared across tests; each < 120 s setup) ----------

# A representative coupled run: cell density scaled to ~0.025 g/L so cell O2
# demand is comparable to (but below) the kLa transfer ceiling — DO drops below
# saturation but the dissolved store stays strictly positive (no clamp). Serves
# tests #1, #2, #5.
@pytest.fixture(scope="module")
def consuming_run() -> dict:
    return _run_coupled(cpa=2.0e10, gas_flow_Lpm=2.0, n_steps=200)


# kLa sweep with strong consumption (denser culture) so the steady-state DO is
# clearly kLa-limited; lower gas flow -> lower kLa -> lower DO. Serves test #3.
@pytest.fixture(scope="module")
def kla_low() -> dict:
    return _run_coupled(cpa=1.0e11, gas_flow_Lpm=0.5, n_steps=200)


@pytest.fixture(scope="module")
def kla_high() -> dict:
    return _run_coupled(cpa=1.0e11, gas_flow_Lpm=5.0, n_steps=200)


# --- PART 1: time-base adapter (fast unit/integration check) ---------------

@pytest.fixture(scope="module")
def core():
    return build_core()


def test_time_base_adapter_do_rises_to_saturation_and_stays_bounded(core):
    """PART 1 physical-sanity check for the seconds->hours adapter.

    With a realistic kLa, dissolved O2 seeded BELOW saturation and NO cell
    consumption, BiRDTransportHours must drive DO monotonically up toward
    ``o2_saturation`` and keep it in [0, ~saturation] over many per-second steps
    (no 3600x over-transport, no divergence, no negative). Run over 2 simulated
    hours (7200 one-second steps).
    """
    cfg = {
        "reactor_type": "bubble_column", "volume_L": 1.0,
        "gas_flow_rate_Lpm": 2.0, "temperature_K": 310.15,
    }
    proc = BiRDTransportHours(config=cfg, core=core)

    do = 2.0  # seeded well below saturation
    cstar = None
    inputs = {"dissolved_co2": 0.5, "biomass": 0.0, "glucose": 0.0,
              "gas_flow_rate_Lpm": 2.0}
    prev = do
    monotonic_up = True
    for _ in range(7200):  # 7200 s = 2 h, interval = 1 s each
        out = proc.update({"dissolved_o2": do, **inputs}, 1.0)
        do += out["dissolved_o2"]          # additive transport delta
        cstar = out["o2_saturation"]
        assert math.isfinite(do), f"DO diverged to {do!r}"
        assert -1e-9 <= do <= cstar + 1e-6, f"DO out of [0, cstar]: {do}"
        if do < prev - 1e-9:
            monotonic_up = False
        prev = do

    assert monotonic_up, "DO did not rise monotonically toward saturation"
    # After 2 h at kLa ~20.8/h the store is at the gas-liquid equilibrium.
    assert do == pytest.approx(cstar, abs=1e-3), (
        f"DO {do} did not approach saturation {cstar}")


def test_time_base_adapter_converts_seconds_to_hours(core):
    """The adapter delegates the SAME physics as the raw process, but with the
    interval divided by 3600 — a 1-second adapter step == a (1/3600)-hour raw
    step. Direct equivalence check (no divergence claim)."""
    from pbg_bioreactordesign import BiRDTransportProcess
    cfg = {"reactor_type": "bubble_column", "volume_L": 1.0,
           "gas_flow_rate_Lpm": 2.0, "temperature_K": 310.15}
    adapter = BiRDTransportHours(config=cfg, core=core)
    raw = BiRDTransportProcess(config=cfg, core=core)
    state = {"dissolved_o2": 5.0, "dissolved_co2": 0.5, "biomass": 0.0,
             "glucose": 0.0, "gas_flow_rate_Lpm": 2.0}
    a = adapter.update(dict(state), 1.0)               # 1 second
    b = raw.update(dict(state), 1.0 / SECONDS_PER_HOUR)  # 1/3600 hour
    assert a["dissolved_o2"] == pytest.approx(b["dissolved_o2"], rel=1e-12)


# --- PART 2: the 5 mbp-03 behavior tests -----------------------------------

def test_one_generation_completes_without_divergence(consuming_run):
    """behavior_test ``one-generation-completes-without-divergence`` (PRIMARY).

    A representative coupled run completes without state divergence: no NaN,
    dissolved O2 stays in [0, o2_saturation] every step, and the population has
    at least one cell.
    """
    run = consuming_run
    cstar = run["cstar_o2"]
    assert math.isfinite(run["do_final"])
    assert all(math.isfinite(v) for v in run["do_series"]), "NaN/inf in DO series"
    assert all(0.0 <= v <= cstar + 1e-6 for v in run["do_series"]), (
        f"DO left [0, {cstar}]: min={min(run['do_series'])}, "
        f"max={max(run['do_series'])}")
    assert run["cell_count"] >= 1.0, f"cell_count={run['cell_count']}"


def test_cells_drop_do_below_saturation(consuming_run):
    """behavior_test ``cells-drop-do-below-saturation`` (PRIMARY).

    With v2ecoli cells consuming O2, the steady-state (second-half-window) mean
    dissolved O2 sits below the gas-liquid (Henry) saturation.
    """
    run = consuming_run
    half = run["do_series"][len(run["do_series"]) // 2:]
    mean_do = statistics.mean(half)
    assert mean_do < run["cstar_o2"], (
        f"second-half mean DO {mean_do:.4f} not below saturation "
        f"{run['cstar_o2']:.4f}")


def test_higher_kla_raises_steady_state_do(kla_low, kla_high):
    """behavior_test ``higher-kla-raises-steady-state-do`` (PRIMARY).

    Across the gas-flow sweep (which drives kLa via BiRD's Higbie correlation),
    the steady-state (second-half-window) mean dissolved O2 increases with kLa:
    less O2-limited at higher transfer.
    """
    assert kla_high["kla_o2"] > kla_low["kla_o2"], "kLa did not increase with gas flow"

    def mean_2h(run):
        s = run["do_series"]
        return statistics.mean(s[len(s) // 2:])

    do_low, do_high = mean_2h(kla_low), mean_2h(kla_high)
    assert do_high > do_low, (
        f"higher kLa ({kla_high['kla_o2']:.2f}) gave DO {do_high:.4f} "
        f"not above lower kLa ({kla_low['kla_o2']:.2f}) DO {do_low:.4f}")


@pytest.mark.skip(reason=(
    "PHYSICAL/MODEL limitation: in this M9-minimal-glucose FBA configuration "
    "v2ecoli reports ZERO CO2 environmental exchange "
    "(agents.*.environment.exchange['CARBON-DIOXIDE'] == 0.0 every step) — the "
    "model balances CO2 internally and does not secrete it to the environment "
    "store, so cells cannot raise dissolved CO2 above saturation. The CO2 leg "
    "of the coupler is wired (verified by code) but cannot be exercised until a "
    "media/condition that yields nonzero CO2 efflux (e.g. acetate overflow / "
    "anaerobic) is used. Not loosened to a fake pass."))
def test_cells_raise_dissolved_co2_above_saturation():
    """behavior_test ``cells-raise-dissolved-co2-above-saturation`` (PRIMARY)."""
    raise NotImplementedError


def test_o2_mass_balance_closes(consuming_run):
    """behavior_test ``o2-mass-balance-closes`` (PRIMARY).

    Cumulative O2 consumed by the cells (computed INDEPENDENTLY from the source
    store agents.*.environment.exchange, counts -> mg/L) must close against the
    reactor's gas-liquid balance: ``consumed == transferred - delta_dissolved``
    (mass conservation of the additive dissolved-O2 store). The ratio
    ``consumed / (transferred - delta_dissolved)`` must sit in [0.95, 1.05].

    ``consumed`` is reconstructed from the cell-side exchange store via
    Avogadro's number, while ``transferred`` and ``delta_dissolved`` come from
    the reactor-side transport diagnostic and the dissolved store — three
    quantities. It validates the coupler's counts->mg/L conversion and that no
    O2 mass leaks across the coupling.

    ⚠ It is only independent if BOTH sides read the cumulative store correctly.
    Before 2026-08-29 neither did: the coupler consumed the running total and
    this test summed it, so the ratio stayed ~1 at any magnitude and the
    criterion could not fail. It is a genuine check now; it was not then.
    """
    run = consuming_run
    cpa = run["cells_per_agent"]
    vol = run["volume_L"]
    counts_to_mgL = cpa / AVOGADRO * 1000.0 / vol * MW_O2

    # Source-side cumulative consumption (positive mg/L).
    #
    # ⚠ `environment.exchange` is a LINEAGE-CUMULATIVE running total, not a
    # per-step delta, so SUMMING the sampled series double-counts: it sums a
    # series that should have been differenced, inflating by ~N/2 over N steps.
    # Until 2026-08-29 this test did exactly that -- and PASSED, because the
    # coupler made the same error, so both sides of the ratio were inflated by
    # the same factor and the axis went green. That is the second catalogued
    # instance of this error class (see scripts/_compare/exchange_flux_basis.py
    # lines 45-51 for the first, where it also cancelled between two arms).
    #
    # The total consumed IS the store's total change, so difference the
    # endpoints rather than summing the series.
    consumed = -(run["o2_counts"][-1] - run["o2_counts"][0]) * counts_to_mgL
    # Reactor-side gas-liquid transfer (cumulative mg/L) + dissolved change.
    transferred = sum(run["o2_transport_deltas"])
    delta_dissolved = run["do_final"] - run["do0"]
    denom = transferred - delta_dissolved

    assert abs(consumed) > 1e-6, "no measurable O2 consumption to balance"
    ratio = consumed / denom
    assert 0.95 <= ratio <= 1.05, (
        f"O2 mass balance did not close: consumed={consumed:.4f} "
        f"transferred={transferred:.4f} delta_dissolved={delta_dissolved:.4f} "
        f"ratio={ratio:.4f}")
