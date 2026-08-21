"""Millard O2-limited respiration (#225 item #4).

Closes the reactor->Millard dissolved-O2 feedback loop: the Millard kinetic
cell's O2 respiration (CYTBO) must THROTTLE when the reactor's dissolved O2
drops, instead of treating O2 as a fixed/unlimited species.

Mechanism: O2 is a ``fixed`` species in the Millard SBML, but its CYTBO rate law
``Vmax/(1+(log(Hout/Hin)/log(10))^2)*(QH2^2*O2 - Q^2/Keq)`` reads [O2] linearly.
Overwriting the fixed-species value each tick (from the reactor's dissolved O2,
mg/L -> mM) therefore drives the respiration flux directly — lower DO -> lower
CYTBO. The ReactorCellCoupler writes the reactor O2 as ``OXYGEN-MOLECULE[p]``
(mM) into environment.external_concentrations; the Millard step now aliases that
v2ecoli name to the SBML id ``O2`` so the overwrite reaches the rate law.
"""
import math
import warnings

warnings.filterwarnings("ignore")
import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism


@pytest.fixture(scope="module")
def core():
    return build_core()


def _cytbo_at(core, o2_key, o2_mM):
    """One Millard tick at the given external O2 concentration; return CYTBO flux."""
    s = MillardPDMPMetabolism(config={}, core=core)
    out = s.update(
        {
            "lqr_control": {},
            "bulk": None,
            "listeners_mass": {"cell_mass": 1000.0, "dry_mass": 300.0},
            "external_concentrations": {o2_key: o2_mM} if o2_key else {},
        },
        1.0,
    )
    return float(out["central_fluxes"]["CYTBO"])


@pytest.mark.sim
@pytest.mark.parametrize(
    "o2_key", ["O2", "OXYGEN-MOLECULE", "OXYGEN-MOLECULE[p]"]
)
def test_cytbo_throttles_at_low_o2(core, o2_key):
    """CYTBO respiration is LOWER at low dissolved O2 than at air-saturated O2,
    for the raw SBML id AND the v2ecoli coupler keys (alias mapping)."""
    high = _cytbo_at(core, o2_key, 0.21)   # ~air-saturated (model default)
    low = _cytbo_at(core, o2_key, 0.02)    # depleted dissolved O2
    assert high > 0.0, f"baseline CYTBO not positive: {high}"
    assert low < high, (
        f"CYTBO not throttled at low O2 via key {o2_key!r}: "
        f"high(0.21mM)={high:.6f}, low(0.02mM)={low:.6f}"
    )
    assert low >= 0.0, f"CYTBO went negative (unstable) at low O2: {low}"


@pytest.mark.sim
@pytest.mark.parametrize("o2_key", ["O2", "OXYGEN-MOLECULE[p]"])
def test_cytbo_monotonic_in_o2(core, o2_key):
    """CYTBO is monotonically increasing in dissolved O2 (no instability)."""
    sweep = [0.21, 0.10, 0.05, 0.02, 0.005]
    vals = [_cytbo_at(core, o2_key, c) for c in sweep]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), (
        f"CYTBO not monotonic decreasing as O2 drops ({o2_key}): "
        f"{list(zip(sweep, vals))}"
    )
    assert all(math.isfinite(v) for v in vals), vals


@pytest.mark.sim
def test_uncoupled_uses_calibrated_default(core):
    """With no external O2 (standalone baseline_millard), CYTBO runs at the
    model's calibrated air-saturated value — NOT throttled."""
    base = _cytbo_at(core, None, None)
    assert base > 0.5, f"uncoupled CYTBO unexpectedly low: {base}"


def _f(x) -> float:
    return float(x.magnitude) if hasattr(x, "magnitude") else float(x)


@pytest.mark.sim
# NOT marked `slow`: CI selects `not slow and not sim` for fast-tests and
# `sim and not slow` for behavior-tests, so `sim` + `slow` together means a test
# runs in NEITHER job. This one measured 16.6 s locally against a ~9 min
# behavior-tests budget, and while it was invisible a change to the reactor
# coupler broke it with every gate green.
def test_coupled_reactor_o2_feedback_closes_loop():
    """Coupled-run sanity: as biomass draws the reactor's dissolved O2 down, the
    bridge feeds that DO into the Millard cell and CYTBO respiration EASES
    (negative feedback) — and the dissolved O2 stays bounded (>=0, finite)
    instead of crashing unbounded/negative.

    Strong-overload regime (high cells_per_agent, low gas flow) so DO falls
    enough that the throttle is unambiguous within a short run.
    """
    import math

    from v2ecoli import build_composite

    c = build_composite(
        "reactor_bird_coupled_millard", seed=0, cache_dir="out/cache",
        cells_per_agent=5.0e12,
        bird_reactor_config={"gas_flow_rate_Lpm": 0.05},
    )

    rows = []  # (DO mg/L, agent O2 ext mM, CYTBO flux)
    for _ in range(12):
        c.run(1)
        ag = c.state["agents"]["0"]
        do = _f(c.state["reactor"]["dissolved_o2"])
        o2e = (ag.get("environment", {})
               .get("external_concentrations", {})
               .get("OXYGEN-MOLECULE[p]"))
        cyt = ag.get("central_fluxes", {}).get("CYTBO")
        rows.append((do, o2e, cyt))

    dos = [r[0] for r in rows]
    o2ext = [r[1] for r in rows]
    cyts = [r[2] for r in rows]

    # 1. Dissolved O2 stays bounded (finite, >=0) — no unbounded crash/negative.
    assert all(math.isfinite(d) and d >= 0.0 for d in dos), dos

    # 2. The bridge actually wired the reactor O2 into the Millard cell's
    #    environment.external_concentrations (was {} before this fix).
    assert all(v is not None for v in o2ext), (
        f"agent O2 boundary not populated by bridge: {o2ext}")
    assert _f(o2ext[-1]) < _f(o2ext[0]), (
        f"agent O2 boundary did not fall with DO: {o2ext[0]} -> {o2ext[-1]}")

    # 3. Negative feedback: CYTBO respiration is LOWER when DO is low than when
    #    DO is high (throttle), and the dynamic O2-driven CYTBO drops overall.
    i_hi = dos.index(max(dos))
    i_lo = dos.index(min(dos))
    assert cyts[i_lo] < cyts[i_hi], (
        f"CYTBO not throttled at low DO: "
        f"CYTBO@maxDO({dos[i_hi]:.2f})={cyts[i_hi]:.4f}, "
        f"CYTBO@minDO({dos[i_lo]:.2f})={cyts[i_lo]:.4f}")
