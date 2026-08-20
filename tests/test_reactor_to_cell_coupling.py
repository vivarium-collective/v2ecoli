"""The reactor -> cell direction of the coupling, and the population scale.

Three defects, all on the path from the bioreactor's bulk state back into the
cell. Each is measured against `reactor_bird_coupled`.

1. **Nothing the reactor computed reached the cell.** `ReactorCellCoupler`
   writes `environment.external_concentrations` under compartment-tagged
   v2ecoli ids (`OXYGEN-MOLECULE[p]`), while `boundary.external` is keyed BARE
   (`OXYGEN-MOLECULE`). `EnvironmentMirror` fails closed on any id it cannot
   match -- silently -- so the match rate was ZERO on every tick and the
   reactor's dissolved gases never reached the cell at all. The composite's own
   docstring claims the mirror "propagates the reactor-derived boundary"; it
   did not. `EnvironmentMirror` now resolves a compartment-tagged id onto the
   bare boundary key, preferring an exact match.

2. **Glucose had no bulk->cell path even in principle.** `glucose_medium_mM`
   was written by the coupler and read by NOTHING, so draining the pool changed
   nothing the cells experienced and glucose-limited growth was unreachable
   however the reactor was configured. It is now written into the environment
   on the same footing as the dissolved gases, and the pool is clamped at zero.

3. **`representative_doubling` grew biomass but not consumption.**
   `PopulationAggregator` applies 2**doublings to biomass and cell_count; this
   Step scaled exchange by `cells_per_agent` alone. By generation 4 the reactor
   held 8x the biomass consuming 1x the glucose and oxygen -- mass appearing
   from nowhere, which makes any closure criterion fail by construction. The
   scale is now derived from `population.cell_count`, which already carries the
   factor, so biomass and exchange cannot drift apart.

Tests marked `discriminates:` fail against the pre-fix tree; the others are
regression guards for behaviour that must NOT change.
"""

from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.environment_mirror import EnvironmentMirror, _resolve_boundary_keys
from v2ecoli.steps.reactor_cell_coupler import (
    AVOGADRO,
    GLUCOSE_EXCHANGE_KEY,
    GLUCOSE_ID,
    GLUCOSE_MEDIUM_LEAF,
    O2_ID,
    ReactorCellCoupler,
)


@pytest.fixture(scope="module")
def core():
    return build_core()


# --- 1. the mirror's id resolution -------------------------------------------


def test_compartment_tagged_id_resolves_onto_the_bare_boundary_key():
    """discriminates: pre-fix this returned no mapping, so the coupler's every
    write was dropped."""
    resolved = _resolve_boundary_keys(
        {"OXYGEN-MOLECULE[p]": 0.2}, {"OXYGEN-MOLECULE": 1.0, "GLC": 11.1}
    )
    assert resolved == {"OXYGEN-MOLECULE[p]": "OXYGEN-MOLECULE"}


def test_exact_match_is_preferred_over_stripping():
    """The EnvironmentDriver's bare-name convention must keep working
    unchanged -- this is the path every mbp-01 run depends on."""
    resolved = _resolve_boundary_keys({"GLC": 5.0}, {"GLC": 11.1})
    assert resolved == {"GLC": "GLC"}


def test_ambiguous_ids_are_dropped_rather_than_last_one_winning():
    """Two compartments reducing to one boundary key is ambiguous. Dropping
    both fails closed; letting the last win would be a silent, order-dependent
    wrong answer."""
    resolved = _resolve_boundary_keys({"X[p]": 1.0, "X[c]": 2.0}, {"X": 0.5})
    assert resolved == {}


def test_untracked_molecule_is_still_skipped():
    assert _resolve_boundary_keys({"NOT-A-MOLECULE[p]": 1.0}, {"GLC": 11.1}) == {}


def test_mirror_emits_a_delta_for_a_compartment_tagged_id(core):
    """discriminates: pre-fix the mirror returned {} for exactly this input --
    which is what made the reactor->cell direction inert.
    """
    mirror = EnvironmentMirror(config={}, core=core)
    out = mirror.next_update(
        1.0,
        {
            "environment": {"external_concentrations": {"GLC[p]": 22.2}},
            "agents": {"0": {"boundary": {"external": {"GLC": 11.1}}}},
        },
    )
    assert out["agents"]["0"]["boundary"]["external"]["GLC"] == pytest.approx(22.2 - 11.1)


# --- 2. glucose reaches the environment --------------------------------------


def _coupler_out(core, *, reactor, agents=None, population=None, config=None):
    # track_medium must be passed EXPLICITLY: config_schema declares it a
    # boolean, so the framework fills it False on omission and the Step's
    # "default ON" branch never fires. See the note in initialize().
    cfg = {"track_medium": True}
    cfg.update(config or {})
    c = ReactorCellCoupler(config=cfg, core=core)
    states = {"reactor": reactor}
    if agents is not None:
        states["agents"] = agents
    if population is not None:
        states["population"] = population
    return c.next_update(1.0, states)


def test_medium_glucose_is_published_to_the_environment(core):
    """discriminates: pre-fix `glucose_medium_mM` was written to the reactor
    store and read by nothing; no GLC key appeared here at all."""
    out = _coupler_out(
        core, reactor={GLUCOSE_MEDIUM_LEAF: 22.2, "volume_L": 1.0}
    )
    assert out["environment"]["external_concentrations"][GLUCOSE_ID] == pytest.approx(22.2)


def test_an_exhausted_pool_reads_as_zero_not_negative(core):
    """A negative concentration is not a physical state, and metabolism's
    import threshold would treat it as merely 'below threshold' rather than
    impossible."""
    out = _coupler_out(
        core, reactor={GLUCOSE_MEDIUM_LEAF: -3.0, "volume_L": 1.0}
    )
    assert out["environment"]["external_concentrations"][GLUCOSE_ID] == 0.0


def test_glucose_draw_is_clamped_at_the_remaining_pool(core):
    """discriminates: pre-fix the medium leaf had no clamp (unlike the
    dissolved gases, which have one), so the pool could be driven negative."""
    huge_uptake = -1.0e24  # counts/step, far more than the pool holds
    out = _coupler_out(
        core,
        reactor={GLUCOSE_MEDIUM_LEAF: 5.0, "volume_L": 1.0},
        agents={"0": {"environment": {"exchange": {GLUCOSE_EXCHANGE_KEY: huge_uptake}}}},
        population={"cell_count": 1.0},
    )
    assert out["reactor"][GLUCOSE_MEDIUM_LEAF] == pytest.approx(-5.0)


# --- 3. the population scale -------------------------------------------------


def _o2_delta(core, *, cell_count, n_agents=1, counts=-1.0e18):
    agents = {
        str(i): {"environment": {"exchange": {"OXYGEN-MOLECULE": counts}}}
        for i in range(n_agents)
    }
    out = _coupler_out(
        core,
        reactor={"dissolved_o2": 1.0e9, "dissolved_co2": 0.0, "volume_L": 1.0},
        agents=agents,
        population={"cell_count": cell_count},
        config={"cells_per_agent": 1.0, "reactor_volume_L": 1.0},
    )
    return out["reactor"]["dissolved_o2"]


def test_exchange_scales_with_the_represented_population(core):
    """discriminates: pre-fix both calls returned the SAME delta, because the
    scale was `cells_per_agent` alone and ignored the growth factor entirely.
    Eight generations of biomass ate one generation's worth of oxygen.
    """
    one_generation = _o2_delta(core, cell_count=1.0)
    eight_fold = _o2_delta(core, cell_count=8.0)
    assert eight_fold == pytest.approx(8.0 * one_generation, rel=1e-9)


def test_scale_falls_back_to_cells_per_agent_without_a_population_store(core):
    """The aggregator is optional; a composite that does not wire it must
    behave exactly as before."""
    agents = {"0": {"environment": {"exchange": {"OXYGEN-MOLECULE": -1.0e18}}}}
    out = _coupler_out(
        core,
        reactor={"dissolved_o2": 1.0e9, "dissolved_co2": 0.0, "volume_L": 1.0},
        agents=agents,
        config={"cells_per_agent": 4.0, "reactor_volume_L": 1.0},
    )
    expected = -1.0e18 * 4.0 / AVOGADRO * 1000.0 / 1.0 * 31.999
    assert out["reactor"]["dissolved_o2"] == pytest.approx(expected, rel=1e-9)


def test_scale_is_per_agent_not_per_population(core):
    """cell_count covers ALL agents, so the per-agent scale must divide by the
    number of agents -- otherwise two agents would each be scaled by the whole
    population and consumption would double-count."""
    one_agent = _o2_delta(core, cell_count=8.0, n_agents=1, counts=-1.0e18)
    two_agents = _o2_delta(core, cell_count=8.0, n_agents=2, counts=-1.0e18)
    assert two_agents == pytest.approx(one_agent, rel=1e-9)


# --- the wiring, in a running composite --------------------------------------


@pytest.mark.sim
def test_reactor_medium_reaches_the_cell_boundary(core):
    """End to end: the cell's boundary glucose must come from the REACTOR pool,
    not from the media recipe it was seeded with.

    discriminates: pre-fix the mirror matched none of the coupler's ids, the
    boundary kept its seeded recipe value (11.101 mM), and the reactor's pool
    (22.2 mM) was invisible to the cell.

    NOTE this test asserts only on glucose. Oxygen's boundary is seeded at
    `inf` (unlimited), and a DELTA cannot move an unlimited value, so the
    dissolved-O2 story is NOT closed by this change -- see the guard in
    EnvironmentMirror. Asserting on O2 here would encode a fix that does not
    exist.
    """
    from v2ecoli import build_composite

    comp = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")

    def boundary_glc():
        value = comp.state["agents"]["0"]["boundary"]["external"]["GLC"]
        return float(getattr(value, "magnitude", value))

    seeded = boundary_glc()
    # The coupler runs at the END of the flow, so its first write is visible to
    # the mirror on the following tick.
    comp.update({}, 1.0)
    comp.update({}, 1.0)

    pool = comp.state["reactor"][GLUCOSE_MEDIUM_LEAF]
    pool = float(getattr(pool, "magnitude", pool))
    assert boundary_glc() == pytest.approx(pool, rel=1e-6), (
        f"boundary glucose {boundary_glc()} does not track the reactor pool {pool} "
        f"(seeded at {seeded})"
    )


def test_an_unresolvable_id_is_counted_and_named_not_merely_dropped(core):
    """A silent drop here is exactly how the reactor's whole gas channel went
    missing with nothing failing. Failing closed is right; failing closed
    INVISIBLY is what cost the time.

    discriminates: pre-fix the Step kept no record and emitted no warning, so
    an unmatched id was indistinguishable from having nothing to say.
    """
    mirror = EnvironmentMirror(config={}, core=core)
    states = {
        "environment": {"external_concentrations": {"NOT-A-MOLECULE[p]": 1.0}},
        "agents": {"0": {"boundary": {"external": {"GLC": 11.1}}}},
    }
    with pytest.warns(RuntimeWarning, match="no boundary.external key"):
        assert mirror.next_update(1.0, states) == {}
    # Counted every tick, but named only once -- a per-tick warning over a
    # 7200-step run would be its own kind of unreadable.
    mirror.next_update(1.0, states)
    assert mirror.skipped_unmatched == {"NOT-A-MOLECULE[p]": 2}


def test_a_resolvable_id_is_not_counted_as_skipped(core):
    """Guard against the tally quietly counting healthy traffic."""
    mirror = EnvironmentMirror(config={}, core=core)
    mirror.next_update(
        1.0,
        {
            "environment": {"external_concentrations": {"GLC[p]": 22.2}},
            "agents": {"0": {"boundary": {"external": {"GLC": 11.1}}}},
        },
    )
    assert mirror.skipped_unmatched == {}


# --- the Millard arm must not be driven by the reactor's glucose -------------


def test_reactor_glucose_does_not_reach_the_millard_cell(core):
    """The coupler publishes glucose into the SHARED top-level environment
    store, and `ReactorMillardEnvBridge` reads that same store.

    `GLC[p]` is already aliased to the SBML species `GLCx`, so without an
    explicit guard the reactor's pool (~22.2 mM) overwrites the Millard model's
    calibrated external glucose (0.00633 mM) every tick -- a ~3500x step change
    in the driver of its glucose rate law, which also overrides GLCx's own
    dynamics and its _GLC_FEED chemostat.

    discriminates: without the guard this test fails, and so does
    `test_millard_o2_limitation.py::test_coupled_reactor_o2_feedback_closes_loop`
    (CYTBO stops throttling at low DO) -- which is the behaviour that caught it.

    This pins a DEFERRAL, not a judgement: whether a coupled reactor should set
    the Millard cell's external glucose is a live modelling question, and
    answering it means re-calibrating that model and regenerating mbp-07's
    committed figures and the millard_vs_beulig report card.
    """
    from v2ecoli.steps.reactor_millard_env_bridge import ReactorMillardEnvBridge

    bridge = ReactorMillardEnvBridge(config={}, core=core)
    out = bridge.next_update(
        1.0,
        {
            "environment": {
                "external_concentrations": {
                    "GLC[p]": 22.2,
                    "OXYGEN-MOLECULE[p]": 0.25,
                }
            },
            "agents": {"0": {}},
        },
    )
    passed = out["agents"]["0"]["environment"]["external_concentrations"]
    assert "GLC[p]" not in passed, (
        "the reactor's medium glucose reached the Millard cell and would "
        "overwrite its calibrated GLCx"
    )
    # The gases must still get through -- that is the whole point of the bridge.
    assert passed["OXYGEN-MOLECULE[p]"] == pytest.approx(0.25)


@pytest.mark.parametrize("spelling", ["GLC", "GLC[p]", "GLC[c]"])
def test_the_glucose_guard_is_keyed_on_the_species_not_the_spelling(core, spelling):
    """`EXTERNAL_NAME_TO_SBML` resolves GLC, GLC[p] and GLC[c] all to `GLCx`.
    A guard keyed on one spelling goes silently inert the moment a producer uses
    another -- including the plausible next step of the coupler emitting bare
    ids to match the boundary convention.

    discriminates: keyed on the literal "GLC[p]", the bare and [c] spellings
    both reach GLCx.
    """
    from v2ecoli.steps.reactor_millard_env_bridge import ReactorMillardEnvBridge

    bridge = ReactorMillardEnvBridge(config={}, core=core)
    out = bridge.next_update(
        1.0,
        {
            "environment": {"external_concentrations": {spelling: 22.2}},
            "agents": {"0": {}},
        },
    )
    passed = (out.get("agents", {}).get("0", {})
                 .get("environment", {}).get("external_concentrations", {}))
    assert spelling not in passed, f"{spelling} resolved to GLCx and got through"


def test_the_bridge_still_passes_what_it_should(core):
    """Guard against the exclusion quietly widening: O2 must still arrive."""
    from v2ecoli.steps.reactor_millard_env_bridge import ReactorMillardEnvBridge

    bridge = ReactorMillardEnvBridge(config={}, core=core)
    out = bridge.next_update(
        1.0,
        {
            "environment": {"external_concentrations": {"OXYGEN-MOLECULE[p]": 0.25}},
            "agents": {"0": {}},
        },
    )
    passed = out["agents"]["0"]["environment"]["external_concentrations"]
    assert passed == {"OXYGEN-MOLECULE[p]": pytest.approx(0.25)}


def test_the_unmatched_tally_counts_agent_ticks(core):
    """The unit is agent-ticks, not ticks. Pinned because the obvious reading
    of the counter is wrong for any multi-agent run."""
    mirror = EnvironmentMirror(config={}, core=core)
    states = {
        "environment": {"external_concentrations": {"NOPE[p]": 1.0}},
        "agents": {
            str(i): {"boundary": {"external": {"GLC": 11.1}}} for i in range(3)
        },
    }
    with pytest.warns(RuntimeWarning):
        mirror.next_update(1.0, states)
    assert mirror.skipped_unmatched == {"NOPE[p]": 3}


def test_the_scale_fallback_records_that_it_fired(core):
    """The fallback silently switches between two different population scales.
    In `fixed` mode the two agree, so nothing shows; under
    `representative_doubling` the fallback under-scales that tick's exchange.
    It must leave a trace either way.

    discriminates: before this, the fallback kept no record at all.
    """
    c = ReactorCellCoupler(config={"cells_per_agent": 4.0, "track_medium": True}, core=core)
    assert c.scale_fallbacks == 0
    agents = {"0": {"environment": {"exchange": {"OXYGEN-MOLECULE": -1.0e18}}}}
    reactor = {"dissolved_o2": 1.0e9, "dissolved_co2": 0.0, "volume_L": 1.0}
    # No population store at all -> fallback.
    c.next_update(1.0, {"reactor": reactor, "agents": agents})
    assert c.scale_fallbacks == 1
    # A readable cell_count -> derived scale, no further fallback.
    c.next_update(1.0, {"reactor": reactor, "agents": agents,
                        "population": {"cell_count": 8.0}})
    assert c.scale_fallbacks == 1
