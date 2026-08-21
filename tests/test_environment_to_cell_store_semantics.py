"""The bulk -> cell direction: does a change in the environment actually reach
metabolism's uptake bounds?

Two defects sat on this path, both in how an update is APPLIED rather than in
how it is computed. `ExchangeData` produced the right answer at every tick and
the store discarded it:

* ``environment.exchange_data.constrained`` was a bare ``map[float]``. A map
  apply walks the UPDATE's keys, so a key the producer stops emitting is never
  removed. When a carbon source falls below the import threshold `ExchangeData`
  correctly drops it from `constrained` (that is how "max flux 0" is expressed
  -- see `Metabolism.exchange_constraints`, where "in neither collection" means
  zero available), but the store kept the stale 20.0 mmol/gDCW/h cap forever.
  The cell went on consuming a substrate that was gone.
* ``unconstrained`` was a bare ``list[string]``, and a list apply CONCATENATES
  (``state + update``). It grew by one full copy of the molecule set per tick --
  never shrinking, and an unbounded leak over a long run.

Both are fixed by declaring the two stores with replace semantics, using the
``overwrite[...]`` wrapper this file's sibling declarations already use for the
listener copies of these same two quantities (`metabolism.py`, `fba_results`).

A third defect lives at the step before: `EnvironmentMirror` writes a DELTA, and
metabolism seeds unlimited molecules (canonically O2) at ``inf``. ``0.0 - inf``
is ``-inf``, which passes an isnan check and then accumulates to NaN in the
additive boundary store, silently poisoning that molecule. The guard now fails
closed on any non-finite delta.

Two kinds of test live here, and the difference is worth stating because it is
easy to mistake the second kind for coverage it does not provide:

* **Discriminating** -- fails against the pre-fix tree. Verified by rebinding
  the two source files to their pre-fix state and re-running: **five** fail --
  the declaration check, the two inf-guard cases, and both sim tests -- and the
  remaining seven pass.
* **Property anchors** -- pass both before and after, because they pin what
  ``overwrite[...]`` and the bare types MEAN. They do not prove the fix is
  wired up; they catch a revert of the declaration (via the source check) or a
  change in the wrapper's semantics underneath us.

Neither kind proves the store is reached in a running composite, which is the
claim that actually matters. ``test_below_threshold_carbon_source_stops_uptake``
at the bottom of this file is the one that does, and it is the reason the file
carries a sim-marked test at all.
"""

from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.environment_mirror import EnvironmentMirror


@pytest.fixture(scope="module")
def core():
    return build_core()


# --- store semantics ---------------------------------------------------------
#
# These pin the PROPERTY the fix depends on, not the spelling of the type name:
# they would fail if the declaration were reverted, and equally if the wrapper
# stopped meaning "replace".

def test_overwrite_map_drops_keys_the_producer_stopped_emitting(core):
    """A carbon source that falls below threshold leaves `constrained` entirely.

    Property anchor (passes pre-fix too): contrasts the wrapper against the
    bare ``map[float]``, which returns {'GLC[p]': 20.0} for both of these --
    the key is unreachable once seeded.
    """
    seeded = {"GLC[p]": 20.0}
    # Producer now emits nothing: the substrate is exhausted.
    assert core.apply("overwrite[map[float]]", dict(seeded), {})[0] == {}
    # Producer emits a different molecule: GLC must not survive.
    assert core.apply("overwrite[map[float]]", dict(seeded), {"ACET[p]": 5.0})[0] == {
        "ACET[p]": 5.0
    }


def test_bare_map_is_key_frozen(core):
    """Documents WHY the wrapper is required, so a future reader can see the
    difference rather than taking the docstring's word for it."""
    assert core.apply("map[float]", {"GLC[p]": 20.0}, {})[0] == {"GLC[p]": 20.0}


def test_overwrite_list_replaces_instead_of_concatenating(core):
    """`unconstrained` must reflect this tick's importer set, not every tick's.

    Property anchor (passes pre-fix too): the bare ``list[string]`` returns
    the concatenation, which is what grew the store by 18 entries per tick.
    """
    assert core.apply("overwrite[list[string]]", ["GLC[p]"], ["O2[p]"])[0] == ["O2[p]"]


def test_bare_list_concatenates(core):
    assert core.apply("list[string]", ["GLC[p]"], ["O2[p]"])[0] == ["GLC[p]", "O2[p]"]


def test_metabolism_declares_replace_semantics_for_exchange_data():
    """The declaration itself, read as DATA rather than as source text -- a
    source-text assertion would break on a reformat and would still pass if the
    declaration were disconnected from the store.

    This is the ONLY discriminator for the store half that runs in the fast
    gate; the two that exercise the real store are sim-marked and run in the
    behavior-tests job. If this is ever deleted, a silent revert of the schema
    change reaches main green.

    discriminates: fails against the pre-fix declaration.
    """
    from v2ecoli.processes.metabolism import Metabolism

    # inputs() closes over no instance state, so it is readable off the class.
    exchange_data = Metabolism.inputs(None)["environment"]["exchange_data"]
    assert exchange_data["constrained"] == "overwrite[map[float]]"
    assert exchange_data["unconstrained"] == "overwrite[list[string]]"


# --- EnvironmentMirror non-finite guard --------------------------------------


def _mirror_update(core, *, driver_conc, boundary_conc):
    mirror = EnvironmentMirror(config={}, core=core)
    states = {
        "environment": {"external_concentrations": {"GLC": driver_conc}},
        "agents": {"0": {"boundary": {"external": {"GLC": boundary_conc}}}},
    }
    out = mirror.next_update(1.0, states)
    if not out:
        return None
    return out["agents"]["0"]["boundary"]["external"].get("GLC")


def test_inf_boundary_yields_no_delta_rather_than_negative_infinity(core):
    """An unlimited (inf) boundary cannot be moved by a delta -- fail closed.

    discriminates: the old isnan guard lets ``0.0 - inf = -inf`` straight
    through, and ``inf + -inf`` accumulates to NaN in the additive boundary
    store, poisoning the molecule for the rest of the run.
    """
    assert _mirror_update(core, driver_conc=0.0, boundary_conc=float("inf")) == 0.0


def test_partial_reduction_of_an_inf_molecule_is_also_refused(core):
    """Not just the zero case: ANY driver write against an inf boundary.

    discriminates: old guard emits -inf here too.
    """
    assert _mirror_update(core, driver_conc=5.0, boundary_conc=float("inf")) == 0.0


def test_nan_driver_value_still_guarded(core):
    """Regression guard for the behaviour the original isnan check provided."""
    assert _mirror_update(core, driver_conc=float("nan"), boundary_conc=1.0) == 0.0


def test_ordinary_finite_delta_is_unchanged(core):
    """The guard must not disturb the normal path -- this is the case every
    existing time-varying-environment run depends on."""
    assert _mirror_update(core, driver_conc=5.0, boundary_conc=11.101) == pytest.approx(
        5.0 - 11.101
    )


def test_unmatched_molecule_is_still_skipped(core):
    """Bare-name convention: a molecule metabolism doesn't track is skipped,
    not crashed on."""
    mirror = EnvironmentMirror(config={}, core=core)
    out = mirror.next_update(
        1.0,
        {
            "environment": {"external_concentrations": {"NOT-A-MOLECULE": 1.0}},
            "agents": {"0": {"boundary": {"external": {"GLC": 1.0}}}},
        },
    )
    assert out == {}



# --- the wiring, in a running composite --------------------------------------


@pytest.mark.sim
def test_below_threshold_carbon_source_stops_uptake(core):
    """End to end: exhaust glucose at the boundary and the cell must stop eating.

    This is the only test in this file that exercises the real store. The unit
    tests above would all still pass if the seam moved and the update stopped
    reaching metabolism at all -- which is precisely the failure being fixed, so
    covering it only at the type level would repeat the original mistake.

    Protocol: run with glucose present, clamp ``boundary.external.GLC`` to zero,
    run on. ``environment.exchange`` is a cumulative counts store, so "uptake
    stopped" means it stops CHANGING.

    discriminates: against the pre-fix tree the store keeps its stale 20.0
    mmol/gDCW/h cap, and the exchange total keeps falling by ~2.9e5 counts per
    tick -- the cell consuming a substrate that is gone.
    """
    from process_bigraph import Composite

    from v2ecoli.composites.ecoli_baseline import baseline

    doc = baseline(core, seed=0, cache_dir="out/cache", emitter="null", max_duration=10.0)
    comp = Composite(doc, core=core)

    def agent():
        return comp.state["agents"]["0"]

    def glc_exchange_total():
        exch = agent().get("environment", {}).get("exchange") or {}
        key = next((k for k in exch if str(k).startswith("GLC")), None)
        val = exch.get(key) if key else None
        return float(getattr(val, "magnitude", val)) if val is not None else None

    def constrained():
        return agent()["environment"]["exchange_data"].get("constrained") or {}

    # --- glucose present: the cap is set and uptake accumulates.
    for _ in range(3):
        comp.update({}, 1.0)
    assert "GLC[p]" in constrained(), "expected an aerobic glucose cap while glucose is present"
    fed_total = glc_exchange_total()
    assert fed_total is not None and fed_total < 0.0, "expected net glucose uptake (negative counts)"

    # --- exhaust it. Bare molecule names in boundary.external, per the mirror's
    #     documented convention.
    external = agent()["boundary"]["external"]
    assert "GLC" in external, f"expected a bare GLC key; saw {sorted(external)[:8]}"
    current = external["GLC"]
    external["GLC"] = (0.0 * current) if hasattr(current, "magnitude") else 0.0

    comp.update({}, 1.0)
    starved_total = glc_exchange_total()

    # The producer drops the key; the store must drop it too.
    assert "GLC[p]" not in constrained(), (
        "the glucose cap survived exhaustion -- the store is key-frozen and the "
        "cell is still permitted to import a substrate that is gone"
    )

    # And metabolism must actually stop importing.
    for _ in range(2):
        comp.update({}, 1.0)
    assert glc_exchange_total() == pytest.approx(starved_total, rel=0, abs=0), (
        "glucose uptake continued after the substrate was exhausted"
    )


@pytest.mark.sim
def test_unconstrained_importer_set_does_not_grow_without_bound(core):
    """`unconstrained` must describe THIS tick's importers.

    discriminates: against the pre-fix tree the list concatenates. Measured
    series: 18 seeded, 54 after the first tick (the initialization pass runs
    exchange_data twice), then +18 per tick -- 108 by tick 4, 126 by tick 5,
    576 by tick 30 -- leaking for the whole run.
    """
    from process_bigraph import Composite

    from v2ecoli.composites.ecoli_baseline import baseline

    doc = baseline(core, seed=0, cache_dir="out/cache", emitter="null", max_duration=8.0)
    comp = Composite(doc, core=core)

    def n_unconstrained():
        return len(comp.state["agents"]["0"]["environment"]["exchange_data"]["unconstrained"])

    comp.update({}, 1.0)
    first = n_unconstrained()
    for _ in range(4):
        comp.update({}, 1.0)
    assert n_unconstrained() == first, (
        f"importer set grew {first} -> {n_unconstrained()} over four ticks; "
        "the store is concatenating rather than replacing"
    )
