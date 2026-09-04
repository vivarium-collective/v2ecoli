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

★ #566 then fixed the third defect below, and the fix is NOT the same wrapper --
the difference is load-bearing and this file pins it deliberately
(``test_boundary_external_replaces_PER_LEAF_not_whole_map``). ``exchange_data``
wants WHOLE-MAP replace, because dropping a key is how "max flux 0" is
expressed. ``boundary.external`` must keep every molecule: its only two writers
(EnvironmentMirror, MediaUpdate) emit PARTIAL dicts, so ``overwrite[map[...]]``
there would delete every molecule the writer did not mention and the cell would
silently lose its media. It is declared ``map[overwrite[float[mM]]]`` --
per-leaf. Same symptom, opposite nesting.

A third defect lived at the step before: `EnvironmentMirror` wrote a DELTA, and
metabolism seeds unlimited molecules (canonically O2) at ``inf``. ``0.0 - inf``
is ``-inf``, which passes an isnan check and then accumulates to NaN in the
additive boundary store, silently poisoning that molecule. #548 made that fail
closed on any non-finite delta -- safe, but not fixed: it left inf-valued
molecules UNREACHABLE by any driver or reactor, which is why dissolved-O2
control did not work. #566 removes the delta entirely: both writers now write an
ABSOLUTE concentration onto per-leaf-overwrite leaves, so there is no inf
arithmetic left to guard. The finiteness guard moved from the DELTA to the
DRIVER'S VALUE, where a bad input is refused rather than neutralised by
arithmetic -- and ``+inf`` is deliberately ALLOWED, being this model's encoding
of "unlimited".

MediaUpdate carried the same defect and nobody had looked: its ``isnan`` guard
covered only the ``inf -> inf`` transition (``inf - inf`` is NaN), while
``inf -> finite`` produced ``-inf``, passed the guard, and accumulated to NaN.
Absolute writes close both transitions by construction.

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


def _declared_boundary_external_type() -> str:
    """The type string metabolism declares for ``boundary.external``.

    Read from source rather than hardcoded, so any test applying through it
    discriminates against a reverted declaration instead of silently testing
    the type it wishes were there.
    """
    import inspect
    import re as _re

    from v2ecoli.processes.metabolism import Metabolism

    src = inspect.getsource(Metabolism.inputs)
    match = _re.search(
        r"'boundary':\s*\{'external':\s*'([^']+)'\}", src
    )
    assert match, (
        "metabolism does not declare a type for boundary.external -- the "
        "declaration was reverted (see #566)"
    )
    return match.group(1)


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


def test_inf_boundary_is_now_writable(core):
    """The #566 fix: a driver CAN move an unlimited boundary.

    discriminates: under delta semantics this returned 0.0 -- the #548 guard
    refused the write, because ``5.0 - inf`` is ``-inf`` and the additive store
    would have landed NaN. That made the failure safe, not fixed, and left
    dissolved-O2 control unreachable by any driver or reactor.
    """
    assert _mirror_update(core, driver_conc=5.0, boundary_conc=float("inf")) == 5.0


def test_zero_write_onto_inf_boundary_REACHES_THE_STORE(core):
    """The vacuous-pass trap, pinned deliberately.

    The mirror emits 0.0 here under BOTH the old and new contract -- the same
    number for opposite reasons (old: "refused, write nothing"; new: "write the
    driver's 0.0"). Asserting on the mirror's output alone cannot tell them
    apart, so this asserts on what the STORE ends up holding, which is the
    thing that actually differs: inf under the old apply, 0.0 under the new.
    """
    emitted = _mirror_update(core, driver_conc=0.0, boundary_conc=float("inf"))
    assert emitted == 0.0
    # Apply through the type metabolism ACTUALLY DECLARES, not a hardcoded one.
    # Hardcoding the new type here would make this test pass against the old
    # source too -- vacuous in exactly the way it exists to catch.
    applied = core.apply(
        _declared_boundary_external_type(),
        {"GLC": float("inf")},
        {"GLC": emitted},
    )[0]["GLC"]
    magnitude = applied.magnitude if hasattr(applied, "magnitude") else applied
    assert magnitude == 0.0, (
        "the store still holds inf -- the mirror's 0.0 was applied as a delta"
    )


def test_ordinary_finite_write_is_absolute_not_a_delta(core):
    """The normal path every time-varying-environment run depends on.

    discriminates: the old contract returned ``5.0 - 11.101`` here. If this
    ever reads as a difference again, the declaration has been reverted and
    every driven concentration is silently wrong by the boundary's value.
    """
    assert _mirror_update(core, driver_conc=5.0, boundary_conc=11.101) == 5.0


def test_driver_may_restore_a_molecule_to_unlimited(core):
    """+inf is a MEANINGFUL value, not a bad one.

    It is this model's encoding of "unlimited" -- the state metabolism seeds
    unconstrained molecules in -- so a driver returning a molecule to unlimited
    is a legitimate write and must not be caught by the finiteness guard.
    """
    assert _mirror_update(core, driver_conc=float("inf"), boundary_conc=5.0) == float(
        "inf"
    )


def test_nan_driver_value_is_refused(core):
    """The guard moved from the delta to the DRIVER'S VALUE, and must still fire.

    Under replace semantics a bad driver value is written verbatim rather than
    neutralised by arithmetic, so refusing it here is load-bearing in a way it
    was not before. Regression guard for the behaviour the original isnan check
    provided.
    """
    assert _mirror_update(core, driver_conc=float("nan"), boundary_conc=1.0) is None


def test_negative_driver_value_is_refused(core):
    """A negative concentration is not physical, and replace semantics would
    now write one straight through."""
    assert _mirror_update(core, driver_conc=-1.0, boundary_conc=5.0) is None


def test_boundary_external_replaces_PER_LEAF_not_whole_map(core):
    """★ The distinction that makes this fix correct rather than catastrophic.

    `#548` fixed the sibling `exchange_data` stores with ``overwrite[map[...]]``,
    where replacing the WHOLE map is the point -- dropping a key is how "max
    flux 0" is expressed. Copying that idiom here would have been silent and
    severe: both writers of ``boundary.external`` (EnvironmentMirror,
    MediaUpdate) emit PARTIAL dicts, so a whole-map replace deletes every
    molecule the writer did not mention and the cell loses its media.

    This pins all three behaviours against each other so a future edit cannot
    quietly swap the nesting.
    """
    seeded = {"GLC": 20.0, "OXYGEN-MOLECULE": float("inf"), "ACET": 0.0}
    write = {"OXYGEN-MOLECULE": 6.58}

    frozen = core.apply("map[float]", dict(seeded), dict(write))[0]
    assert frozen["OXYGEN-MOLECULE"] == float("inf"), "bare float leaves accumulate"

    whole_map = core.apply("overwrite[map[float]]", dict(seeded), dict(write))[0]
    assert set(whole_map) == {"OXYGEN-MOLECULE"}, (
        "overwrite[map[...]] replaces the WHOLE map -- this is why #548's "
        "idiom must not be copied onto boundary.external"
    )

    per_leaf = core.apply("map[overwrite[float]]", dict(seeded), dict(write))[0]
    assert per_leaf["OXYGEN-MOLECULE"] == 6.58
    assert set(per_leaf) == set(seeded), "per-leaf overwrite must keep every molecule"


def test_metabolism_declares_per_leaf_replace_for_boundary_external():
    """Source-level check: the declaration is what makes the apply replace.

    Cheap, runs in the fast gate, and catches a revert that the unit tests
    above would not -- they exercise the type directly rather than the wiring.
    """
    import inspect

    from v2ecoli.processes.metabolism import Metabolism

    src = inspect.getsource(Metabolism.inputs)
    assert "'boundary': {'external': 'map[overwrite[float[mM]]]'}" in src, (
        "boundary.external must be declared per-leaf overwrite; see #566"
    )
    assert "'boundary': 'node'" not in src, "the untyped declaration is back"


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


@pytest.mark.sim
def test_driver_can_limit_dissolved_oxygen_end_to_end(core):
    """★ The claim #566 actually makes: an UNLIMITED molecule is now drivable.

    Everything else in this file, and all six of mbp-01's end-to-end tests, are
    BLIND to this change -- they exercise glucose, which was already finite and
    therefore already reachable by a delta. They pass identically before and
    after. This is the only test that can see the difference.

    Full path, no shortcuts: EnvironmentDriver writes the top-level store ->
    EnvironmentMirror resolves the id and writes boundary.external ->
    ExchangeData re-derives the constraint set -> metabolism sees it.

    discriminates: against the pre-fix tree ``boundary.external`` stays at
    ``inf`` forever. The mirror computed ``0.0 - inf = -inf``, #548's guard
    refused it as non-finite, and the write became a defined no-op -- safe, but
    leaving dissolved-O2 control unreachable by any driver or reactor.
    """
    from process_bigraph import Composite

    from v2ecoli.composites.ecoli_time_varying_env import baseline_time_varying_env
    from v2ecoli.steps.environment_driver import (
        ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        TRAJ_CLAMP_TO_VALUE,
    )

    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "OXYGEN-MOLECULE": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": 0.0},
        },
    )
    comp = Composite(doc, core=core)

    agent = comp.state["agents"]["0"]
    seeded = agent["boundary"]["external"]["OXYGEN-MOLECULE"]
    assert float(getattr(seeded, "magnitude", seeded)) == float("inf"), (
        "precondition: metabolism seeds O2 unlimited. If this ever starts "
        "finite, this test silently stops testing anything."
    )

    comp.run(2)

    agent = comp.state["agents"]["0"]
    driven = agent["boundary"]["external"]["OXYGEN-MOLECULE"]
    driven = float(getattr(driven, "magnitude", driven))
    assert driven == pytest.approx(0.0, abs=1e-9), (
        f"boundary.external['OXYGEN-MOLECULE'] is {driven!r} after the driver "
        f"clamped it to 0.0 -- an unlimited boundary is still unreachable"
    )

    # ...and the constraint set must actually respond. The threshold switch in
    # exchange_data_from_concentrations is generic over molecules, so once O2
    # can be moved below it, the existing machinery gates O2 with no O2-specific
    # code anywhere.
    unconstrained = agent["environment"]["exchange_data"].get("unconstrained") or []
    assert "OXYGEN-MOLECULE[p]" not in set(unconstrained), (
        "O2 fell below the import threshold but is still an unconstrained "
        "importer -- the boundary moved and metabolism did not notice"
    )


@pytest.mark.sim
def test_other_inf_molecules_are_untouched_when_one_is_driven(core):
    """Driving ONE unlimited molecule must not disturb the other eight.

    Guards the failure mode that per-leaf overwrite exists to prevent: nine
    molecules are seeded at inf, and a whole-map replace (or a writer emitting a
    partial dict onto one) would silently strip the rest.
    """
    from process_bigraph import Composite

    from v2ecoli.composites.ecoli_time_varying_env import baseline_time_varying_env
    from v2ecoli.steps.environment_driver import (
        ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        TRAJ_CLAMP_TO_VALUE,
    )

    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "OXYGEN-MOLECULE": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": 0.0},
        },
    )
    comp = Composite(doc, core=core)

    def inf_molecules():
        ext = comp.state["agents"]["0"]["boundary"]["external"]
        return {
            k for k, v in ext.items()
            if float(getattr(v, "magnitude", v)) == float("inf")
        }

    before = inf_molecules()
    assert len(before) > 1, "precondition: more than one unlimited molecule"

    comp.run(2)

    after = inf_molecules()
    assert after == before - {"OXYGEN-MOLECULE"}, (
        f"driving O2 disturbed other molecules: expected "
        f"{sorted(before - {'OXYGEN-MOLECULE'})}, got {sorted(after)}"
    )

    ext = comp.state["agents"]["0"]["boundary"]["external"]
    assert set(ext) >= before, "molecules disappeared from boundary.external entirely"
