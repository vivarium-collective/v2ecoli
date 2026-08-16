"""Unit tests for v2ecoli.composites.ecoli_baseline."""

import os

import pytest


@pytest.mark.fast
def test_baseline_function_is_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import ecoli_baseline  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "ecoli_baseline" in names


@pytest.mark.fast
def test_baseline_function_signature():
    """The generator takes (core, *, seed, cache_dir,
    transcript_initiation_mode, polypeptide_initiation_mode, config_overrides,
    feature toggles, emitter, bundle)."""
    import inspect
    from v2ecoli.composites.ecoli_baseline import baseline
    sig = inspect.signature(baseline)
    # Subset (not exact-equality) so legitimate ADDITIVE signature growth — e.g.
    # the #373 composite unification added media/knockouts/n_seeds/study/analyses/… —
    # doesn't hard-break this contract test. The stable core params must remain.
    required = {
        "core", "seed", "cache_dir", "transcript_initiation_mode",
        "polypeptide_initiation_mode", "config_overrides",
        "ppgpp_regulation", "trna_attenuation", "supercoiling",
        "mass_conservation", "emitter", "bundle", "features",
        "injected_processes"}
    missing = required - set(sig.parameters)
    assert not missing, f"baseline() lost core params: {missing}"


@pytest.mark.sim
def test_baseline_returns_a_document():
    """End-to-end: call baseline() with the test fixture cache, assert it
    returns a process-bigraph document dict."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    assert isinstance(doc, dict)
    assert len(doc) > 0


@pytest.mark.sim
def test_baseline_composite_has_no_unbound_core_instances():
    """Regression guard for item 57: ``_get_step_config``'s Requester/Evolver
    wrapping (``ecoli_baseline.py``, the ``PARTITIONED_PROCESSES`` branch)
    constructed both with a single positional config dict and no ``core=``
    kwarg. ``EcoliStep.__init__`` then fell back to the ambient
    ``_CURRENT_CORE`` global -- already reset to ``None`` by the
    ``set_current_core(None)`` call made right after building the shared
    partitioned-process object (or never set at all on a ``process_cache``
    hit). The resulting ``core=None`` instance built and ticked without
    error -- ``.core`` is only ever read later, by ``bigraph_schema``'s
    ``Link.serialize`` (``instance.core.access(...)``) when
    ``Composite.serialize_state()`` finally runs -- so this silently
    shipped a composite that crashed only at the very last step of a real
    run, confirmed live via sim 154 (sms-ecoli build 64, commit c2ae8eb).

    This walks the ENTIRE real composite state (not just the two known
    Requester/Evolver ports) so any future construction site with the same
    disease is caught here too, not just the ones already found by hand.

    Real, non-mocked construction: builds the actual document via the real
    cache and inspects the real instances Composite() realizes, rather than
    ticking a full generation (unnecessary -- ``.core`` is set once at
    construction and never touched again, so the defect is observable
    immediately after Composite() builds, with no simulation runtime cost)."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    from _core_binding_check import unbound_core_instances

    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    composite = Composite(doc, core=core)

    agent = composite.state["agents"]["0"]
    partitioned_steps = [k for k in agent if k.endswith("_requester") or k.endswith("_evolver")]
    assert partitioned_steps, (
        "expected at least one Requester/Evolver step under agents.0 -- "
        "PARTITIONED_PROCESSES wiring may have changed; update this test's "
        "assumptions rather than silently passing on zero coverage")

    unbound = unbound_core_instances(composite.state)
    assert not unbound, (
        f"instance(s) with core=None found in the real composite state: {unbound} -- "
        "Composite.serialize_state() will crash on these with "
        "AttributeError: 'NoneType' object has no attribute 'access' "
        "(bigraph_schema/methods/serialize.py, Link.serialize) the moment "
        "a real run reaches its final state write.")
