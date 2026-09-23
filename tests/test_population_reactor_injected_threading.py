"""``baseline_population`` / ``reactor_bird_coupled`` must forward ``injected_processes``.

Same class as the batch-path audit P0 fixed in #640, two call sites over. Both
composites build THROUGH ``ecoli_baseline.baseline``, but neither accepted nor
forwarded ``injected_processes``: ``baseline_population`` called the builder with
only ``(core, seed, cache_dir, config_overrides)`` and ``reactor_bird_coupled``
had no such parameter at all.

Consequence: the capability was UNREACHABLE at population/reactor scale. ⚠ Not
silent -- and the distinction is deliberate, because #640's batch case WAS.
``build_generator`` validates override keys against the declared parameter set,
so a config naming ``injected_processes`` here raised ``ValueError``, and the
Python kwarg raised ``TypeError``. A dropped value and a missing parameter need
different fixes, and conflating them mis-describes both.
⊕ Separately and still true: a CALLER that never requests injection gets a clean
run with the cache's default metabolism and a bit-exact zero product. That is a
real silent-zero, but its mechanism is "never asked", not "request dropped".

Two axes are pinned deliberately, because each is separately sufficient to make
the feature unreachable:
  1. the value is FORWARDED to baseline()  (a dropped kwarg)
  2. the parameter is DECLARED in the composite schema (an undeclared parameter
     cannot be set from a study/config-driven build even if the Python kwarg
     works)
"""

import pytest

SWAP = {"swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"}}


def _capture(monkeypatch, module_path, attr):
    """Replace a builder with a recorder, returning the captured kwargs dict."""
    import importlib
    mod = importlib.import_module(module_path)
    seen = {}

    def fake(core=None, **kwargs):
        seen.update(kwargs)
        # Minimal document shape the callers go on to mutate.
        return {"state": {}, "composition": {}}

    monkeypatch.setattr(mod, attr, fake)
    return seen


# --- axis 1: the value reaches baseline() -------------------------------------

def test_baseline_population_forwards_injected_processes(monkeypatch):
    from v2ecoli.composites import ecoli_population

    seen = _capture(monkeypatch, "v2ecoli.composites.ecoli_population",
                    "_baseline_builder")
    monkeypatch.setattr(ecoli_population, "add_population_aggregator",
                        lambda document, core, **kw: document)
    ecoli_population.baseline_population(object(), injected_processes=SWAP)

    assert "injected_processes" in seen, (
        "baseline_population dropped injected_processes — a population run "
        "against a pathway cache would build the cache's default metabolism")
    assert (seen["injected_processes"]["swap_processes"]
            == {"ecoli-metabolism": "ecoli-metabolism-redux"})


def test_baseline_population_default_is_none_not_absent(monkeypatch):
    """Default must still be passed, so baseline() applies its own default and
    the no-injection path stays byte-identical to before."""
    from v2ecoli.composites import ecoli_population

    seen = _capture(monkeypatch, "v2ecoli.composites.ecoli_population",
                    "_baseline_builder")
    monkeypatch.setattr(ecoli_population, "add_population_aggregator",
                        lambda document, core, **kw: document)
    ecoli_population.baseline_population(object())
    # `seen.get(...) in (None, {})` would be satisfied by a key that was NEVER
    # PASSED -- the exact condition this test denies -- so assert PRESENCE
    # first. Mutation-checked: deleting the forward in baseline_population
    # leaves the .get() form green and this form red.
    assert "injected_processes" in seen, (
        "the kwarg must be passed even when unset, so baseline() applies its "
        "own default rather than the caller silently omitting it")
    assert seen["injected_processes"] is None


def test_reactor_bird_coupled_forwards_injected_processes(monkeypatch):
    """The reactor composite must forward THROUGH baseline_population to baseline().

    ⚠ Capture at ``_baseline_builder``, NOT at ``baseline_population``. Patching
    ``baseline_population`` would prove only that the reactor hands the value to
    something of that name, and would stay GREEN if the forwarding INSIDE
    ``baseline_population`` were deleted — i.e. it would not test the word
    "THROUGH" in this docstring. Mutation-checked both ways.
    """
    from v2ecoli.composites import reactor_bird_coupled as rbc
    from v2ecoli.composites import ecoli_population

    seen = _capture(monkeypatch, "v2ecoli.composites.ecoli_population",
                    "_baseline_builder")
    monkeypatch.setattr(ecoli_population, "add_population_aggregator",
                        lambda document, core, **kw: document)
    monkeypatch.setattr(rbc, "add_reactor_coupling",
                        lambda document, core, **kw: document)
    rbc.reactor_bird_coupled(object(), injected_processes=SWAP)

    assert "injected_processes" in seen, (
        "reactor_bird_coupled dropped injected_processes — the coupled CD2 "
        "demo would secrete nothing into the reactor with no error")
    assert (seen["injected_processes"]["swap_processes"]
            == {"ecoli-metabolism": "ecoli-metabolism-redux"})


# --- axis 2: the parameter is reachable from a config-driven build ------------

@pytest.mark.parametrize("module_path,func_name", [
    ("v2ecoli.composites.ecoli_population", "baseline_population"),
    ("v2ecoli.composites.reactor_bird_coupled", "reactor_bird_coupled"),
])
def test_injected_processes_is_a_declared_parameter(module_path, func_name):
    """Declared in the composite_generator schema, not just a Python kwarg.

    A study drives these composites through the declared parameter set; a kwarg
    that exists in the signature but not in the schema is unreachable from a
    config and the feature would still be dead on the path that matters.
    """
    import importlib
    fn = getattr(importlib.import_module(module_path), func_name)

    entry = getattr(fn, "_composite_generator_entry", None)
    assert entry is not None, (
        f"{func_name} has no _composite_generator_entry — the introspection "
        f"point this test relies on moved; fix the test, do not skip it")
    params = entry.parameters

    assert "injected_processes" in params, (
        f"{func_name} accepts injected_processes as a kwarg but does not "
        f"DECLARE it — unreachable from a config-driven build")


# --- signature-level guard ----------------------------------------------------

@pytest.mark.parametrize("module_path,func_name", [
    ("v2ecoli.composites.ecoli_population", "baseline_population"),
    ("v2ecoli.composites.reactor_bird_coupled", "reactor_bird_coupled"),
])
def test_signature_accepts_injected_processes(module_path, func_name):
    import importlib
    import inspect
    fn = getattr(importlib.import_module(module_path), func_name)
    target = inspect.unwrap(fn)
    assert "injected_processes" in inspect.signature(target).parameters, (
        f"{func_name} does not accept injected_processes")
