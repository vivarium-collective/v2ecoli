"""#505: a config_overrides / knockouts perturbation must survive division.

v2ecoli's Division step rebuilds each daughter via ``baseline()`` from the plain
cache. It threads ``injected_processes`` and the exchange-flux map to that
rebuild — but a perturbation supplied as ``config_overrides`` (a variant /
sensitivity sweep, or a knockout, which folds into config_overrides) was NOT
threaded, so it applied to generation 1 only and every daughter silently reverted
to the unperturbed cached configs. This breaks any multi-generation perturbation
study.

Asserted at the seam (initialize + the baseline() call source + the _helpers
division config) rather than by running two generations — a real two-generation
run is ~25 minutes, and the defect is entirely in what the daughter rebuild is
given. Same discipline as ``test_the_division_step_passes_the_flux_map_to_each_daughter``.
"""
import inspect

import pytest

from v2ecoli.composites import _helpers
from v2ecoli.steps import division as division_mod


@pytest.mark.fast
def test_division_initialize_lifts_config_overrides_off_its_parameters():
    step = object.__new__(division_mod.Division)
    step.parameters = {"injected_processes": None,
                       "config_overrides": {"ecoli-polypeptide-elongation.basal_elongation_rate": 19.5}}
    division_mod.Division.initialize(step, step.parameters)
    assert getattr(step, "_config_overrides", None) == {
        "ecoli-polypeptide-elongation.basal_elongation_rate": 19.5}, (
        "Division.initialize dropped config_overrides")


@pytest.mark.fast
def test_the_daughter_rebuild_hands_config_overrides_to_baseline():
    """Holding the overrides without passing them on would satisfy the
    initialize test while still producing a lineage that reverts at generation 2."""
    src = inspect.getsource(division_mod)
    call = src[src.index("doc = baseline("):]
    call = call[:call.index(")")]
    assert "config_overrides=" in call, (
        "the daughter's baseline() rebuild does not pass config_overrides, so a "
        "perturbation reverts to the cached configs from generation 2 on (#505)")


@pytest.mark.fast
def test_the_division_config_is_threaded_only_when_overrides_are_present():
    """The guard: a plain baseline stashes config_overrides=None, and the
    division config must then be byte-for-byte unchanged — no config_overrides
    key at all — so the normal FBA lineage is untouched."""
    src = inspect.getsource(_helpers)
    # The threading reads the stash off the loader and sets it only when truthy.
    assert "_config_overrides" in src and "getattr(loader, '_config_overrides'" in src, (
        "_helpers does not thread config_overrides from the loader into div_config")
    block = src[src.index("getattr(loader, '_config_overrides'"):]
    guard = block[:block.index("div_config['config_overrides']")]
    assert "if _overrides" in guard, (
        "config_overrides is threaded unconditionally; a None/empty override must "
        "leave the plain baseline's division config unchanged")
