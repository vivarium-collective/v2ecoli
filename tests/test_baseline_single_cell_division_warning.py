"""Issue #495 stopgap: single-cell ecoli_baseline has no division-stop.

Non-breaking coverage for the two cheap pieces shipped for #495:
  (a) the corrected ``n_generations`` documentation states plainly that it is
      inert on a single-cell (n_seeds=1, n_generations=1) run and that such a
      run does NOT stop at division; and
  (b) building a single-cell baseline document emits a runtime warning that the
      run will continue past division.

The doc assertions are cache-free and always run. The live-warning assertion
needs the ParCa cache to build a real document, so it is skipped when absent.
"""
import inspect
import os
import warnings

import pytest

import v2ecoli.composites.ecoli_baseline as ecoli_baseline
from v2ecoli.composites.ecoli_baseline import baseline

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


def test_n_generations_doc_states_single_cell_has_no_division_stop():
    """Cache-free: the corrected #495 documentation is present."""
    src = inspect.getsource(ecoli_baseline)

    # Param-schema description for n_generations: must say it only engages under
    # batch mode / has no effect on the single-cell default.
    assert "ONLY engages under batch mode" in src
    assert "NO effect on the" in src

    # baseline() docstring: must state single-cell has no division-stop and cite
    # the issue.
    doc = inspect.getdoc(baseline) or src
    assert "#495" in src
    assert "no division-stop" in doc.lower() or "no division-stop" in src.lower()
    # Points readers at the current way to bound to one cell cycle (batch mode).
    assert "n_seeds=2" in src


@pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_single_cell_build_warns_no_division_stop():
    """A standalone single-cell build surfaces the #495 no-division-stop note."""
    from v2ecoli.core import build_core
    core = build_core()
    with pytest.warns(UserWarning, match=r"no division-stop"):
        # Force the warning to fire even if a prior build in this process
        # already tripped warnings' once-per-location filter.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            baseline(core=core, seed=0, cache_dir=CACHE)
