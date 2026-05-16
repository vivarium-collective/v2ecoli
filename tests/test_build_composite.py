"""Smoke tests for the v2ecoli.build_composite top-level helper."""

import pytest


@pytest.mark.fast
def test_build_composite_unknown_name_raises():
    from v2ecoli import build_composite
    with pytest.raises(ValueError, match="unknown composite architecture"):
        build_composite("nonexistent", seed=0)


@pytest.mark.fast
def test_generators_registered_under_short_names():
    """After importing v2ecoli, the registry contains all three architectures."""
    import v2ecoli  # noqa: F401 — forces decorator registration
    from pbg_superpowers.composite_generator import _REGISTRY
    names = {e.name for e in _REGISTRY.values() if e.module.startswith("v2ecoli.")}
    assert {"baseline", "departitioned", "reconciled"} <= names


@pytest.mark.sim
def test_build_composite_baseline_returns_composite():
    import os
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli import build_composite
    from process_bigraph.composite import Composite
    comp = build_composite("baseline", seed=0, cache_dir="out/cache")
    assert isinstance(comp, Composite)


@pytest.mark.sim
def test_build_composite_each_architecture():
    import os
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli import build_composite
    for name in ("baseline", "departitioned", "reconciled"):
        comp = build_composite(name, seed=0, cache_dir="out/cache")
        assert comp is not None, f"{name} produced no composite"


@pytest.mark.sim
def test_build_composite_accepts_core_override():
    import os
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli import build_composite
    from v2ecoli.core import build_core
    core = build_core()
    comp = build_composite("baseline", seed=0, cache_dir="out/cache", core=core)
    assert comp.core is core
