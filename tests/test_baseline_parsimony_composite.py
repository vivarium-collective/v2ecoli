"""Unit tests for v2ecoli.composites.baseline_parsimony."""

import os

import pytest


@pytest.mark.fast
def test_baseline_parsimony_function_is_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import baseline_parsimony  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "baseline_parsimony" in names


@pytest.mark.sim
def test_generator_appends_pack_step():
    """End-to-end: build the composite (doc-shape only, no run) and verify
    'pack_step' is wired into the per-agent cell state as a final execution
    layer, pointed at the VERIFIED real store paths:

    - 'bulk', 'shape', 'global_time' are top-level cell-state stores (same
      paths ShapeStep itself reads/writes — see baseline.py:895-919).
    - 'full_chromosome' lives under ['unique', 'full_chromosome'] — the SAME
      path v2ecoli/steps/division.py's MarkDPeriod uses (wired in
      v2ecoli/composites/_helpers.py:1483), confirmed empirically by
      building baseline() and inspecting the built cell state's keys.
    """
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    import v2ecoli
    from v2ecoli.core import build_core

    core = build_core()
    comp = v2ecoli.build_composite(
        "baseline_parsimony", core=core, seed=0, cache_dir="out/cache",
        study="ecoli-3d", emitter="null")  # doc-shape only; do not run

    state = comp.state
    cell = next(iter(state["agents"].values()))

    assert "pack_step" in cell
    ps = cell["pack_step"]
    # Composite parses "local:X" addresses into {'protocol': 'local', 'data': 'X'}
    # once built (verified against shape_step's built address, same pattern).
    address = ps["address"]
    address_str = address["data"] if isinstance(address, dict) else address
    assert address_str.endswith("EcoliPackStep")
    assert ps["inputs"]["bulk"] == ["bulk"]
    assert ps["inputs"]["shape"] == ["shape"]
    assert ps["inputs"]["global_time"] == ["global_time"]
    assert ps["inputs"]["full_chromosome"] == ["unique", "full_chromosome"]

    # pack_step must be wired as the FINAL execution layer: strictly lower
    # priority than every other step (process-bigraph: lower priority number
    # fires later), and depend (via an injected flow token) on whatever was
    # previously the last step in the flow (shape_step).
    priorities = {name: edge["priority"] for name, edge in cell.items()
                  if isinstance(edge, dict) and "priority" in edge}
    assert priorities["pack_step"] == min(priorities.values())
    assert any(key.startswith("_layer_in_") for key in ps["inputs"])
