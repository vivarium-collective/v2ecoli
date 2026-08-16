"""Unit tests for v2ecoli.composites.ecoli_structural."""

import os

import pytest


@pytest.mark.fast
def test_baseline_parsimony_function_is_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import ecoli_structural  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "ecoli_structural" in names


@pytest.mark.fast
def test_pack_out_dir_reads_workspace_yaml_layout(tmp_path, monkeypatch):
    """_resolve_pack_out_dir honours workspace.yaml's layout.studies via the
    shared viva_workspace resolver, returning the absolute
    ``<root>/<layout.studies>/<study>/viz/3d`` path (a custom studies dir here
    proves the layout map is genuinely read)."""
    from v2ecoli.composites.ecoli_structural import _resolve_pack_out_dir

    (tmp_path / "workspace.yaml").write_text(
        "layout:\n  studies: custom/studies\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_pack_out_dir(None, "itest") == str(
        tmp_path / "custom" / "studies" / "itest" / "viz" / "3d")


@pytest.mark.fast
def test_pack_out_dir_falls_back_without_workspace_yaml(tmp_path, monkeypatch):
    """No workspace.yaml on the CWD ancestry -> the conventional CWD-relative
    ``workspace/studies/<study>/viz/3d`` default (unchanged legacy fallback)."""
    from v2ecoli.composites.ecoli_structural import _resolve_pack_out_dir

    monkeypatch.chdir(tmp_path)  # empty dir, no workspace.yaml above

    assert _resolve_pack_out_dir(None, "itest") == "workspace/studies/itest/viz/3d"


@pytest.mark.fast
def test_pack_out_dir_flat_default_when_layout_studies_unset(tmp_path, monkeypatch):
    """workspace.yaml present but with no layout.studies key -> the shared
    resolver's conventional FLAT ``studies/`` layout (absolute). (v2ecoli's real
    workspace.yaml always sets layout.studies, so this only affects bare
    workspaces.)"""
    from v2ecoli.composites.ecoli_structural import _resolve_pack_out_dir

    (tmp_path / "workspace.yaml").write_text("name: v2ecoli\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_pack_out_dir(None, "itest") == str(
        tmp_path / "studies" / "itest" / "viz" / "3d")


@pytest.mark.fast
def test_resolve_pack_out_dir_explicit_override_wins(tmp_path, monkeypatch):
    """An explicit out_dir override is used verbatim — never re-derived from
    the workspace layout, even when a workspace.yaml with a DIFFERENT studies
    root is present in CWD."""
    from v2ecoli.composites.ecoli_structural import _resolve_pack_out_dir

    (tmp_path / "workspace.yaml").write_text(
        "layout:\n  studies: some/other/studies\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_pack_out_dir("explicit/out/dir", "itest") == "explicit/out/dir"


@pytest.mark.fast
def test_resolve_pack_out_dir_derives_default_from_layout(tmp_path, monkeypatch):
    """No override -> derives the absolute '<root>/<layout.studies>/<study>/viz/3d'
    from the shared resolver."""
    from v2ecoli.composites.ecoli_structural import _resolve_pack_out_dir

    (tmp_path / "workspace.yaml").write_text(
        "layout:\n  studies: workspace/studies\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_pack_out_dir(None, "itest") == str(
        tmp_path / "workspace" / "studies" / "itest" / "viz" / "3d")


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
        "ecoli_structural", core=core, seed=0, cache_dir="out/cache",
        study="s01-birth-and-division", emitter="null")  # doc-shape only; do not run

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


def test_generator_appends_pack_step_with_core_none():
    """Regression: the pack step must be appended on the STANDARD build path,
    which builds the document with ``core=None`` (viva_superpowers'
    ``build_generator``/``to_document`` forwards ``None`` to the builder; the
    real core is attached only at ``Composite(...)`` construction, and the
    ``EcoliPackStep`` link is registered via the ``core_extensions`` hook).

    Before the fix the whole append was gated on ``core is not None`` and
    returned the bare baseline early — so every real dashboard/study run built
    a composite with NO pack step and silently wrote zero snapshot packs, even
    though :func:`test_generator_appends_pack_step` (which passes a real core)
    stayed green.
    """
    # Building the composite loads the ParCa cache (out/cache/initial_state.json).
    # Gate on that FILE, not just the dir or the CI env: the fast-tests CI job
    # runs this build-only test but does NOT build the ParCa cache, so a
    # dir/CI-based guard let it run and FileNotFoundError'd. Skip cleanly when
    # the cache isn't present; the full behavior-tests job (which builds it) runs
    # the assertion.
    if not os.path.isfile("out/cache/initial_state.json"):
        pytest.skip("ParCa cache (out/cache/initial_state.json) not present; "
                    "build via `python scripts/build_cache.py`")
    from viva_superpowers.composite_generator import (
        _REGISTRY, build_generator, discover_generators,
    )
    from v2ecoli.composites import ecoli_structural  # noqa: F401 — fires decorator

    if not _REGISTRY:
        discover_generators()
    entry = _REGISTRY["v2ecoli.composites.ecoli_structural.ecoli_structural"]
    # core=None — exactly how the dashboard subprocess builds the document.
    doc = build_generator(
        entry,
        overrides={"seed": 0, "cache_dir": "out/cache",
                   "study": "s01-birth-and-division"},
    )
    agents = doc["state"]["agents"]
    cell = next(iter(agents.values()))
    assert "pack_step" in cell, (
        "pack_step missing when the document is built with core=None — the "
        "standard run path would write no snapshot packs")
    # And the link is discoverable for the runner via the core_extensions hook.
    assert _register_pack_step_hook_registers_link()


def _register_pack_step_hook_registers_link():
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_structural import _register_ecoli_pack_step
    core = build_core()
    _register_ecoli_pack_step(core)
    return "EcoliPackStep" in getattr(core, "link_registry", {})
