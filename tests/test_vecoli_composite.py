"""Unit tests for v2ecoli.composites.vecoli — Task 1 of the comparison
convergence Phase-1 plan (register vEcoli as a template-made Composite).

See docs/superpowers/plans/2026-08-01-comparison-convergence-phase-1.md Task 1
and docs/superpowers/specs/2026-08-01-comparison-general-runner-convergence-design.md
§2/§5.
"""

import os

import pytest


@pytest.mark.fast
def test_vecoli_composite_registers():
    """The 'vecoli' generator is discoverable via the same registry
    ecoli_baseline is discoverable through (mirrors
    tests/test_composites_baseline.py::test_baseline_function_is_registered)."""
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import vecoli  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "vecoli" in names


@pytest.mark.fast
def test_vecoli_composite_declares_reference_repo_param():
    """The fork is an explicit, declared param (spec §5) — not an ambient
    env var — so it's discoverable the same way a caller/UI discovers any
    other composite param."""
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import vecoli  # noqa: F401 — fires decorator
    entries = [e for e in _REGISTRY.values() if e.name == "vecoli"]
    assert entries, "no 'vecoli' generator registered"
    entry = entries[0]
    assert "reference_repo" in entry.parameters
    assert entry.parameters["reference_repo"]["type"] == "string"


@pytest.mark.sim
def test_vecoli_composite_builds():
    """Constructing the composite (reference_repo=$V2E_VECOLI_DIR,
    condition='basal', seed=0) returns a document without running a sim.

    Hermetic in intent (build-only, no comp.run()/comp.update()) but the
    underlying engine builder still needs a real vEcoli fork checkout +
    matching ParCa simData to *construct* EcoliSim — so this only runs when
    $V2E_VECOLI_DIR is set (mirrors the cache_dir-presence skip pattern
    tests/test_composites_baseline.py::test_baseline_returns_a_document
    uses for ecoli_baseline's cache dependency). Marked `sim` so it is
    excluded from the broad `-m "not sim"` sweep regardless of env.
    """
    fork_dir = os.environ.get("V2E_VECOLI_DIR")
    if not fork_dir or not os.path.isdir(fork_dir):
        pytest.skip(
            "V2E_VECOLI_DIR not set (or not a directory); "
            "test_vecoli_composite_builds needs a real vEcoli fork checkout "
            "to construct the engine (build-only, no simulation run).")

    from v2ecoli.core import build_core
    from v2ecoli.composites.vecoli import vecoli

    core = build_core()
    doc = vecoli(core=core, reference_repo=fork_dir, condition="basal", seed=0)

    assert isinstance(doc, dict)
    assert "state" in doc
    state = doc["state"]
    assert "agents" in state
    assert "0" in state["agents"]
    assert "vivarium_ecoli" in state["agents"]["0"]
