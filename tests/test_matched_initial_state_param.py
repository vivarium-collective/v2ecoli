"""Matched-initial-state as a declarative ``ecoli_baseline`` param.

Phase 2 Task 1: the candidate composite (v2ecoli) must be able to start from
a REFERENCE vEcoli's ParCa ``simData.cPickle`` — the same t=0 both engines
share in a comparison — via a plain ``match_simdata: str | None`` param,
reusing the SAME mechanism ``scripts/run_comparison_ensemble.py`` already
applies through ``--match-vecoli-simdata``/``--match-initial-state``
(``_vecoli_reference_state`` + ``_apply_bulk_overlay``), not a
reimplementation.

Fully hermetic: no ParCa cache or upstream vEcoli fork checkout required.
``_vecoli_reference_state`` is the only heavy piece (it builds the genuine
upstream vEcoli engine) and is monkeypatched out; the cache bundle loader is
monkeypatched to a tiny synthetic bundle, and the 55-process execution-layer
build is monkeypatched to an empty flow (a document-assembly concern
orthogonal to matched-initial-state) so the real ``baseline()`` document
builder runs end-to-end quickly.
"""
from __future__ import annotations

import numpy as np
import pytest

import v2ecoli.composites.ecoli_baseline as eb
import scripts.run_comparison_ensemble as rce


BULK_DTYPE = np.dtype([('id', 'U20'), ('count', 'i8')])


def _fake_bulk():
    return np.array(
        [('A', 10), ('B', 20), ('C', 30)], dtype=BULK_DTYPE)


def _fake_bundle():
    return {
        "initial_state": {
            "bulk": _fake_bulk(),
            "environment": {"media_id": "minimal"},
        },
        "configs": {},
        "unique_names": [],
        "dry_mass_inc_dict": {},
    }


@pytest.fixture
def hermetic_baseline_build(monkeypatch):
    """Make ``ecoli_baseline.baseline()`` buildable without a real ParCa
    cache: stub the cache loader with a tiny synthetic bundle and skip the
    (unrelated) 55-process execution-layer assembly by making the flow
    empty. Matched-initial-state only touches ``cell_state['bulk']``, which
    is populated straight from the (fake) cache bundle regardless."""
    monkeypatch.setattr(eb, "load_cache_bundle", lambda cache_dir: _fake_bundle())
    monkeypatch.setattr(eb, "build_execution_layers", lambda features=None: [])


def test_match_simdata_param_is_declared():
    """``match_simdata`` is a first-class declared parameter of the
    ``ecoli_baseline`` composite (not bespoke CLI logic), defaulting to
    None/unset."""
    entry = eb.baseline._composite_generator_entry
    assert "match_simdata" in entry.parameters
    decl = entry.parameters["match_simdata"]
    assert decl["default"] is None
    assert decl["type"] == "string"


def test_match_simdata_none_leaves_default_initial_state_unchanged(
        hermetic_baseline_build, monkeypatch):
    """The default path (``match_simdata=None``) is byte-identical to
    before: the matched-init mechanism is never invoked, and the initial
    bulk counts come straight from the cache bundle, untouched."""
    def _must_not_be_called(*a, **k):
        raise AssertionError(
            "_vecoli_reference_state must not be called when match_simdata "
            "is unset")

    monkeypatch.setattr(rce, "_vecoli_reference_state", _must_not_be_called)

    doc = eb.baseline(cache_dir="fake_cache", seed=0)  # match_simdata omitted
    bulk = doc["state"]["agents"]["0"]["bulk"]

    assert list(bulk["id"]) == ["A", "B", "C"]
    assert list(bulk["count"]) == [10, 20, 30]  # unchanged from the cache


def test_match_simdata_overlays_bulk_from_reference_simdata(
        hermetic_baseline_build, monkeypatch):
    """With ``match_simdata=<path>`` set, the candidate's t=0 bulk counts are
    overlaid from that reference's genuine-vEcoli pre-run state — DIFFERENT
    from the default-cache initial state — via the SAME
    ``_vecoli_reference_state``/``_apply_bulk_overlay`` mechanism
    ``run_comparison_ensemble.py`` uses (reused, not reimplemented)."""
    calls = []

    def fake_reference_state(sim_data_path, condition, seed, fork_dir):
        calls.append((sim_data_path, condition, seed, fork_dir))
        # A known, DIFFERENT bulk count for 'A' and 'B'; 'C' absent from the
        # reference (must be left untouched — only molecules present in BOTH
        # engines are overlaid).
        return {"A": 999, "B": 5}, {}

    monkeypatch.setattr(rce, "_vecoli_reference_state", fake_reference_state)

    doc = eb.baseline(
        cache_dir="fake_cache", seed=7,
        match_simdata="/fake/reference/simData.cPickle")
    bulk = doc["state"]["agents"]["0"]["bulk"]

    # Routed into the reused mechanism with the right args (abspath, seed
    # threaded through; condition defaults to "basal" — Task 4 threads the
    # per-config condition explicitly).
    assert len(calls) == 1
    sim_data_path, condition, seed, fork_dir = calls[0]
    assert sim_data_path.endswith("/fake/reference/simData.cPickle")
    assert condition == "basal"
    assert seed == 7

    # The overlay actually changed the candidate's initial state to match
    # the reference's — different from the default-cache draw (10, 20).
    counts = dict(zip(bulk["id"], bulk["count"]))
    assert counts["A"] == 999
    assert counts["B"] == 5
    # 'C' is not in the reference bulk: left as the cache's own value.
    assert counts["C"] == 30


def test_match_simdata_batch_mode_fails_loud(monkeypatch):
    """match_simdata is not yet wired through batch mode (n_seeds>1 /
    n_generations>1 dispatches to a separate run-time orchestrator); it must
    fail loud rather than silently ignore the param."""
    with pytest.raises(ValueError, match="match_simdata"):
        eb.baseline(
            cache_dir="fake_cache", n_seeds=2,
            match_simdata="/fake/reference/simData.cPickle")
