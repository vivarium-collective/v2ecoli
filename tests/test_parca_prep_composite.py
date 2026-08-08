"""``parca_prep`` composite: ParCa's pull-or-compute contract wrapped as an
ordinary ``@composite_generator`` study (Gap 1). Hermetic -- every test
monkeypatches ``v2ecoli.composites.parca_prep.resolve_or_build_parca`` (the
module-level name ``parca_prep`` actually looks up), so no real ParCa build
or file I/O ever runs.
"""
from __future__ import annotations

import v2ecoli.composites.parca_prep as parca_prep_module
from v2ecoli.composites.parca_prep import parca_prep


def test_candidate_reused_no_build_call(monkeypatch):
    calls = []

    def fake_resolve(engine, cache_dir, *, reference_repo="", build=False):
        calls.append({"engine": engine, "cache_dir": cache_dir, "build": build})
        return {"status": "reused", "path": cache_dir, "reason": "verify_cache_version passed"}

    monkeypatch.setattr(parca_prep_module, "resolve_or_build_parca", fake_resolve)

    result = parca_prep(candidate_cache_dir="candidate_cache")

    state = result["state"]["parca_prep"]
    assert state["candidate"] == {"status": "reused", "path": "candidate_cache"}
    assert "reference" not in state
    assert len(calls) == 1
    assert calls[0]["build"] is False


def test_candidate_stale_then_rebuilt(monkeypatch):
    calls = []

    def fake_resolve(engine, cache_dir, *, reference_repo="", build=False):
        calls.append(build)
        if not build:
            return {"status": "stale", "path": cache_dir, "reason": "inputs_hash mismatch"}
        return {"status": "rebuilt", "path": cache_dir, "reason": "inputs_hash mismatch"}

    monkeypatch.setattr(parca_prep_module, "resolve_or_build_parca", fake_resolve)

    result = parca_prep(candidate_cache_dir="candidate_cache")

    state = result["state"]["parca_prep"]
    assert state["candidate"] == {"status": "rebuilt", "path": "candidate_cache"}
    assert calls.count(True) == 1
    assert calls.count(False) == 1


def test_candidate_stale_no_rebuild_when_build_if_stale_false(monkeypatch):
    calls = []

    def fake_resolve(engine, cache_dir, *, reference_repo="", build=False):
        calls.append(build)
        return {"status": "stale", "path": cache_dir, "reason": "inputs_hash mismatch"}

    monkeypatch.setattr(parca_prep_module, "resolve_or_build_parca", fake_resolve)

    result = parca_prep(candidate_cache_dir="candidate_cache", build_if_stale=False)

    state = result["state"]["parca_prep"]
    assert state["candidate"] == {"status": "stale", "path": "candidate_cache"}
    assert len(calls) == 1
    assert calls[0] is False


def test_reference_cache_dir_resolves_both_engines(monkeypatch):
    seen_engines = []

    def fake_resolve(engine, cache_dir, *, reference_repo="", build=False):
        seen_engines.append(engine)
        return {"status": "reused", "path": cache_dir, "reason": "ok"}

    monkeypatch.setattr(parca_prep_module, "resolve_or_build_parca", fake_resolve)

    result = parca_prep(
        candidate_cache_dir="candidate_cache",
        reference_cache_dir="reference_cache",
        reference_repo="/fake/vecoli-fork",
    )

    state = result["state"]["parca_prep"]
    assert state["candidate"] == {"status": "reused", "path": "candidate_cache"}
    assert state["reference"] == {"status": "reused", "path": "reference_cache"}
    assert seen_engines == ["candidate", "reference"]


def test_reference_omitted_when_reference_cache_dir_empty(monkeypatch):
    def fake_resolve(engine, cache_dir, *, reference_repo="", build=False):
        return {"status": "reused", "path": cache_dir, "reason": "ok"}

    monkeypatch.setattr(parca_prep_module, "resolve_or_build_parca", fake_resolve)

    result = parca_prep(candidate_cache_dir="candidate_cache")

    assert "reference" not in result["state"]["parca_prep"]
