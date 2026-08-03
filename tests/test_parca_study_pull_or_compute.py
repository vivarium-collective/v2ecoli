"""Comparison convergence Phase 2: ParCa study pull-or-compute contract.

Hermetic — NO real ParCa build ever runs. `resolve_or_build_parca(...,
build=False)` (the default, and the only mode exercised here) is pure CHECK
logic: file/sidecar reads plus one `git rev-parse HEAD` subprocess call. The
`build=True` compute branch (`_build_candidate`/`_build_reference`) is never
invoked by this file.

This is the regression suite for the gate-e2e-report.md §A blocker: a
vEcoli-native ParCa cache built against an older fork commit than the one
currently checked out silently produces a version-skewed simData (a 45-vs-41
molecule-array shape mismatch several frames into a real sim step). The
reference-engine tests below simulate exactly that skew via the
producing-commit sidecar, without needing a real vEcoli checkout.
"""
from __future__ import annotations

import json

import pytest

import v2ecoli.workflow.parca_study as parca_study
from v2ecoli.library.cache_version import StaleCacheError


# --- candidate (v2ecoli) ----------------------------------------------------

def test_candidate_reused_when_verify_cache_version_passes(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache_full"
    cache_dir.mkdir()
    monkeypatch.setattr(parca_study, "verify_cache_version", lambda cache_dir: None)

    result = parca_study.resolve_or_build_parca(
        parca_study.CANDIDATE_ENGINE, str(cache_dir))

    assert result["status"] == "reused"
    assert result["path"] == str(cache_dir)


def test_candidate_stale_when_verify_cache_version_raises(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache_full"
    cache_dir.mkdir()

    def _raise(cache_dir):
        raise StaleCacheError("inputs_hash mismatch")

    monkeypatch.setattr(parca_study, "verify_cache_version", _raise)

    result = parca_study.resolve_or_build_parca(
        parca_study.CANDIDATE_ENGINE, str(cache_dir))

    assert result["status"] in ("stale", "rebuilt")
    assert result["status"] != "reused"
    assert "inputs_hash mismatch" in result["reason"]


def test_candidate_stale_when_cache_absent(tmp_path):
    # No monkeypatch -- real verify_cache_version runs against a directory
    # that has no cache_version.json at all, which raises StaleCacheError
    # (the pre-versioning-cache case) purely from file-presence checks; no
    # ParCa build/heavy compute happens.
    cache_dir = tmp_path / "does-not-exist"

    result = parca_study.resolve_or_build_parca(
        parca_study.CANDIDATE_ENGINE, str(cache_dir))

    assert result["status"] in ("stale", "rebuilt")
    assert result["status"] != "reused"


def test_candidate_build_true_actually_invokes_compute_branch(tmp_path, monkeypatch):
    """Sanity check on the status vocabulary: build=True + a needed rebuild
    reports "rebuilt" (not "stale"), and the compute branch is what a caller
    would gate behind `build=True` -- but even here the compute branch itself
    is stubbed via monkeypatch, so no real ParCa build ever runs."""
    cache_dir = tmp_path / "cache_full"
    cache_dir.mkdir()

    def _raise(cache_dir):
        raise StaleCacheError("stale")

    monkeypatch.setattr(parca_study, "verify_cache_version", _raise)
    called = {}
    monkeypatch.setattr(parca_study, "_build_candidate",
                        lambda cache_dir: called.setdefault("built", cache_dir))

    result = parca_study.resolve_or_build_parca(
        parca_study.CANDIDATE_ENGINE, str(cache_dir), build=True)

    assert result["status"] == "rebuilt"
    assert called["built"] == str(cache_dir)


# --- reference (vEcoli) -----------------------------------------------------

def _stage_reference_cache(tmp_path, commit: str | None):
    cache_dir = tmp_path / "vecoli_parca"
    cache_dir.mkdir()
    (cache_dir / "simData.cPickle").write_bytes(b"fake-simdata")
    if commit is not None:
        parca_study.write_producing_commit(str(cache_dir), commit)
    return cache_dir


def test_reference_reused_when_producing_commit_matches_head(tmp_path, monkeypatch):
    cache_dir = _stage_reference_cache(tmp_path, commit="abc123")
    monkeypatch.setattr(parca_study, "_current_vecoli_commit",
                        lambda reference_repo: "abc123")

    result = parca_study.resolve_or_build_parca(
        parca_study.REFERENCE_ENGINE, str(cache_dir),
        reference_repo="/fake/vecoli-fork")

    assert result["status"] == "reused"


def test_reference_stale_when_producing_commit_mismatches_head(tmp_path, monkeypatch):
    """Simulates gate-e2e-report.md §A: the cache was built by an older
    commit than the fork's current HEAD."""
    cache_dir = _stage_reference_cache(tmp_path, commit="OLD-commit-14f04a3f-era")
    monkeypatch.setattr(parca_study, "_current_vecoli_commit",
                        lambda reference_repo: "NEW-commit-d2f95129")

    result = parca_study.resolve_or_build_parca(
        parca_study.REFERENCE_ENGINE, str(cache_dir),
        reference_repo="/fake/vecoli-fork")

    assert result["status"] in ("stale", "rebuilt")
    assert result["status"] != "reused"
    assert "version-skew" in result["reason"]


def test_reference_stale_when_no_provenance_sidecar(tmp_path, monkeypatch):
    cache_dir = _stage_reference_cache(tmp_path, commit=None)
    monkeypatch.setattr(parca_study, "_current_vecoli_commit",
                        lambda reference_repo: "abc123")

    result = parca_study.resolve_or_build_parca(
        parca_study.REFERENCE_ENGINE, str(cache_dir),
        reference_repo="/fake/vecoli-fork")

    assert result["status"] != "reused"


def test_reference_stale_when_simdata_absent(tmp_path, monkeypatch):
    cache_dir = tmp_path / "vecoli_parca"
    cache_dir.mkdir()
    monkeypatch.setattr(parca_study, "_current_vecoli_commit",
                        lambda reference_repo: "abc123")

    result = parca_study.resolve_or_build_parca(
        parca_study.REFERENCE_ENGINE, str(cache_dir),
        reference_repo="/fake/vecoli-fork")

    assert result["status"] != "reused"
    assert "does not exist" in result["reason"]


def test_reference_stale_when_current_head_unresolvable(tmp_path, monkeypatch):
    cache_dir = _stage_reference_cache(tmp_path, commit="abc123")
    monkeypatch.setattr(parca_study, "_current_vecoli_commit",
                        lambda reference_repo: None)

    result = parca_study.resolve_or_build_parca(
        parca_study.REFERENCE_ENGINE, str(cache_dir),
        reference_repo="/fake/vecoli-fork")

    assert result["status"] != "reused"


# --- provenance sidecar round-trip -----------------------------------------

def test_write_and_read_producing_commit_round_trips(tmp_path):
    cache_dir = tmp_path / "vecoli_parca"
    parca_study.write_producing_commit(str(cache_dir), "abc123")

    assert parca_study.read_producing_commit(str(cache_dir)) == "abc123"

    sidecar = cache_dir / parca_study.PROVENANCE_FILENAME
    assert json.loads(sidecar.read_text())["vecoli_commit"] == "abc123"


def test_read_producing_commit_none_when_absent(tmp_path):
    assert parca_study.read_producing_commit(str(tmp_path)) is None


# --- prerequisite_edge shape (matches vivarium_workbench's own convention) --

def test_prerequisite_edge_matches_study_seed_shape():
    edge = parca_study.prerequisite_edge("parca")
    assert edge == {"study": "parca", "relation": "leads-to"}


# --- unknown engine ----------------------------------------------------------

def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        parca_study.resolve_or_build_parca("bogus-engine", "/tmp/whatever")
