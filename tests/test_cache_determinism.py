"""ParCa cache determinism: diagnosable fingerprint + atomic locked build.

Covers the v2ecoli-side root-cause fix for vivarium-workbench#1105:

a1 — a *diagnosable* fingerprint. ``candidate_repo_roots`` resolves each
INPUT_FILES entry against the nearest ``workspace.yaml`` above the calling
process's cwd, so two processes with different cwds can hash *different* source
trees for one ``out/cache`` and disagree — a StaleCacheError a fresh process
can't reproduce. The old mismatch message printed ``files differ: []`` whenever
the divergence was in context/chain rather than a file, making it undebuggable.
``fingerprint_diff`` now reports which file/context/chain fields differ and,
per file, the absolute path each side resolved from.

a3 — an *atomic* build. ``_write_sim_input_bundle`` builds into a sibling temp
dir under an exclusive lock and atomically renames into place, so a reader never
sees a half-written bundle and a mid-write failure leaves no partial directory.
``build_cache.py --if-stale`` is a no-op when this process already sees a valid
cache, so two sessions that both rebuild converge instead of ping-ponging.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import v2ecoli.core as core
from v2ecoli.library.cache_version import (
    CACHE_VERSION_FILENAME,
    StaleCacheError,
    compute_cache_version,
    fingerprint_diff,
    verify_cache_version,
    write_cache_version,
)


pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# a1 — fingerprint_diff diagnoses a same-rel-path-from-two-roots divergence
# ---------------------------------------------------------------------------

def test_fingerprint_diff_reports_file_resolved_from_two_roots(tmp_path):
    """The exact cwd-divergence incident: the SAME rel path resolves to two
    DIFFERENT roots with different bytes. ``fingerprint_diff`` must name the
    file, both hashes, AND the absolute path each side read from — that path
    pair is what tells a debugger "same file name, different tree"."""
    files = ("f.txt",)
    root_a = tmp_path / "tree_a"
    root_b = tmp_path / "tree_b"
    (root_a).mkdir()
    (root_b).mkdir()
    (root_a / "f.txt").write_bytes(b"AAA")
    (root_b / "f.txt").write_bytes(b"BBB")

    cv_a = compute_cache_version(repo_root=str(root_a), files=files)
    cv_b = compute_cache_version(repo_root=str(root_b), files=files)

    assert cv_a.inputs_hash != cv_b.inputs_hash

    diff = fingerprint_diff(cv_a, cv_b)
    assert "f.txt" in diff["files"]
    entry = diff["files"]["f.txt"]
    assert entry["stored"] != entry["current"]
    assert entry["stored"] is not None and entry["current"] is not None
    assert entry["stored_path"] == os.path.abspath(str(root_a / "f.txt"))
    assert entry["current_path"] == os.path.abspath(str(root_b / "f.txt"))
    assert entry["stored_path"] != entry["current_path"]
    # The two roots were also the two cwds' resolution — record both cwds.
    assert "current_cwd" in diff


def test_context_only_mismatch_does_not_print_empty_file_diff(tmp_path):
    """A context-only mismatch (an unpickle-affecting package bump, identical
    files) used to print ``files differ: []`` — the undebuggable message. The
    error must now show the context diff, carry an EMPTY files section, and
    attach the structured diff, never the misleading empty file list."""
    real = compute_cache_version()
    faked_ctx = dict(real.context)
    faked_ctx["dill"] = "0.0.0-fake"  # dill IS folded, so inputs_hash moves
    stored = compute_cache_version(context=faked_ctx)
    # Same files as a fresh recompute; only the context (dill) differs.
    assert stored.per_file_hashes == real.per_file_hashes

    write_cache_version(str(tmp_path), version=stored)

    with pytest.raises(StaleCacheError) as excinfo:
        verify_cache_version(str(tmp_path))

    msg = str(excinfo.value)
    assert "files differ: []" not in msg
    assert "dill" in msg  # the real culprit is named
    # Structured diff is attached and its files section is empty (context-only).
    diff = excinfo.value.diff
    assert diff is not None
    assert diff["files"] == {}
    assert "dill" in diff["context_fold_affecting"]


def test_advisory_context_drift_warns_but_loads(tmp_path):
    """A fit-path-but-not-unpickle package bump (scipy) does NOT fail the load
    (a1.3) — it warns and is surfaced under ``context_advisory``."""
    real = compute_cache_version()
    faked_ctx = dict(real.context)
    faked_ctx["scipy"] = "0.0.0-fake"  # advisory: recorded, not folded
    stored = compute_cache_version(context=faked_ctx)
    assert stored.inputs_hash == real.inputs_hash  # scipy not in the hash

    write_cache_version(str(tmp_path), version=stored)

    with pytest.warns(UserWarning, match="fit-path environment drifted"):
        verify_cache_version(str(tmp_path))  # loads cleanly, only warns


# ---------------------------------------------------------------------------
# a3 — atomic, locked bundle build
# ---------------------------------------------------------------------------

def _boom_writer(*_a, **_k):
    """Stand-in for the real bundle body: write a partial artifact, then die."""
    bundle_dir = _a[1]
    os.makedirs(bundle_dir, exist_ok=True)
    with open(os.path.join(bundle_dir, "sim_data_cache.dill"), "wb") as f:
        f.write(b"partial-bytes")
    raise RuntimeError("mid-write failure")


def _no_build_temps(parent: Path) -> list[Path]:
    return [p for p in parent.iterdir() if p.name.startswith(".build-")]


def test_atomic_build_leaves_no_partial_dir_on_failure(tmp_path, monkeypatch):
    """A mid-write failure must leave neither the target nor a sibling temp
    build dir behind — a reader sees nothing, not a half-written bundle."""
    monkeypatch.setattr(core, "_write_sim_input_bundle_into", _boom_writer)
    parent = tmp_path / "out"
    bundle = parent / "cache"

    with pytest.raises(RuntimeError, match="mid-write failure"):
        core._write_sim_input_bundle(None, str(bundle))

    assert not bundle.exists()               # no partial target
    assert _no_build_temps(parent) == []     # no leftover temp build dir


def test_atomic_build_preserves_prior_bundle_on_failure(tmp_path, monkeypatch):
    """A failed rebuild-in-place must leave the previous good bundle intact and
    drop no ``.old-*`` rollback dir."""
    parent = tmp_path / "out"
    bundle = parent / "cache"
    bundle.mkdir(parents=True)
    (bundle / "marker.txt").write_text("GOOD")

    monkeypatch.setattr(core, "_write_sim_input_bundle_into", _boom_writer)
    with pytest.raises(RuntimeError, match="mid-write failure"):
        core._write_sim_input_bundle(None, str(bundle))

    assert (bundle / "marker.txt").read_text() == "GOOD"
    assert _no_build_temps(parent) == []
    assert [p for p in parent.iterdir() if ".old-" in p.name] == []


def test_atomic_build_restores_prior_when_final_rename_fails(tmp_path,
                                                             monkeypatch):
    """If the previous bundle was already moved aside but the final swap fails,
    the previous bundle is restored — the target is never left missing."""
    parent = tmp_path / "out"
    bundle = parent / "cache"
    bundle.mkdir(parents=True)
    (bundle / "marker.txt").write_text("GOOD")

    def ok_writer(_loader, bundle_dir, **_k):
        with open(os.path.join(bundle_dir, "marker.txt"), "w") as f:
            f.write("NEW")

    monkeypatch.setattr(core, "_write_sim_input_bundle_into", ok_writer)

    real_rename = os.rename

    def flaky_rename(src, dst):
        if os.path.basename(str(src)).startswith(".build-") \
                and os.path.abspath(str(dst)) == os.path.abspath(str(bundle)):
            raise OSError("final swap failed")
        return real_rename(src, dst)

    monkeypatch.setattr(core.os, "rename", flaky_rename)

    with pytest.raises(OSError, match="final swap failed"):
        core._write_sim_input_bundle(None, str(bundle))

    assert bundle.exists()
    assert (bundle / "marker.txt").read_text() == "GOOD"  # rolled back
    assert _no_build_temps(parent) == []
    assert [p for p in parent.iterdir() if ".old-" in p.name] == []


def test_atomic_build_success_swaps_and_cleans(tmp_path, monkeypatch):
    """On success the new bundle lands atomically and the ``.old-*`` rollback
    dir is cleaned up."""
    parent = tmp_path / "out"
    bundle = parent / "cache"
    bundle.mkdir(parents=True)
    (bundle / "marker.txt").write_text("OLD")

    def ok_writer(_loader, bundle_dir, **_k):
        with open(os.path.join(bundle_dir, "marker.txt"), "w") as f:
            f.write("NEW")

    monkeypatch.setattr(core, "_write_sim_input_bundle_into", ok_writer)
    core._write_sim_input_bundle(None, str(bundle))

    assert (bundle / "marker.txt").read_text() == "NEW"
    assert _no_build_temps(parent) == []
    assert [p for p in parent.iterdir() if ".old-" in p.name] == []


def test_atomic_build_rejects_concurrent_builder(tmp_path):
    """A second builder that finds the lock held fails fast with a clear
    message rather than interleaving writes into the same dir."""
    fcntl = getattr(core, "fcntl", None)
    if fcntl is None or not hasattr(fcntl, "flock"):
        pytest.skip("flock unavailable on this platform")

    parent = tmp_path / "out"
    bundle = parent / "cache"
    parent.mkdir(parents=True)
    lock_path = str(bundle) + ".lock"
    held = open(lock_path, "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="another process is building"):
            core._write_sim_input_bundle(None, str(bundle))
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()


# ---------------------------------------------------------------------------
# a3 — build_cache.py --if-stale
# ---------------------------------------------------------------------------

def test_if_stale_is_noop_when_cache_valid(tmp_path, monkeypatch):
    """``build_cache(..., if_stale=True)`` must skip the whole rebuild — never
    even load the fixture — when this process already sees a valid cache."""
    import scripts.build_cache as bc

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # A cache_version.json that verify_cache_version accepts for THIS process.
    write_cache_version(str(cache_dir))
    assert (cache_dir / CACHE_VERSION_FILENAME).exists()

    def _fail(*_a, **_k):
        raise AssertionError("fixture load must not run when --if-stale skips")

    monkeypatch.setattr(bc, "load_parca_state", _fail)
    monkeypatch.setattr(bc, "hydrate_sim_data_from_state", _fail)
    monkeypatch.setattr(bc, "save_sim_input", _fail)

    # os.chdir back afterwards — build_cache chdirs to repo_root.
    cwd = os.getcwd()
    try:
        bc.build_cache("models/parca/parca_state.pkl.gz", str(cache_dir),
                       if_stale=True)
    finally:
        os.chdir(cwd)


def test_if_stale_builds_when_no_cache(tmp_path, monkeypatch):
    """``--if-stale`` against an empty dir still builds (the fixture load is
    reached) — it only skips a *valid* cache."""
    import scripts.build_cache as bc

    reached = {"loaded": False}

    def _sentinel_load(_fixture):
        reached["loaded"] = True
        raise RuntimeError("stop after reaching the build path")

    monkeypatch.setattr(bc, "load_parca_state", _sentinel_load)

    cwd = os.getcwd()
    try:
        with pytest.raises(RuntimeError, match="stop after reaching"):
            bc.build_cache("models/parca/parca_state.pkl.gz",
                           str(tmp_path / "empty_cache"), if_stale=True)
    finally:
        os.chdir(cwd)
    assert reached["loaded"] is True
