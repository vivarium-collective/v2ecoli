"""Two-root resolution: v2ecoli consumed as an INSTALLED dependency.

When v2ecoli is a git dependency (e.g. sms-ecoli), its SOURCE files live
under the installed package but its DATA file
(``models/parca/parca_state.pkl.gz``) lives in the consuming WORKSPACE — no
single ``repo_root`` resolves both. ``candidate_repo_roots`` /
``compute_cache_version`` must resolve each INPUT_FILES entry against the
workspace root first, then the package root, so a data file that only
exists in the workspace still resolves. Hermetic: builds throwaway
shadow-workspace / shadow-package directories under ``tmp_path``, never
touches the real repo tree.
"""
from __future__ import annotations

import os

import pytest

from v2ecoli.library.cache_version import (
    INPUT_FILES,
    _default_repo_root,
    candidate_repo_roots,
    compute_cache_version,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_standalone_checkout_collapses_to_one_root():
    """In this repo (has both workspace.yaml and v2ecoli/ source), the
    workspace root and the package root are the same directory, so
    candidate_repo_roots collapses to a single entry — identical to the old
    single-root behavior."""
    roots = candidate_repo_roots()
    assert len(roots) == 1
    assert os.path.normpath(roots[0]) == os.path.normpath(_default_repo_root())


def test_installed_dependency_two_root_resolution(tmp_path, monkeypatch):
    """Simulate v2ecoli installed as a dependency: source lives under a
    package root, the ParCa data file lives under a separate workspace
    root that has NO v2ecoli source at all. compute_cache_version must
    still resolve every INPUT_FILES entry by trying the workspace root
    first and falling back to the package root.
    """
    workspace_root = tmp_path / "consuming_workspace"
    package_root = tmp_path / "site-packages-shadow"
    (workspace_root / "models" / "parca").mkdir(parents=True)
    (workspace_root / "models" / "parca" / "parca_state.pkl.gz").write_bytes(
        b"fake-parca-state")

    # Copy the real source files (content matters for the hash-stability
    # assertion elsewhere; here we just need them to exist) into the shadow
    # package root, everything except the data file.
    source_entries = [f for f in INPUT_FILES if f != "models/parca/parca_state.pkl.gz"]
    for rel in source_entries:
        src = os.path.join(REPO_ROOT, rel)
        dst = package_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(open(src, "rb").read())

    def fake_find_workspace_root():
        return str(workspace_root)

    monkeypatch.setattr(
        "viva_workspace.find_workspace_root", fake_find_workspace_root, raising=False)
    monkeypatch.setattr(
        "v2ecoli.library.cache_version._default_repo_root",
        lambda: str(package_root))

    version = compute_cache_version()

    assert set(version.per_file_hashes) == set(INPUT_FILES)
    # The data file resolved from the workspace root, not the package root
    # (it doesn't exist there at all) — assert its hash matches what we wrote.
    import hashlib
    expected = hashlib.sha256(b"fake-parca-state").hexdigest()
    assert version.per_file_hashes["models/parca/parca_state.pkl.gz"] == expected


def test_genuinely_missing_file_still_raises(tmp_path, monkeypatch):
    """Neither candidate root has the file -> FileNotFoundError, same
    fail-loud behavior as before (no silent MISSING sentinel)."""
    workspace_root = tmp_path / "empty_workspace"
    package_root = tmp_path / "empty_package"
    workspace_root.mkdir()
    package_root.mkdir()

    monkeypatch.setattr(
        "viva_workspace.find_workspace_root",
        lambda: str(workspace_root), raising=False)
    monkeypatch.setattr(
        "v2ecoli.library.cache_version._default_repo_root",
        lambda: str(package_root))

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        compute_cache_version(files=("does-not-exist.py",))


def test_inputs_hash_unchanged_by_candidate_root_resolution():
    """The per-entry candidate-root search must not change WHICH bytes get
    hashed for the standalone repo — same content in, same inputs_hash out.
    Cross-checked against explicit repo_root=REPO_ROOT (the pre-fix call
    shape, still single-root) to prove the new default-None path picks up
    byte-identical files."""
    via_candidate_roots = compute_cache_version()
    via_explicit_single_root = compute_cache_version(repo_root=REPO_ROOT)

    assert via_candidate_roots.per_file_hashes == via_explicit_single_root.per_file_hashes
    # context/build_params are probed identically in the same process, so
    # the full inputs_hash also matches.
    assert via_candidate_roots.inputs_hash == via_explicit_single_root.inputs_hash
