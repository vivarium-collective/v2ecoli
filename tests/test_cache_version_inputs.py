"""Guards on the ParCa cache fingerprint's own inputs — not the cache itself.

Companion to ``tests/test_cache_version.py`` (which is ``pytest.mark.sim``
and needs a built ``sim_data_cache`` fixture). Everything here is hermetic:
no ParCa cache required, so it runs under the default ``-m "not sim"`` CI
lane too.

Regression this guards: the composite family was renamed to the ``ecoli_*``
scheme in 645fe178, but ``cache_version.INPUT_FILES`` kept the old names.
Five of eleven entries silently pointed at files that no longer existed;
``compute_cache_version`` hashed each to a constant ``"MISSING"`` sentinel
instead of raising, so editing the real composites stopped moving
``inputs_hash`` at all — the exact "fingerprint collapses to a constant"
hazard the module's own docstring warns about. Separately, the CI
``hashFiles(...)`` key list drifted from ``INPUT_FILES`` and only
*accidentally* agreed because both sides named dead files.
"""
from __future__ import annotations

import os
import re
import shutil

import pytest

from v2ecoli.library.cache_version import INPUT_FILES, compute_cache_version


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CI_YML = os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml")


def test_input_files_all_exist():
    """Every path cache_version fingerprints must actually be on disk.

    A dead path doesn't error — it used to hash to a stable "MISSING"
    sentinel — so this is the only thing standing between a rename and a
    fingerprint that silently stops covering that file forever.
    """
    missing = [rel for rel in INPUT_FILES
               if not os.path.exists(os.path.join(REPO_ROOT, rel))]
    assert not missing, (
        f"INPUT_FILES entries do not exist on disk: {missing}. "
        f"A composite/module was renamed or deleted without updating "
        f"v2ecoli/library/cache_version.py:INPUT_FILES."
    )


def _parse_ci_hashfiles_list(ci_yml_text: str) -> set[str]:
    """Extract the quoted path list from the behavior-tests hashFiles(...) call.

    The block looks like:
        key: >-
          parca-cache-v4-${{ hashFiles(
            'path/one',
            'path/two'
          ) }}
    There is exactly one hashFiles(...) call in ci.yml (the ParCa cache
    restore key); grab everything between its parens and pull out the
    single-quoted strings.
    """
    match = re.search(r"hashFiles\((.*?)\)", ci_yml_text, re.DOTALL)
    assert match is not None, "no hashFiles(...) call found in ci.yml"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_ci_key_matches_input_files():
    """CI's cache-busting key must fingerprint exactly what the code does.

    If these two lists diverge, the failure mode is silent: create a file
    at one of the mismatched names and the in-code hash moves while the CI
    key doesn't (or vice versa) -> actions/cache reports a hit -> the
    "Build ParCa cache on miss" step is skipped -> every sim test fails at
    conftest.py's verify_cache_version with StaleCacheError.
    """
    with open(CI_YML) as f:
        ci_text = f.read()
    ci_paths = _parse_ci_hashfiles_list(ci_text)
    assert ci_paths == set(INPUT_FILES), (
        f"ci.yml hashFiles(...) list disagrees with INPUT_FILES.\n"
        f"Only in ci.yml:      {sorted(ci_paths - set(INPUT_FILES))}\n"
        f"Only in INPUT_FILES: {sorted(set(INPUT_FILES) - ci_paths)}"
    )


def test_missing_input_file_raises():
    """A nonexistent INPUT_FILES entry must raise, never hash to a sentinel.

    Encoding "MISSING" as a stable string is how the fingerprint quietly
    stopped covering 5/11 files after the ecoli_* rename: the file could be
    edited freely and inputs_hash never moved. A vanished fingerprint input
    is a bug, not a state.
    """
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        compute_cache_version(
            repo_root=REPO_ROOT,
            files=("v2ecoli/library/sim_data.py", "does-not-exist.py"),
        )


def test_editing_a_composite_moves_the_fingerprint(tmp_path):
    """Editing a real composite's bytes must move inputs_hash.

    Hermetic: copies one real INPUT_FILES entry into a throwaway repo_root
    under tmp_path and mutates the copy, never the checked-out repo. This
    is the actual property A1 restores — before the fix, this file's
    dead-name sibling (e.g. the old baseline_millard.py) could be edited
    all day without moving inputs_hash at all.
    """
    target_rel = "v2ecoli/composites/ecoli_baseline.py"
    assert target_rel in INPUT_FILES
    original = os.path.join(REPO_ROOT, target_rel)
    assert os.path.exists(original)

    shadow_root = tmp_path / "shadow_repo"
    shadow_path = shadow_root / target_rel
    shadow_path.parent.mkdir(parents=True)
    shutil.copy2(original, shadow_path)

    before = compute_cache_version(repo_root=str(shadow_root), files=(target_rel,))

    with open(shadow_path, "ab") as f:
        f.write(b"\n# fingerprint-test mutation\n")

    after = compute_cache_version(repo_root=str(shadow_root), files=(target_rel,))

    assert before.inputs_hash != after.inputs_hash
    assert before.per_file_hashes[target_rel] != after.per_file_hashes[target_rel]
