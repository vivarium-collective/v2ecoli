"""Task A7/A8/A9: build_params and context must be part of the fingerprint.

PARCA_REVIEW.md A7/A8/A9 documents that several real fit inputs were absent
from ``cache_version.CacheVersion``: the build parameters a bundle was made
with (condition, fixed_media, seed, n_seeds, patch/condition-manifest id)
and the runtime-environment package versions (scipy, numpy, numba, dill,
cvxpy, ecos, stochastic-arrow). Two distinct fits produced byte-identical
``cache_version.json``. These tests are hermetic — no ParCa cache or fit
required — and exercise ``compute_cache_version`` directly with explicit
``build_params``/``context`` overrides.
"""
from __future__ import annotations

from v2ecoli.library.cache_version import (
    CONTEXT_PACKAGES,
    compute_cache_version,
    probe_context,
)


def test_n_seeds_changes_fingerprint():
    """Two CacheVersions differing only in resolved n_seeds must diverge.

    This is the direct regression test for A8: V2PARCA_N_SEEDS was read once
    at module import and never recorded anywhere a cache fingerprint could
    see it, so fits with N_SEEDS=3 and N_SEEDS=10 produced byte-identical
    cache_version.json.
    """
    cv_3 = compute_cache_version(build_params={"n_seeds": 3})
    cv_10 = compute_cache_version(build_params={"n_seeds": 10})

    assert cv_3.inputs_hash != cv_10.inputs_hash
    assert cv_3.build_params["n_seeds"] == 3
    assert cv_10.build_params["n_seeds"] == 10

    # Identical build_params -> identical hash (same context, same files).
    cv_10_again = compute_cache_version(build_params={"n_seeds": 10})
    assert cv_10.inputs_hash == cv_10_again.inputs_hash


def test_condition_media_seed_change_fingerprint():
    """Changing condition / fixed_media / seed in build_params moves inputs_hash.

    Direct regression test for A7: ``out/cache`` (basal) and
    ``out/cache-stage1-heuristic`` (dnaA-patched) previously produced
    identical cache_version.json because none of these were fingerprinted.
    """
    basal = compute_cache_version(
        build_params={"condition": None, "fixed_media": None, "seed": 0})
    acetate = compute_cache_version(
        build_params={"condition": "acetate", "fixed_media": "minimal_acetate",
                      "seed": 0})
    diff_seed = compute_cache_version(
        build_params={"condition": None, "fixed_media": None, "seed": 7})

    assert basal.inputs_hash != acetate.inputs_hash
    assert basal.inputs_hash != diff_seed.inputs_hash
    assert acetate.inputs_hash != diff_seed.inputs_hash

    # Identical build_params + identical inputs -> identical hash.
    basal_again = compute_cache_version(
        build_params={"condition": None, "fixed_media": None, "seed": 0})
    assert basal.inputs_hash == basal_again.inputs_hash
    assert basal.per_file_hashes == basal_again.per_file_hashes

    # A patch/condition-manifest id also moves the fingerprint.
    patched = compute_cache_version(
        build_params={"condition": "acetate", "fixed_media": "minimal_acetate",
                      "seed": 0, "condition_manifest_hash": "deadbeef" * 8})
    assert patched.inputs_hash != acetate.inputs_hash


def test_dependency_version_in_context_changes_fingerprint():
    """A changed recorded dependency version moves inputs_hash; identical
    env -> identical hash.

    Direct regression test for A9: a cache built under one scipy/numpy/etc
    and loaded under another previously passed verification unchanged.
    """
    real_context = probe_context()

    upgraded_context = dict(real_context)
    upgraded_context["scipy"] = "999.999.999"

    cv_before = compute_cache_version(context=real_context)
    cv_after = compute_cache_version(context=upgraded_context)

    assert cv_before.inputs_hash != cv_after.inputs_hash

    cv_before_again = compute_cache_version(context=real_context)
    assert cv_before.inputs_hash == cv_before_again.inputs_hash


def test_context_block_records_expected_packages():
    """context carries all 8 packages, each a version string or "absent"."""
    expected = {"python", *CONTEXT_PACKAGES}
    assert expected == {
        "python", "scipy", "numpy", "numba", "dill", "cvxpy", "ecos",
        "stochastic-arrow",
    }

    cv = compute_cache_version()

    assert set(cv.context.keys()) == expected
    for key, value in cv.context.items():
        assert isinstance(value, str) and value, (
            f"context[{key!r}] must be a non-empty string (version or "
            f"'absent' sentinel), got {value!r}")


def test_probe_context_never_raises_on_missing_package(monkeypatch):
    """A package that fails to import/report a version records 'absent',
    not a crash — A9's 'gracefully handle a package not being importable'."""
    import importlib.metadata as metadata

    real_version = metadata.version

    def _boom(name):
        if name == "scipy":
            raise ModuleNotFoundError("no scipy here")
        return real_version(name)

    monkeypatch.setattr(metadata, "version", _boom)

    ctx = probe_context()
    assert ctx["scipy"] == "absent"


def test_build_params_default_is_none_filled():
    """No build_params supplied -> a plain basal build (all None), matching
    a two-call default-vs-default comparison used by verify_cache_version's
    parameterless recompute."""
    cv = compute_cache_version()
    assert cv.build_params == {
        "condition": None,
        "fixed_media": None,
        "seed": None,
        "n_seeds": None,
        "condition_manifest_hash": None,
    }


def test_to_dict_from_dict_roundtrip_preserves_context_and_build_params():
    cv = compute_cache_version(build_params={"condition": "acetate", "seed": 3})
    from v2ecoli.library.cache_version import CacheVersion

    roundtripped = CacheVersion.from_dict(cv.to_dict())
    assert roundtripped == cv
