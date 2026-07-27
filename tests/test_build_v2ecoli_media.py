"""Regression test: _build_v2ecoli must build non-basal conditions on the
condition's required media, not silently on the base cache's basal media.

Bug: ``_build_v2ecoli`` computed ``expected_media`` (the condition's required
nutrients) and asserted the built composite's ``environment.media_id`` against
it, but never actually PASSED ``media=expected_media`` to ``build_composite``.
For non-basal conditions (with_aa/succinate/no_oxygen/acetate) the per-
condition cache regen bakes only molecule counts, not the environment media,
so the build silently ran on the base cache's basal ``minimal`` media and the
fail-loud assertion then raised a RuntimeError.

Builds a real composite from ``out/cache_full`` (present in this worktree via
symlink), so this is a ``sim``-marked test (slower, cache-dependent) like
``tests/test_baseline_injected.py``.
"""
import os
import pickle

import pytest

pytestmark = pytest.mark.sim

CACHE_DIR = "out/cache_full"


def _cache_supports_with_aa(cache_dir: str) -> bool:
    """True iff ``cache_dir`` is a condition-complete ParCa cache that can
    build the ``with_aa`` condition.

    CI's ``out/cache_full`` (built by ``scripts/build_cache.py``, the
    ``--mode fast`` fixture) only carries the basal TF condition set, so
    ``with_aa``'s per-condition regen in ``_build_v2ecoli`` produces no
    bundle. Mirror the same lookup ``_build_v2ecoli`` does (condition's
    required media via ``sim_data.conditions``, checked against
    ``sim_data.external_state.saved_media``) to detect that up front,
    without triggering the RuntimeError.
    """
    sd_path = os.path.join(cache_dir, "simData.cPickle")
    if not os.path.exists(sd_path):
        return False
    with open(sd_path, "rb") as f:
        sim_data = pickle.load(f)
    expected_media = (sim_data.conditions.get("with_aa", {}) or {}).get("nutrients")
    if expected_media is None:
        return False
    saved_media = getattr(sim_data.external_state, "saved_media", {}) or {}
    return expected_media in saved_media


def test_with_aa_builds_on_correct_media():
    if not os.path.isdir(CACHE_DIR) or not _cache_supports_with_aa(CACHE_DIR):
        pytest.skip(
            "full-condition ParCa cache required (out/cache_full must support "
            "the with_aa condition); CI uses the --mode fast fixture, which "
            "omits non-basal conditions")
    from scripts.run_comparison_ensemble import _build_v2ecoli

    comp = _build_v2ecoli(0, "with_aa", CACHE_DIR, overrides=None)

    agents = comp.state.get("agents", {}) if hasattr(comp, "state") else {}
    ag = agents.get("0", comp.state)
    media_id = (ag.get("environment", {}) or {}).get("media_id")

    assert media_id == "minimal_plus_amino_acids", (
        f"expected v2ecoli with_aa build to run on 'minimal_plus_amino_acids', "
        f"got {media_id!r}"
    )
