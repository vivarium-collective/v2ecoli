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

import pytest

pytestmark = pytest.mark.sim

CACHE_DIR = "out/cache_full"


def test_with_aa_builds_on_correct_media():
    from scripts.run_comparison_ensemble import _build_v2ecoli

    comp = _build_v2ecoli(0, "with_aa", CACHE_DIR, overrides=None)

    agents = comp.state.get("agents", {}) if hasattr(comp, "state") else {}
    ag = agents.get("0", comp.state)
    media_id = (ag.get("environment", {}) or {}).get("media_id")

    assert media_id == "minimal_plus_amino_acids", (
        f"expected v2ecoli with_aa build to run on 'minimal_plus_amino_acids', "
        f"got {media_id!r}"
    )
