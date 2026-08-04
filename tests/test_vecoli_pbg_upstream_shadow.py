"""Regression: the ``vecoli`` composite must load pure-python vEcoli source
(``reconstruction``/``validation``) from the upstream checkout, not the
pre-imported copy in the installed ``vEcoli`` dist.

The bug this guards against: ``_ensure_upstream`` prepends the checkout to
``sys.path`` but only evicted cached ``ecoli``/``configs`` modules, leaving a
stale ``reconstruction`` (from the installed dist, pre-``2208460b``) cached.
Its ``two_component_system`` stoich matrix has 41 rows while the checkout's
sim_data reports 45 ``modified_molecules`` -> MonomerCounts crashes with
``operands could not be broadcast together with shapes (45,) (41,)``.

Hermetic: exercises the module-eviction contract directly (a full composite
build is environment-gated on ``$V2E_VECOLI_DIR`` -- see test_vecoli_composite).
"""
import sys

from v2ecoli.library.vecoli_pbg_upstream import (
    _SHADOW_SOURCE_PKGS, _purge_shadow_source_modules)


def test_shadow_pkg_set_covers_reconstruction_and_validation():
    # The two source trees added to fix the 45-vs-41 crash...
    assert "reconstruction" in _SHADOW_SOURCE_PKGS
    assert "validation" in _SHADOW_SOURCE_PKGS
    # ...plus the originals.
    assert "ecoli" in _SHADOW_SOURCE_PKGS
    assert "configs" in _SHADOW_SOURCE_PKGS
    # wholecell must NOT be purged: it stays the installed Cython-compiled tree.
    assert "wholecell" not in _SHADOW_SOURCE_PKGS


def test_purge_evicts_shadow_source_but_keeps_wholecell_and_unrelated():
    sentinel_recon = object()
    sentinel_recon_sub = object()
    sentinel_validation = object()
    sentinel_wholecell = object()
    sentinel_unrelated = object()
    added = {
        "reconstruction": sentinel_recon,
        "reconstruction.ecoli.dataclasses.process.two_component_system":
            sentinel_recon_sub,
        "validation.ecoli.validation_data": sentinel_validation,
        "wholecell.utils.filepath": sentinel_wholecell,
        "v2ecoli_unrelated_pkg.mod": sentinel_unrelated,
    }
    # Don't clobber anything real that happens to be imported already.
    added = {k: v for k, v in added.items() if k not in sys.modules}
    sys.modules.update(added)
    try:
        _purge_shadow_source_modules()

        # Every shadow-source module (exact name or dotted child) is gone.
        assert "reconstruction" not in sys.modules
        assert ("reconstruction.ecoli.dataclasses.process.two_component_system"
                not in sys.modules)
        assert "validation.ecoli.validation_data" not in sys.modules
        # wholecell + unrelated packages are untouched.
        if "wholecell.utils.filepath" in added:
            assert sys.modules.get("wholecell.utils.filepath") is sentinel_wholecell
        if "v2ecoli_unrelated_pkg.mod" in added:
            assert sys.modules.get("v2ecoli_unrelated_pkg.mod") is sentinel_unrelated
    finally:
        for k in list(added):
            sys.modules.pop(k, None)


def test_reconstruction_prefix_match_is_not_substring_greedy():
    """A package that merely starts with the same letters (e.g.
    ``reconstruction_helpers``) must NOT be evicted -- only ``reconstruction``
    and its dotted children."""
    decoy = object()
    if "reconstruction_helpers" in sys.modules:
        return  # don't disturb a real module
    sys.modules["reconstruction_helpers"] = decoy
    try:
        _purge_shadow_source_modules()
        assert sys.modules.get("reconstruction_helpers") is decoy
    finally:
        sys.modules.pop("reconstruction_helpers", None)
