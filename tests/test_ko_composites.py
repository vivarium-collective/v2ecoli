"""Tests for the translation-level knockout composites (KO_baseline,
KO_batch_baseline) and their shared perturbation machinery.

The resolver + override builder run against the real ParCa cache (gated on it
being present); the composite-level tests build documents (cheap, cache-only)
and assert the knockout reaches the live process, without running a simulation.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.perturbations import (
    UnknownPerturbationTarget,
    resolve_targets,
    translation_efficiency_override,
)

CACHE = "out/cache"
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE),
    reason="ParCa cache (out/cache) required for KO resolution")

# A known coding gene and its monomer id (verified against the cache).
LACY_GENE = "EG10526"
LACY_MONOMER = "LACY-MONOMER[m]"
TPI_GENE = "EG11015"

_PI_KEY = "ecoli-polypeptide-initiation.translation_efficiencies"


@pytest.fixture(scope="module")
def bundle():
    return load_cache_bundle(CACHE)


# --- resolver ----------------------------------------------------------------

@_needs_cache
def test_resolve_gene_and_monomer_id_agree(bundle):
    """A gene id and its monomer id resolve to the SAME monomer index."""
    r = resolve_targets(bundle, [LACY_GENE, LACY_MONOMER])
    assert r[LACY_GENE] == r[LACY_MONOMER]


@_needs_cache
def test_resolve_bare_monomer_name(bundle):
    """The compartment suffix is optional when it is unambiguous."""
    full = resolve_targets(bundle, [LACY_MONOMER])[LACY_MONOMER]
    bare = LACY_MONOMER.split("[", 1)[0]
    assert resolve_targets(bundle, [bare])[bare] == full


@_needs_cache
def test_resolve_unknown_target_raises(bundle):
    with pytest.raises(UnknownPerturbationTarget, match="unknown targets"):
        resolve_targets(bundle, ["NOT_A_GENE_999"])


# --- override builder --------------------------------------------------------

@_needs_cache
def test_override_zeros_only_the_targets(bundle):
    before = np.asarray(
        bundle["configs"]["ecoli-polypeptide-initiation"]["translation_efficiencies"])
    ov = translation_efficiency_override(bundle, [LACY_GENE, TPI_GENE])
    assert set(ov) == {_PI_KEY}
    after = np.asarray(ov[_PI_KEY])
    idx = list(resolve_targets(bundle, [LACY_GENE, TPI_GENE]).values())
    assert np.all(after[idx] == 0.0)
    # Every other efficiency is untouched, and the original array is not mutated.
    assert np.allclose(np.delete(after, idx), np.delete(before, idx))
    assert before[idx[0]] != 0.0  # sanity: it was nonzero to begin with


@_needs_cache
def test_override_knockdown_multiplier(bundle):
    before = np.asarray(
        bundle["configs"]["ecoli-polypeptide-initiation"]["translation_efficiencies"])
    ov = translation_efficiency_override(bundle, {LACY_GENE: 0.5})
    after = np.asarray(ov[_PI_KEY])
    i = resolve_targets(bundle, [LACY_GENE])[LACY_GENE]
    assert after[i] == pytest.approx(before[i] * 0.5)


def test_override_empty_is_a_noop():
    # No cache access at all when there is nothing to perturb.
    assert translation_efficiency_override({"configs": {}}, []) == {}
    assert translation_efficiency_override({"configs": {}}, {}) == {}


@_needs_cache
def test_override_rejects_negative_multiplier(bundle):
    with pytest.raises(ValueError, match="non-negative"):
        translation_efficiency_override(bundle, {LACY_GENE: -1.0})


# --- KO_baseline composite ---------------------------------------------------

@_needs_cache
def test_ko_baseline_zeroes_the_process_translation_efficiency():
    """The knockout must reach the built PolypeptideInitiation process — not
    just the config dict — so no ribosome initiates on that mRNA."""
    from process_bigraph import Composite
    from v2ecoli.composites.ko_baseline import KO_baseline

    core = build_core()
    doc = KO_baseline(core=core, knockouts=[LACY_GENE])
    comp = Composite({"state": doc["state"]}, core=core)
    node = comp.state["agents"]["0"]["ecoli-polypeptide-initiation"]
    params = node["instance"].parameters
    te = np.asarray(params["translation_efficiencies"])
    mids = [str(x) for x in params["monomer_ids"]]
    assert te[mids.index(LACY_MONOMER)] == 0.0
    assert int((te == 0.0).sum()) == 1  # exactly the one KO, nothing else


@_needs_cache
def test_ko_baseline_empty_knockouts_is_plain_baseline():
    """No knockouts → byte-identical translation efficiencies to baseline."""
    from process_bigraph import Composite
    from v2ecoli.composites.ko_baseline import KO_baseline
    from v2ecoli.composites.baseline import baseline

    core = build_core()

    def _te(doc):
        comp = Composite({"state": doc["state"]}, core=core)
        p = comp.state["agents"]["0"]["ecoli-polypeptide-initiation"]["instance"].parameters
        return np.asarray(p["translation_efficiencies"])

    assert np.array_equal(_te(KO_baseline(core=core, knockouts=[])),
                          _te(baseline(core=core)))


@_needs_cache
def test_ko_baseline_unknown_gene_fails_at_build():
    from v2ecoli.composites.ko_baseline import KO_baseline
    with pytest.raises(UnknownPerturbationTarget):
        KO_baseline(core=build_core(), knockouts=["NOPE_123"])


# --- KO_batch_baseline composite ---------------------------------------------

@_needs_cache
def test_ko_batch_baseline_document_carries_the_panel_wide_override():
    """The doc is cheap to build (no ParCa run) and the runner config carries the
    knockout as a base_config_override applied to every branch."""
    from v2ecoli.composites.ko_batch_baseline import KO_batch_baseline

    doc = KO_batch_baseline(core=build_core(), n_seeds=3, knockouts=[LACY_GENE])
    rc = doc["state"]["batch_runner"]["config"]
    assert rc["n_seeds"] == 3
    bov = rc["base_config_overrides"]
    assert set(bov) == {_PI_KEY}
    bundle = load_cache_bundle(CACHE)
    i = resolve_targets(bundle, [LACY_GENE])[LACY_GENE]
    assert np.asarray(bov[_PI_KEY])[i] == 0.0


@_needs_cache
def test_ko_batch_baseline_unknown_gene_fails_at_build():
    """Resolution happens at document-build time, not inside a Ray worker."""
    from v2ecoli.composites.ko_batch_baseline import KO_batch_baseline
    with pytest.raises(UnknownPerturbationTarget):
        KO_batch_baseline(core=build_core(), knockouts=["NOPE_123"])


def test_both_ko_composites_registered():
    from pbg_superpowers.composite_generator import _REGISTRY
    import v2ecoli.composites  # noqa: F401 — fires the decorators
    names = {e.name for e in _REGISTRY.values() if hasattr(e, "name")}
    assert {"KO_baseline", "KO_batch_baseline"} <= names


# --- the base_config_overrides workflow hook ---------------------------------

def test_base_config_overrides_apply_to_every_branch():
    """expand_branches must merge base_config_overrides into EVERY branch — the
    mechanism that makes a KO panel-wide rather than one variant arm."""
    from v2ecoli.workflow.variants import expand_branches

    branches = expand_branches({
        "n_init_sims": 3,
        "base_config_overrides": {"p.k": 0.0},
    })
    assert len(branches) == 3
    assert all(b.overrides.get("p.k") == 0.0 for b in branches)
    # A base override is panel-wide, so it must NOT masquerade as a per-branch
    # variant marker in the branch metadata.
    assert all("override:p.k" not in b.metadata for b in branches)


def test_branch_variant_override_wins_over_base_on_key_clash():
    from v2ecoli.workflow.variants import expand_branches
    branches = expand_branches({
        "n_init_sims": 1,
        "base_config_overrides": {"p.k": 0.0},
        "variants": {"hi": {"target": "p.k", "value": [9.0]}},
    })
    # branches: baseline (base only) + the variant arm (variant wins).
    by_variant = {b.variant_name: b.overrides.get("p.k") for b in branches}
    assert by_variant["baseline"] == 0.0
    assert by_variant["hi"] == 9.0
