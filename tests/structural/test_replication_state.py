"""Fast unit tests for the live-state replication/RNAP helpers in
v2ecoli.structural.build: chromosome_state_from_live, rnaps_from_live, and
classify_domains (ported from feat/3d-transcription-translation).

These feed small synthetic structured arrays directly to the helpers — no
sim, no ParCa cache, no parsimony binary required.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.structural import build as B


# ── classify_domains ────────────────────────────────────────────────────────

@pytest.mark.fast
def test_classify_domains_single_chromosome_root_is_not_daughter():
    """A domain equal to the (sole) chromosome's root -> chromosome 0, not a daughter."""
    chrom_idx, is_daughter = B.classify_domains(
        domain_children={}, full_chromosome_domains=[0],
        query_domains=np.array([0, 0], dtype="i4"))
    assert list(chrom_idx) == [0, 0]
    assert list(is_daughter) == [False, False]


@pytest.mark.fast
def test_classify_domains_child_domain_is_daughter():
    """A domain that's a transitive CHILD of the chromosome root -> same
    chromosome index, is_daughter=True."""
    domain_children = {0: [1, 2], 2: [3]}
    chrom_idx, is_daughter = B.classify_domains(
        domain_children, full_chromosome_domains=[0],
        query_domains=np.array([0, 1, 3], dtype="i4"))
    assert list(chrom_idx) == [0, 0, 0]
    assert list(is_daughter) == [False, True, True]   # 1 is a direct child; 3 is a grandchild


@pytest.mark.fast
def test_classify_domains_routes_by_chromosome_and_flags_daughters():
    """Two chromosomes (roots 10, 20); domain 11 is a child of root 10 (daughter
    copy of chromosome 0), domain 20 is root of chromosome 1 (not a daughter)."""
    domain_children = {10: [11]}
    chrom_idx, is_daughter = B.classify_domains(
        domain_children, full_chromosome_domains=[10, 20],
        query_domains=np.array([10, 11, 20], dtype="i4"))
    assert list(chrom_idx) == [0, 0, 1]
    assert list(is_daughter) == [False, True, False]


@pytest.mark.fast
def test_classify_domains_unmatched_defaults_to_zero_not_daughter():
    """A domain with no matching lineage falls back to chromosome_index=0,
    is_daughter=False (never fabricated as True)."""
    chrom_idx, is_daughter = B.classify_domains(
        domain_children={}, full_chromosome_domains=[0],
        query_domains=np.array([999], dtype="i4"))
    assert list(chrom_idx) == [0]
    assert list(is_daughter) == [False]


# ── chromosome_state_from_live ──────────────────────────────────────────────

def _unique_array(rows, fields):
    """Build a structured numpy array with `_entryState` + the given fields."""
    dtype = [("_entryState", "i1")] + fields
    return np.array(rows, dtype=dtype)


@pytest.mark.fast
def test_chromosome_state_counts_only_active_full_chromosome_rows():
    """n_chromosomes = the ACTIVE (_entryState=1) row count, not the raw
    (padded) array length — unique-molecule stores are pre-allocated with
    spare inactive capacity."""
    fc = _unique_array(
        [(1, 0), (1, 5), (0, 0), (0, 0)],   # 2 active, 2 inactive padding rows
        [("domain_index", "i4")])
    n_chromosomes, fork_fraction = B.chromosome_state_from_live(fc)
    assert n_chromosomes == 2
    assert fork_fraction == 0.0   # no active_replisome supplied -> unreplicated default


@pytest.mark.fast
def test_chromosome_state_no_entrystate_field_treats_all_rows_active():
    """Synthetic arrays without an _entryState field (as a unit test would
    build) are treated as all-active — lets tests hand in bare arrays."""
    fc = np.array([(0,), (0,), (0,)], dtype=[("domain_index", "i4")])
    n_chromosomes, _ = B.chromosome_state_from_live(fc)
    assert n_chromosomes == 3


@pytest.mark.fast
def test_chromosome_state_fork_fraction_from_active_replisome_coordinates():
    """fork_fraction = mean(|active replisome coordinates|) / REPLICHORE_BP —
    matching capture_structural_snapshot.py's live-state derivation."""
    fc = _unique_array([(1, 0)], [("domain_index", "i4")])
    rep = _unique_array(
        [(1, 500_000), (1, -300_000), (0, 999_999)],  # last row inactive -> excluded
        [("coordinates", "i8")])
    n_chromosomes, fork_fraction = B.chromosome_state_from_live(fc, rep)
    assert n_chromosomes == 1
    expected = ((500_000 + 300_000) / 2) / B.REPLICHORE_BP
    assert fork_fraction == pytest.approx(expected)


@pytest.mark.fast
def test_chromosome_state_empty_active_replisome_falls_back_to_zero():
    """No active forks (all replisomes inactive, or none present) -> 0.0,
    not fabricated."""
    fc = _unique_array([(1, 0)], [("domain_index", "i4")])
    rep = _unique_array([(0, 123456)], [("coordinates", "i8")])  # inactive only
    _, fork_fraction = B.chromosome_state_from_live(fc, rep)
    assert fork_fraction == 0.0


# ── rnaps_from_live ──────────────────────────────────────────────────────────

@pytest.mark.fast
def test_rnaps_from_live_empty_array_returns_empty_list():
    assert B.rnaps_from_live(None) == []
    empty = _unique_array([], [("domain_index", "i4"), ("coordinates", "i8"), ("is_forward", "?")])
    assert B.rnaps_from_live(empty) == []


@pytest.mark.fast
def test_rnaps_from_live_basic_fields_without_chromosome_data():
    """Without full_chromosome/chromosome_domain, every RNAP defaults to
    chromosome_index=0, is_daughter=False (not fabricated)."""
    rnap = _unique_array(
        [(1, 0, 100_000, True), (1, 0, -50_000, False)],
        [("domain_index", "i4"), ("coordinates", "i8"), ("is_forward", "?")])
    rnaps = B.rnaps_from_live(rnap)
    assert rnaps == [
        {"coordinates": 100_000, "domain_index": 0, "is_forward": True,
         "chromosome_index": 0, "is_daughter": False},
        {"coordinates": -50_000, "domain_index": 0, "is_forward": False,
         "chromosome_index": 0, "is_daughter": False},
    ]


@pytest.mark.fast
def test_rnaps_from_live_filters_inactive_rows():
    rnap = _unique_array(
        [(1, 0, 100_000, True), (0, 0, -999, True)],  # 2nd row inactive
        [("domain_index", "i4"), ("coordinates", "i8"), ("is_forward", "?")])
    rnaps = B.rnaps_from_live(rnap)
    assert len(rnaps) == 1
    assert rnaps[0]["coordinates"] == 100_000


@pytest.mark.fast
def test_rnaps_from_live_classifies_chromosome_and_daughter_via_domain_tree():
    """One RNAP on the root domain of chromosome 0 (not a daughter), one RNAP
    on a child domain (a daughter copy) -> classify_domains routes both
    correctly through rnaps_from_live end-to-end."""
    rnap = _unique_array(
        [(1, 0, 10_000, True), (1, 1, 20_000, True)],
        [("domain_index", "i4"), ("coordinates", "i8"), ("is_forward", "?")])
    full_chromosome = _unique_array([(1, 0)], [("domain_index", "i4")])
    chromosome_domain = _unique_array(
        [(1, 0, [1, -1])],
        [("domain_index", "i4"), ("child_domains", "i4", (2,))])

    rnaps = B.rnaps_from_live(rnap, full_chromosome, chromosome_domain)

    by_coord = {r["coordinates"]: r for r in rnaps}
    assert by_coord[10_000]["chromosome_index"] == 0
    assert by_coord[10_000]["is_daughter"] is False
    assert by_coord[20_000]["chromosome_index"] == 0
    assert by_coord[20_000]["is_daughter"] is True


@pytest.mark.fast
def test_rnaps_from_live_missing_is_forward_defaults_true():
    """A schema without an is_forward field (shouldn't happen live, but
    defensively handled) defaults every RNAP to forward-strand."""
    rnap = _unique_array([(1, 0, 100_000)], [("domain_index", "i4"), ("coordinates", "i8")])
    rnaps = B.rnaps_from_live(rnap)
    assert rnaps[0]["is_forward"] is True


# ── pack_from_state wiring ──────────────────────────────────────────────────

@pytest.mark.fast
def test_pack_from_state_threads_replication_state_into_chromosome(monkeypatch, tmp_path):
    """rnaps / n_chromosomes / fork_fraction reach the Chromosome recipe object
    build_pack receives, and supercoil stays None (diffuse nucleoid preserved)."""
    captured = {}

    def fake_select_ingredients(counts, locations=None, *, top_n=40):
        return []

    def fake_build_pack(ingredients, capsule, chromosome, *, out_dir, name, scale,
                        proxy_lod, envelope=None):
        captured["chromosome"] = chromosome
        return {"placements": []}

    monkeypatch.setattr(B, "select_ingredients", fake_select_ingredients)
    monkeypatch.setattr(B, "build_pack", fake_build_pack)

    rnaps = [{"coordinates": 42, "domain_index": 0, "is_forward": True,
             "chromosome_index": 0, "is_daughter": False}]
    B.pack_from_state(str(tmp_path), "t", {}, volume_fl=1.0,
                      rnaps=rnaps, n_chromosomes=2, fork_fraction=0.37)

    chrom = captured["chromosome"]
    assert chrom.supercoil is None                 # diffuse nucleoid preserved
    assert chrom.n_chromosomes == 2
    assert chrom.fork_fraction == 0.37
    assert chrom.rnaps == rnaps
    assert chrom.rnap_marker == "rna_polymerase"    # matches the CURATED ingredient id
    assert chrom.fork_marker == "replisome"
    assert chrom.oric_marker == "oriC"
    assert chrom.ter_marker == "terminus"


@pytest.mark.fast
def test_pack_from_state_defaults_are_backward_compatible(monkeypatch, tmp_path):
    """Callers that don't pass rnaps/n_chromosomes/fork_fraction (e.g. the
    existing build_model() npz path) still get the old unreplicated,
    RNAP-free behaviour."""
    captured = {}

    def fake_build_pack(ingredients, capsule, chromosome, *, out_dir, name, scale,
                        proxy_lod, envelope=None):
        captured["chromosome"] = chromosome
        return {"placements": []}

    monkeypatch.setattr(B, "select_ingredients", lambda *a, **k: [])
    monkeypatch.setattr(B, "build_pack", fake_build_pack)

    B.pack_from_state(str(tmp_path), "t", {}, volume_fl=1.0)

    chrom = captured["chromosome"]
    assert chrom.n_chromosomes == 1
    assert chrom.fork_fraction == 0.0
    assert chrom.rnaps == []
