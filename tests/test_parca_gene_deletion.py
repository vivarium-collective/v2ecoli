"""
Coordinate-handling tests for chromosome-level gene deletion in the vendored
ParCa fork (`KnowledgeBaseEcoli._delete_gene` and friends).

COORDINATE CONVENTION under test: left_end_pos / right_end_pos are 1-based and
INCLUSIVE, matching the EcoCyc-derived flat files -- a feature spanning [L, R]
occupies genome_sequence[L - 1 : R]. This is verifiable against the real data:
of 174 '+'-strand genes in genes.tsv, 157 start with ATG under [L - 1 : R] and
0 do under [L : R + 1].

The convention is the reason these tests exist. Feature LENGTHS stay correct
under an off-by-one splice while the SEQUENCE silently shifts, so any test that
only counts bases passes on a genome that is wrong.

Every case below sits in a branch that well-separated fixtures never reach.
"""

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)

# Deletion under test throughout: 1-based inclusive [500, 600], length 101.
DEL_LEFT = 500
DEL_RIGHT = 600
DEL_LEN = DEL_RIGHT - DEL_LEFT + 1


_DEFAULT = object()


def feature(id_, left, right, common_name=_DEFAULT):
    """`common_name` defaults to the id; pass None explicitly to test a null."""
    return {
        "id": id_,
        "left_end_pos": left,
        "right_end_pos": right,
        "common_name": id_ if common_name is _DEFAULT else common_name,
    }


def update(data, data_type="gene", del_gene_id="EG_TEST"):
    """Run the coordinate updater over `data` in place and return it."""
    kb = object.__new__(KnowledgeBaseEcoli)
    kb._update_global_coordinates_for_deletion(
        data, data_type, del_gene_id, DEL_LEFT, DEL_RIGHT
    )
    return data


def coords(row):
    return (row["left_end_pos"], row["right_end_pos"])


# --------------------------------------------------------------------------
# The case matrix: every relative position of a feature against a deletion.
# --------------------------------------------------------------------------

# (label, input coords, expected output coords or None when removed)
CASES = [
    # Entirely upstream -- untouched. Includes the boundary case of a feature
    # ending on the base immediately before the cut.
    ("before", (100, 200), (100, 200)),
    ("before_abutting", (400, DEL_LEFT - 1), (400, DEL_LEFT - 1)),
    # Entirely downstream -- shifts left by the deleted length. Includes the
    # boundary case of a feature starting on the base immediately after.
    ("after", (700, 800), (700 - DEL_LEN, 800 - DEL_LEN)),
    ("after_abutting", (DEL_RIGHT + 1, 700), (DEL_LEFT, 700 - DEL_LEN)),
    # Wholly inside the deletion -- removed along with it.
    ("contained", (520, 540), None),
    ("contained_exact", (DEL_LEFT, DEL_RIGHT), None),
    ("contained_at_left_edge", (DEL_LEFT, 540), None),
    ("contained_at_right_edge", (560, DEL_RIGHT), None),
    # Starts before, ends inside -- truncated at the cut. THIS IS THE CASE THE
    # left-edge guard silently skipped (returned 450-550 unchanged).
    ("overlaps_left", (450, 550), (450, DEL_LEFT - 1)),
    ("overlaps_left_minimal", (450, DEL_LEFT), (450, DEL_LEFT - 1)),
    ("overlaps_left_to_right_edge", (450, DEL_RIGHT), (450, DEL_LEFT - 1)),
    # Starts inside, ends after -- surviving 3' portion begins at the cut.
    ("overlaps_right", (550, 700), (DEL_LEFT, 700 - DEL_LEN)),
    ("overlaps_right_minimal", (DEL_RIGHT, 700), (DEL_LEFT, 700 - DEL_LEN)),
    ("overlaps_right_from_left_edge", (DEL_LEFT, 700), (DEL_LEFT, 700 - DEL_LEN)),
    # Starts before, ends after -- keeps both flanks, loses del_len.
    ("spans", (400, 700), (400, 700 - DEL_LEN)),
    ("spans_minimal", (DEL_LEFT - 1, DEL_RIGHT + 1), (DEL_LEFT - 1, DEL_LEFT)),
]


@pytest.mark.parametrize("data_type", ["gene", "tu", "dna_site"])
@pytest.mark.parametrize("label,given,expected", CASES, ids=[c[0] for c in CASES])
def test_coordinate_case_matrix(data_type, label, given, expected):
    """Every relative position, for every feature kind."""
    data = [feature(label, *given)]
    with pytest.warns(UserWarning) if (
        expected is None and data_type in ("gene", "dna_site")
    ) else _no_warning_context():
        update(data, data_type=data_type)

    if expected is None:
        assert data == [], f"{label} should have been removed with the deletion"
    else:
        assert len(data) == 1, f"{label} should have been retained"
        assert coords(data[0]) == expected


def _no_warning_context():
    """Null context so the parametrized test can branch on warning expectation."""
    import contextlib

    return contextlib.nullcontext()


def test_classification_is_total():
    """
    Every (left <= right) feature position classifies into exactly one case --
    no fallthrough. The upstream implementation's `else: print("this is a
    deletion case that has not been considered")` is what this replaces.
    """
    labels = set()
    for left in range(495, 606):
        for right in range(left, 610):
            case = KnowledgeBaseEcoli._classify_against_deletion(
                left, right, DEL_LEFT, DEL_RIGHT
            )
            assert case in {
                "before",
                "after",
                "contained",
                "spans",
                "overlaps_left",
                "overlaps_right",
            }
            labels.add(case)
    # The sweep must actually exercise every branch, or it proves nothing.
    assert labels == {
        "before",
        "after",
        "contained",
        "spans",
        "overlaps_left",
        "overlaps_right",
    }


# --------------------------------------------------------------------------
# List-mutation defects: removal must not perturb iteration.
# --------------------------------------------------------------------------


def test_adjacent_contained_features_are_both_removed():
    """
    Two features fully inside one deletion. Removing during iteration skips the
    element after each removed one, leaving the second in place.
    """
    data = [
        feature("after", 700, 800),
        feature("contained_a", 510, 520),
        feature("contained_b", 530, 540),
    ]
    with pytest.warns(UserWarning):
        update(data)

    assert [row["id"] for row in data] == ["after"]


def test_feature_after_a_removed_one_still_shifts():
    """
    The silent-corruption case: a SURVIVING downstream feature that follows a
    removed one in the list must still be shifted. Mutating during iteration
    skips it, leaving wrong coordinates with no error raised.
    """
    data = [
        feature("after1", 700, 800),
        feature("contained", 520, 540),
        feature("after2", 900, 1000),
    ]
    with pytest.warns(UserWarning):
        update(data)

    assert [row["id"] for row in data] == ["after1", "after2"]
    assert coords(data[0]) == (700 - DEL_LEN, 800 - DEL_LEN)
    assert coords(data[1]) == (900 - DEL_LEN, 1000 - DEL_LEN)


def test_contained_feature_as_first_row_does_not_raise():
    """
    A contained feature with no preceding row leaves the branch's coordinate
    variables unbound if the branch falls through instead of continuing.
    """
    data = [feature("contained", 520, 540)]
    with pytest.warns(UserWarning):
        update(data)

    assert data == []


# --------------------------------------------------------------------------
# Guards and annotation semantics.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [(None, None), ("", ""), (None, 700), (700, None), ("", 700), (700, "")],
)
def test_rows_without_usable_coordinates_are_skipped(left, right):
    """
    Half-populated rows must be skipped, not compared -- comparing None against
    an int raises TypeError.
    """
    data = [feature("no_coords", left, right)]
    update(data)

    assert coords(data[0]) == (left, right)


def test_only_content_losing_features_are_annotated():
    """
    The `_removed_<gene>` marker records that a feature LOST sequence. A
    feature that merely shifted downstream keeps its name -- otherwise a
    knockout tags every feature downstream of the cut, badly overstating how
    much of the chromosome it touched.
    """
    shifted = feature("tu_after", 700, 800, common_name="tu_after")
    truncated = feature("tu_overlap", 450, 550, common_name="tu_overlap")
    untouched = feature("tu_before", 100, 200, common_name="tu_before")
    data = [shifted, truncated, untouched]

    update(data, data_type="tu", del_gene_id="EG10526")

    assert shifted["common_name"] == "tu_after"
    assert untouched["common_name"] == "tu_before"
    assert truncated["common_name"] == "tu_overlap_removed_EG10526"


def test_null_common_name_annotates_cleanly():
    row = feature("tu_overlap", 450, 550, common_name=None)
    update([row], data_type="tu", del_gene_id="EG10526")

    assert row["common_name"] == "_removed_EG10526"


# --------------------------------------------------------------------------
# Sequence content: the splice must remove exactly the gene's own bases.
# --------------------------------------------------------------------------


def _kb(genome, genes, tus=None, dna_sites=None):
    """
    A KnowledgeBaseEcoli carrying only what the deletion path touches. Avoids
    standing up the full flat-file load for a coordinate test.
    """
    kb = object.__new__(KnowledgeBaseEcoli)
    kb.genome_sequence = genome
    kb.genes = genes
    kb.transcription_units = tus if tus is not None else []
    kb.dna_sites = dna_sites if dna_sites is not None else []
    return kb


def _gene_seq(genome, row):
    """A feature's sequence under the 1-based inclusive convention."""
    return genome[row["left_end_pos"] - 1 : row["right_end_pos"]]


def test_deletion_removes_exactly_the_genes_own_bases():
    """
    A gene at 1-based [11, 20] must remove genome[10:20] -- not genome[11:21].
    Length is preserved under either, so only content can tell them apart.
    """
    genome = "AAAAAAAAAA" "CCCCCCCCCC" "GGGGGGGGGG"  # 1-10 A, 11-20 C, 21-30 G
    kb = _kb(genome, [feature("target", 11, 20)])

    kb._delete_gene("target")

    assert kb.genome_sequence == "AAAAAAAAAA" "GGGGGGGGGG"
    assert "C" not in kb.genome_sequence


def test_adjacent_downstream_gene_keeps_its_sequence():
    """
    THE off-by-one detector. An immediately-adjacent downstream gene keeps its
    coordinates-minus-del_len; under a shifted splice its FIRST base becomes
    the deleted gene's first base -- a corrupted start codon, at the exact
    adjacency that operons make ubiquitous.
    """
    genome = "TTTTTTTTTT" "CCCCCCCCCC" "ATGGGCAATA"  # 1-10, 11-20, 21-30
    downstream = feature("downstream", 21, 30)
    kb = _kb(genome, [feature("target", 11, 20), downstream])
    original = _gene_seq(genome, downstream)
    assert original == "ATGGGCAATA"

    kb._delete_gene("target")

    assert coords(downstream) == (11, 20)
    assert _gene_seq(kb.genome_sequence, downstream) == original


def test_every_retained_gene_keeps_its_sequence():
    """
    The round-trip invariant, at small scale: after a deletion, every retained
    gene's sequence read from the modified genome is byte-identical to its
    sequence read from the original. This is the invariant the build-integrity
    study applies to the whole chromosome.
    """
    genome = (
        "ACGTACGTAC"  # 1-10   upstream
        "TTTTTTTTTT"  # 11-20  target (deleted)
        "GGCCTTAAGG"  # 21-30  adjacent downstream
        "CATGCATGCA"  # 31-40  far downstream
    )
    genes = [
        feature("upstream", 1, 10),
        feature("target", 11, 20),
        feature("adjacent", 21, 30),
        feature("far", 31, 40),
    ]
    kb = _kb(genome, genes)
    before = {g["id"]: _gene_seq(genome, g) for g in genes}

    kb._delete_gene("target")

    retained = [g for g in kb.genes if g["id"] != "target"]
    assert len(retained) == 3
    for gene in retained:
        assert _gene_seq(kb.genome_sequence, gene) == before[gene["id"]], (
            f"{gene['id']} sequence changed across the deletion"
        )


def test_deleted_gene_coordinates_are_nulled():
    kb = _kb("ACGT" * 10, [feature("target", 11, 20)])

    kb._delete_gene("target")

    assert coords(kb.genes[0]) == (None, None)
    assert len(kb.genome_sequence) == 40 - 10


def test_transcription_unit_loses_the_deleted_gene():
    """A multi-gene TU keeps its identity but drops the deleted member."""
    genome = "ACGT" * 25  # 100 bases
    tu = {
        "id": "tu1",
        "genes": ["target", "other"],
        "left_end_pos": 5,
        "right_end_pos": 40,
        "common_name": "tu1",
    }
    kb = _kb(
        genome,
        [feature("target", 11, 20), feature("other", 25, 40)],
        tus=[tu],
    )

    kb._delete_gene("target")

    assert tu["genes"] == ["other"]
    assert tu["common_name"] == "tu1_removed_target"
    # The TU spans the deletion, so it keeps both flanks and loses 10 bases.
    assert coords(tu) == (5, 30)


def test_solo_transcription_unit_is_nulled():
    """A TU carrying only the deleted gene has no meaningful coordinates."""
    genome = "ACGT" * 25
    tu = {
        "id": "tu1",
        "genes": ["target"],
        "left_end_pos": 11,
        "right_end_pos": 20,
        "common_name": "tu1",
    }
    kb = _kb(genome, [feature("target", 11, 20)], tus=[tu])

    kb._delete_gene("target")

    assert coords(tu) == (None, None)


def test_dna_sites_shift_with_the_deletion():
    """oriC/terC and other DNA sites are coordinate-bearing features too."""
    genome = "ACGT" * 250  # 1000 bases
    ori = feature("oriC", 50, 60, common_name="oriC")
    ter = feature("terC", 900, 910, common_name="TerC")
    kb = _kb(genome, [feature("target", 500, 600)], dna_sites=[ori, ter])

    kb._delete_gene("target")

    assert coords(ori) == (50, 60)
    assert coords(ter) == (900 - DEL_LEN, 910 - DEL_LEN)


def test_multiple_sequential_deletions():
    """
    Deletions compose: the second is expressed in coordinates the first already
    shifted, so each must leave the knowledge base self-consistent.
    """
    genome = (
        "AAAAAAAAAA"  # 1-10   keep_a
        "CCCCCCCCCC"  # 11-20  del_1
        "GGGGGGGGGG"  # 21-30  keep_b
        "TTTTTTTTTT"  # 31-40  del_2
    )
    keep_a = feature("keep_a", 1, 10)
    keep_b = feature("keep_b", 21, 30)
    kb = _kb(
        genome,
        [keep_a, feature("del_1", 11, 20), keep_b, feature("del_2", 31, 40)],
    )

    kb._delete_gene("del_1")
    kb._delete_gene("del_2")

    assert kb.genome_sequence == "AAAAAAAAAA" "GGGGGGGGGG"
    assert _gene_seq(kb.genome_sequence, keep_a) == "AAAAAAAAAA"
    assert _gene_seq(kb.genome_sequence, keep_b) == "GGGGGGGGGG"


def test_deleting_an_unknown_gene_raises():
    kb = _kb("ACGT" * 10, [feature("target", 11, 20)])

    with pytest.raises(AssertionError, match="no such gene"):
        kb._delete_gene("not_a_gene")


def test_deleting_a_gene_twice_raises():
    kb = _kb("ACGT" * 10, [feature("target", 11, 20)])
    kb._delete_gene("target")

    with pytest.raises(AssertionError, match="no coordinates"):
        kb._delete_gene("target")


# --------------------------------------------------------------------------
# Genome scale: the same invariant against the real EcoCyc knowledge base.
#
# Hand-built fixtures only cover the adjacencies you thought to write down. The
# real chromosome supplies ~12.8k coordinate-bearing features and the operon
# structure that makes overlapping and nested features ordinary. ~3 s per load.
# --------------------------------------------------------------------------

KB_FLAGS = {
    "operons_on": True,
    "remove_rrna_operons": False,
    "remove_rrff": False,
    "stable_rrna": False,
}
TARGET_GENE = "EG10526"  # lacY -- the gene the tranche-A knockout study uses
FEATURE_KINDS = ("genes", "transcription_units", "dna_sites")


def _coord_bearing(kb):
    for kind in FEATURE_KINDS:
        for row in getattr(kb, kind):
            if KnowledgeBaseEcoli._has_coordinates(row):
                yield kind, row


@pytest.fixture(scope="module")
def wild_type():
    return KnowledgeBaseEcoli(**KB_FLAGS)


@pytest.fixture(scope="module")
def knockout():
    return KnowledgeBaseEcoli(gene_deletions=[TARGET_GENE], **KB_FLAGS)


@pytest.fixture(scope="module")
def deleted_span(wild_type):
    target = next(g for g in wild_type.genes if g["id"] == TARGET_GENE)
    return target["left_end_pos"], target["right_end_pos"]


def test_real_genome_shortens_by_exactly_the_gene_length(
    wild_type, knockout, deleted_span
):
    left, right = deleted_span
    assert len(knockout.genome_sequence) == len(wild_type.genome_sequence) - (
        right - left + 1
    )


def test_real_deleted_gene_is_nulled(knockout):
    gene = next(g for g in knockout.genes if g["id"] == TARGET_GENE)
    assert (gene["left_end_pos"], gene["right_end_pos"]) == (None, None)


def test_real_genome_round_trip_invariant(wild_type, knockout, deleted_span):
    """
    THE genome-scale invariant, in two branches:

      * a feature whose length is unchanged kept all of its content, so its
        sequence must be byte-identical across the deletion;
      * a feature that shrank straddled the deletion, so its sequence must be
        its original with exactly the deleted span excised -- both flanks
        intact and correctly rejoined.

    An off-by-one splice preserves every LENGTH, so only the second branch can
    catch it. This is the invariant genotype-00-build-integrity applies as a
    study axis.
    """
    del_left, _ = deleted_span
    wt_genome = wild_type.genome_sequence
    before = {
        (kind, row["id"]): _gene_seq(wt_genome, row)
        for kind, row in _coord_bearing(wild_type)
    }

    intact, shrunk, corrupted = 0, 0, []
    for kind, row in _coord_bearing(knockout):
        if row["id"] == TARGET_GENE:
            continue
        key = (kind, row["id"])
        if key not in before:
            continue
        old_seq = before[key]
        new_seq = _gene_seq(knockout.genome_sequence, row)

        if len(new_seq) == len(old_seq):
            intact += 1
            if new_seq != old_seq:
                corrupted.append((kind, row["id"], "content changed"))
            continue

        shrunk += 1
        lost = len(old_seq) - len(new_seq)
        offset = max(del_left, row["left_end_pos"]) - row["left_end_pos"]
        expected = old_seq[:offset] + old_seq[offset + lost :]
        if new_seq != expected:
            corrupted.append((kind, row["id"], "excision misaligned"))

    assert not corrupted, f"{len(corrupted)} corrupted features: {corrupted[:10]}"
    # Guard against the assertions above passing vacuously.
    assert intact > 10000, f"expected the full feature set, saw {intact}"
    assert shrunk > 0, (
        "no feature straddled the deletion -- pick a gene inside an operon"
    )


def test_real_coordinates_stay_within_the_chromosome(knockout):
    genome_length = len(knockout.genome_sequence)
    out_of_range = [
        (kind, row["id"], row["left_end_pos"], row["right_end_pos"])
        for kind, row in _coord_bearing(knockout)
        if not 1 <= row["left_end_pos"] <= row["right_end_pos"] <= genome_length
    ]

    assert not out_of_range, (
        f"features outside [1, {genome_length}]: {out_of_range[:10]}"
    )
