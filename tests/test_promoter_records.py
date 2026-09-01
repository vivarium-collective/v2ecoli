"""Invariants for the promoter -> transcript table.

Phase 1 of the promoter/transcript split
(docs/promoter_transcript_split_scope.html). The table is additive —
nothing consumes it yet — so these tests are what keeps it honest until
initiation is keyed by promoter.
"""

import collections

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)
from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.getter_functions import (
    EXCLUDED_RNA_TYPES,
    GetterFunctions,
)


@pytest.fixture(scope="module")
def raw_data():
    return KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
    )


@pytest.fixture(scope="module")
def promoter_records():
    raw = KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
    )
    # Mirror the filter the real caller uses (getter_functions.py:216) —
    # genes need chromosomal positions and must not be pseudo/phantom.
    gene_rna_type = {rna["gene_id"]: rna["type"] for rna in raw.rnas}
    valid_gene_ids = {
        gene["id"]
        for gene in raw.genes
        if gene["left_end_pos"] is not None
        and gene["right_end_pos"] is not None
        and gene_rna_type[gene["id"]] not in EXCLUDED_RNA_TYPES
    }
    getter = GetterFunctions.__new__(GetterFunctions)
    getter._build_promoter_records(raw, valid_gene_ids)
    return getter.get_promoter_records()


def test_every_promoter_has_required_fields(promoter_records):
    assert promoter_records, "no promoters recorded"
    for record in promoter_records:
        assert record["id"]
        assert record["transcript_id"]
        assert isinstance(record["coordinate"], int)
        assert record["direction"] in ("+", "-")
        assert len(record["gene_tuple"]) > 0
        assert record["level"] in ("tu", "cistron")


def test_transcript_is_always_itself_a_promoter(promoter_records):
    """The canonical transcript is one of the TSSs, and maps to itself."""
    by_id = {r["id"]: r for r in promoter_records}
    for record in promoter_records:
        transcript = record["transcript_id"]
        assert transcript in by_id
        assert by_id[transcript]["transcript_id"] == transcript


def test_one_transcript_per_gene_tuple(promoter_records):
    """Promoters sharing a gene set must all drive the same transcript.

    This is the property the dedup loop enforces and the split relies on:
    a gene tuple names exactly one RNA species. It holds across both
    namespaces because a cistron-level record is only emitted for a gene
    no transcription unit covers, so the two can never collide.
    """
    gene_tuple_to_transcripts = collections.defaultdict(set)
    for record in promoter_records:
        gene_tuple_to_transcripts[record["gene_tuple"]].add(record["transcript_id"])
    offenders = {
        gene_tuple: transcripts
        for gene_tuple, transcripts in gene_tuple_to_transcripts.items()
        if len(transcripts) != 1
    }
    assert not offenders, f"gene tuples with multiple transcripts: {offenders}"


def test_cistron_only_transcripts_get_a_promoter(promoter_records):
    """Genes transcribed outside any TU are transcripts too, and need one.

    ``rna_data`` holds a bare cistron entry for every gene no transcription
    unit covers. Without a promoter record these would silently stop being
    transcribed once initiation is keyed by promoter. They are recognisable
    by driving themselves and carrying a single gene.
    """
    cistron_level = [r for r in promoter_records if r["level"] == "cistron"]
    assert len(cistron_level) > 100, (
        f"only {len(cistron_level)} cistron-level promoters; "
        "genes outside transcription units are losing their promoter"
    )
    for record in cistron_level:
        assert len(record["gene_tuple"]) == 1


def test_no_promoter_for_maturation_products(promoter_records, raw_data):
    """A covered gene must not get a cistron-level promoter.

    rna_data holds a cistron entry only for genes no TU covers. The
    sequence builder is laxer and also keeps an entry for a *covered*
    non-mRNA gene, but those are maturation products — aspV-tRNA, 6S-RNA
    and the like live in mature_rna_data, disjoint from rna_data, and are
    produced by rna_maturation from a TU precursor rather than
    transcribed. Emitting promoters for them would invent transcription
    that does not happen.
    """
    covered = set()
    for tu in raw_data.transcription_units:
        covered |= set(tu["genes"])
    offenders = [
        r for r in promoter_records
        if r["level"] == "cistron" and set(r["gene_tuple"]) & covered
    ]
    assert not offenders, (
        f"{len(offenders)} cistron promoters for covered genes, e.g. "
        f"{[r['id'] for r in offenders[:5]]}"
    )


def test_multi_promoter_operons_are_preserved(promoter_records):
    """Dedup drops duplicate TUs; this table must keep them as promoters.

    The impact assessment counted ~428 duplicate gene-tuple groups. If
    this collapses toward zero, promoter identity is being lost again.
    """
    per_transcript = collections.Counter(r["transcript_id"] for r in promoter_records)
    multi = [t for t, n in per_transcript.items() if n > 1]
    assert len(multi) > 300, (
        f"only {len(multi)} transcripts have multiple promoters; "
        "promoter identity looks lost"
    )
    assert len(promoter_records) > len(per_transcript)


@pytest.mark.parametrize(
    "promoter_id,expected_transcript",
    [
        # rpsU-dnaG-rpoD: rpsUp1/p2/p3 all drive one transcript.
        ("TU00352", "TU00352"),
        ("TU00434", "TU00352"),
        ("TU00435", "TU00352"),
        # ptsHI-crr, six promoters.
        ("TU00483", "TU0-45232"),
        ("TU0-45232", "TU0-45232"),
        # ftsZ.
        ("TU0-1425", "TU0-1423"),
    ],
)
def test_known_operon_mappings(promoter_records, promoter_id, expected_transcript):
    by_id = {r["id"]: r for r in promoter_records}
    assert by_id[promoter_id]["transcript_id"] == expected_transcript


def test_tss_is_the_strand_appropriate_end(promoter_records):
    """Coordinate is the 5' end: left on +, right on -.

    Guards the reverse-strand case, where taking left_end_pos would put
    the TSS at the 3' end of the transcript.
    """
    forward = [r for r in promoter_records if r["direction"] == "+"]
    reverse = [r for r in promoter_records if r["direction"] == "-"]
    assert forward and reverse

    by_id = {r["id"]: r for r in promoter_records}
    # gltX is on the reverse strand with three promoters; the canonical TU
    # is the one with the *smallest* TSS coordinate among them only if the
    # ordering were naive, so assert the real relationship instead.
    gltx = [r for r in promoter_records
            if by_id[r["transcript_id"]]["id"] == "TU0-6409"]
    if gltx:
        assert all(r["direction"] == "-" for r in gltx)
        # reverse-strand TSSs sit at the high-coordinate end of the span
        assert min(r["coordinate"] for r in gltx) > 2519208
