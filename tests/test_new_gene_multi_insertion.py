"""Multiple noncontiguous new-gene insertions: discovery, and splice order.

A new-gene payload is either ONE contiguous cassette (files directly under
``new_gene_data/<option>/``) or SEVERAL noncontiguous cassettes, each in its own
subdirectory with its own ``insertion_location.tsv``. This module covers the
second shape.

⭐ THE COORDINATE CONVENTION, which is the whole point of the regression test
below: ``insertion_pos`` is a coordinate in the ORIGINAL, UNSPLICED genome.
Payloads name their site by the real locus it corresponds to, so a declared
position only means anything in the original frame.

Each splice shifts every coordinate above it by the cassette length, and
``_update_gene_insertion_location`` reads ``insertion_pos`` RAW from the TSV
while comparing it against ``genes``/``transcription_units``/``dna_sites`` that
a previous splice has already shifted. Nothing re-frames a pending cassette's
declared position. So cassettes must be spliced HIGH-TO-LOW: a later, lower
splice then moves an already-placed cassette together with all of its
neighbours, preserving its locus. Splicing low-to-high instead reads every
subsequent declared position in the wrong frame.

⛔ WHY THE OBVIOUS ASSERTIONS DO NOT WORK, and this cost a measurement to learn:

* "both genes are present" stays GREEN through the bug -- a misplaced cassette
  is still loaded, still joined, and still present.
* The loader's own ``"has been shifted from X to Y"`` message CANNOT detect it.
  ``insert_pos`` moves ONLY on collision, so a wrong-frame position that lands
  in an intergenic gap is placed SILENTLY at the wrong locus; and a collision
  shift that does occur bears no relation to the cassette length. A shift can be
  innocent and silence proves nothing, so neither the presence nor the magnitude
  of that message is evidence either way.

⇒ The only sound check is the CONTROL: resolve the same cassette ALONE, resolve
it again with another cassette spliced BELOW it, and require the same
original-frame locus both times. The tolerance is ABSOLUTE -- never expressed in
terms of the loader's own reported relocation, which is the number that hides
this.

Everything here is hermetic and public: cassettes are synthesised from the
public ``gfp`` insertion's column layout with invented identifiers, at round
loci. No private payload is referenced and none is needed.
"""

from pathlib import Path

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

pytestmark = pytest.mark.fast

# Column layout is borrowed from the public ``gfp`` insertion so the joins into
# the base tables keep matching column sets; only identifiers and coordinates
# below are ours.
BASE_INSERTION = "gfp"

# Two invented cassettes. The LOW one is what displaces the HIGH one when the
# splice order is wrong, so its length is the shift the regression test looks
# for -- deliberately not a round number, so an accidental match cannot pass.
LOW = dict(subdir="probe_low", pos=200_000, gene="NG901", length=1_259)
HIGH = dict(subdir="probe_high", pos=400_000, gene="NG902", length=717)


def _gfp_sources():
    """{filename: source Path} for every file of the public insertion."""
    index = SourceBundle()._index
    prefix = f"new_gene_data__{BASE_INSERTION}__"
    return {Path(p).name: Path(p) for k, p in index.items() if k.startswith(prefix)}


def _synthetic_seq(length):
    """A DNA string of exactly ``length`` bases."""
    return ("ATGC" * (length // 4 + 1))[:length]


def _cassette_files(spec):
    """The six TSVs of one synthetic cassette, as {filename: text}.

    Layout mirrors the public insertion exactly; only ids, the declared
    position and the gene length differ. Gene coordinates are RELATIVE to the
    insertion (1..length) -- the loader converts them to genome coordinates.
    """
    gene, sub, length = spec["gene"], spec["subdir"], spec["length"]
    rna, monomer = f"{gene}_RNA", f"NG-{gene}-MONOMER"
    # The protein sequence is unused by the loader path under test; a stub of
    # the right shape keeps the column set valid.
    protein_seq = "M" + "A" * (length // 3 - 1)

    return {
        "insertion_location.tsv": (
            '"subdirectory"\t"insertion_pos"\t"direction"\n'
            f'"{sub}"\t{spec["pos"]}\t"+"\n'
        ),
        "genes.tsv": (
            '"id"\t"symbol"\t"synonyms"\t"left_end_pos"\t"right_end_pos"'
            '\t"direction"\t"rna_ids"\n'
            f'"{gene}"\t"{sub}"\t["{sub}"]\t1\t{length}\t"+"\t["{rna}"]\n'
        ),
        "gene_sequences.tsv": (
            '"id"\t"symbol"\t"synonyms"\t"gene_seq"\n'
            f'"{gene}"\t"{sub}"\t["{sub}"]\t"{_synthetic_seq(length)}"\n'
        ),
        "rnas.tsv": (
            '"id"\t"common_name"\t"synonyms"\t"type"\t"modified_forms"'
            '\t"gene_id"\t"monomer_ids"\t"anticodon"\t"coding_segments"\n'
            f'"{rna}"\t"{sub}"\t["{sub}"]\t"mRNA"\t[]\t"{gene}"'
            f'\t["{monomer}"]\tnull\t[]\n'
        ),
        "proteins.tsv": (
            '"id"\t"common_name"\t"synonyms"\t"seq"\t"experimental_compartment"'
            '\t"computational_compartment"\t"cleavage_of_initial_methionine"'
            '\t"selenocysteine_at_opal"\t"protein_feature_cofactors"'
            '\t"in_complex"\n'
            f'"{monomer}"\t"{sub}"\t["{sub}"]\t"{protein_seq}"'
            '\t["CCO-CYTOSOL"]\t["CCO-CYTOSOL"]\t0\t[]\t{}\t0\n'
        ),
        "rna_half_lives.tsv": '"id"\t"half_life (units.min)"\n',
        "protein_half_lives_measured.tsv": '"id"\t"half life (units.min)"\t"_comments"\n',
    }


def _write_payload(root: Path, option: str, specs, nested=True):
    """Write a new-gene payload as a bundle overlay; return the manifest path.

    ``nested=True`` puts each cassette in its own subdirectory under
    ``option`` (the multi-cassette shape); ``nested=False`` writes a single
    cassette's files directly under ``option`` (the contiguous shape). Both
    resolve through the bundle, which is how payloads actually arrive -- there
    is no local ``flat/`` tree in this repo.
    """
    for spec in specs:
        dest = root / "flat" / "new_gene_data" / option
        if nested:
            dest = dest / spec["subdir"]
        dest.mkdir(parents=True, exist_ok=True)
        for name, text in _cassette_files(spec).items():
            (dest / name).write_text(text)

    manifest = root / "reference_bundle_overlay.tsv"
    rows = ["\t".join(["canonical_key", "source_path", "description", "schema_name"])]
    for f in sorted((root / "flat").rglob("*.tsv")):
        rel = f.relative_to(root / "flat").as_posix()
        rows.append(
            "\t".join(
                [rel[: -len(".tsv")].replace("/", "__"), f"flat/{rel}", "test fixture", ""]
            )
        )
    manifest.write_text("\n".join(rows) + "\n")
    return manifest


def _kb(manifest, option):
    return KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
        new_genes_option=option,
        bundle=SourceBundle(overrides=manifest),
    )


def _locus(kb, gene_id):
    """Genome coordinate the cassette's gene was placed at."""
    rows = [g for g in kb.genes if g["id"] == gene_id]
    assert len(rows) == 1, f"expected exactly one {gene_id} row, got {len(rows)}"
    return rows[0]["left_end_pos"]


# --------------------------------------------------------------------------
# Discovery -- the bundle branch, which is NOT a port
# --------------------------------------------------------------------------
# v2ecoli resolves payloads through a bundle; the upstream fork has only a
# filesystem. A bundle has no directories at all, so the cassette subdirectory
# is recovered from the canonical key's SEGMENTS. Scanning the filesystem under
# a bundle would find nothing and silently DEMOTE a multi-cassette payload to a
# single-cassette one -- which builds, and looks entirely healthy. So the
# assertion has to be on the discovered COUNT, not on whether keys parse.


def test_a_nested_bundle_payload_discovers_every_cassette(tmp_path):
    manifest = _write_payload(tmp_path, "probe_pair", [LOW, HIGH])
    kb = _kb(manifest, "probe_pair")

    found = kb._new_gene_insertion_subdirs("probe_pair")
    # COUNT and membership, not order: discovery is alphabetical and the splice
    # loop re-sorts by declared position, so pinning discovery order here would
    # assert something the loader does not promise.
    assert sorted(found) == sorted([LOW["subdir"], HIGH["subdir"]]), (
        "a multi-cassette payload must not be demoted to a single cassette"
    )


def test_a_contiguous_bundle_payload_reports_no_subdirectories(tmp_path):
    """The single-cassette shape keeps its previous behaviour exactly."""
    manifest = _write_payload(tmp_path, "probe_solo", [HIGH], nested=False)
    kb = _kb(manifest, "probe_solo")

    assert kb._new_gene_insertion_subdirs("probe_solo") == []
    assert _locus(kb, HIGH["gene"]) > 0


# --------------------------------------------------------------------------
# ⭐ The regression test: ALONE vs COMPOSED, asserted absolutely
# --------------------------------------------------------------------------


def test_a_cassettes_locus_does_not_depend_on_another_insertion_below_it(tmp_path):
    """A cassette must land at the same place whether or not it has company.

    ⛔ MUTATION TEST FOR THIS ASSERTION: change the splice loop's
    ``sort(..., reverse=True)`` to ``sort(...)`` (ascending, which is what the
    upstream fork ships) and this test MUST fail. If it still passes, the test
    is wrong -- not the mutation.
    """
    alone_manifest = _write_payload(tmp_path / "alone", "probe_alone", [HIGH])
    composed_manifest = _write_payload(tmp_path / "composed", "probe_pair", [LOW, HIGH])

    alone = _locus(_kb(alone_manifest, "probe_alone"), HIGH["gene"])
    composed_kb = _kb(composed_manifest, "probe_pair")
    composed = _locus(composed_kb, HIGH["gene"])

    # The low cassette splices BELOW the high one, so in the composed build the
    # high cassette has been carried up by exactly the low cassette's length --
    # together with every one of its neighbours, which is what preserves the
    # locus. Removing that known shift recovers the ORIGINAL-frame position,
    # and it is the original-frame position that must match.
    composed_in_original_frame = composed - LOW["length"]

    assert composed_in_original_frame == alone, (
        f"{HIGH['subdir']} resolved to {alone} alone but to "
        f"{composed_in_original_frame} (original frame) when composed with "
        f"{LOW['subdir']} -- a cassette's locus must not depend on what else "
        f"is inserted. Off by {composed_in_original_frame - alone} bp."
    )

    # The low cassette is spliced last and nothing moves after it, so its own
    # locus is directly comparable and pins the other end of the ordering.
    assert _locus(composed_kb, LOW["gene"]) >= LOW["pos"]


def test_new_gene_rows_are_appended_highest_locus_first(tmp_path):
    """Pin the order cassette rows enter the base tables in.

    ⚠ THIS IS A DOWNSTREAM CONTRACT, NOT AN INTERNAL DETAIL, and it is a
    consequence of the descending splice order rather than an independent
    choice. Rows are joined per cassette in splice order, so the base tables
    receive the HIGHEST-locus cassette first.

    Post-ParCa induction (``v2ecoli.perturbations.new_genes``) applies
    per-target expression and translation-efficiency vectors POSITIONALLY,
    against the order ``new_gene_indices()`` reports. If that order derives from
    the order rows were appended here, then the splice order decides which
    weight lands on which gene -- and a vector written against a different
    convention is applied silently to the wrong genes, with the counts still
    matching so no length check can see it.

    ⇒ Anything pairing a weight vector against new genes must resolve targets
    BY NAME. This test exists so that a change to the splice order shows up as
    a failure here rather than as a plausible-looking strain downstream.
    """
    manifest = _write_payload(tmp_path, "probe_pair", [LOW, HIGH])
    kb = _kb(manifest, "probe_pair")

    appended = [g["id"] for g in kb.genes if g["id"] in (LOW["gene"], HIGH["gene"])]
    assert appended == [HIGH["gene"], LOW["gene"]], (
        "new-gene rows must enter the base tables highest-locus-first; "
        "downstream positional weight vectors are indexed on this order"
    )
