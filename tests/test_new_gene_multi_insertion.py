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
#
# ⛔ THE LOCI ARE NOT ARBITRARY, and picking round ones cost us a suite that
# passed on a payload that could not build. Conflict resolution is SINGLE-PASS
# (see ``_update_gene_insertion_location``): it moves past everything straddling
# the declared point, once, and never re-checks. A declared position whose
# RESOLVED position still lands inside a transcription unit therefore stays
# there -- and the resulting genome shifts a gene without shifting its own TU,
# which fails downstream in ``transcription.py`` when ParCa builds sim_data.
# ``[m@2026-08-31]`` 10.4% of random declared positions land in that state, and
# 200_000 -- the obvious round choice, used here originally -- is one of them:
# it resolves to 208609, still inside two TUs. Every test in this file passed on
# it, because they all stop at the knowledge-base layer and the failure is
# downstream.
#
# ⇒ Both loci below are checked to resolve CLEAN in one pass, and both still
# RELOCATE, so the rule-conformance test has a non-zero shift to verify rather
# than trivially passing on a locus that never moves.
LOW = dict(subdir="probe_low", pos=120_000, gene="NG901", length=1_259)
HIGH = dict(subdir="probe_high", pos=400_000, gene="NG902", length=717)


def _gfp_sources():
    """{filename: source Path} for every file of the public insertion."""
    index = SourceBundle()._index
    prefix = f"new_gene_data__{BASE_INSERTION}__"
    return {Path(p).name: Path(p) for k, p in index.items() if k.startswith(prefix)}


def _synthetic_seq(length, seed=0):
    """A deterministic, NON-REPEATING DNA string of exactly ``length`` bases.

    ⚠ Deliberately not a short repeat. A cassette built from ``"ATGC" * n``
    ends in whatever base the cycle lands on, and a boundary off-by-one that
    swaps the cassette's last base for a host gene's first base is then
    invisible whenever those two happen to agree -- which for a 4-cycle is a
    quarter of the time. A previous version of this fixture used exactly that
    and a real off-by-one compared EQUAL by coincidence.
    """
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


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
            f'"{gene}"\t"{sub}"\t["{sub}"]\t"{_synthetic_seq(length, seed=spec["pos"])}"\n'
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

    Rows are joined per cassette in splice order, so the base tables receive the
    HIGHEST-locus cassette first. That is a consequence of the descending splice
    order rather than an independent choice, which is why it is pinned here: a
    change to the splice order should surface as a failure in this file.

    ⛔ WHAT THIS ORDER DOES *NOT* DETERMINE. Post-ParCa induction
    (``v2ecoli.perturbations.new_genes``) applies per-target expression and
    translation-efficiency vectors POSITIONALLY, against the order
    ``new_gene_indices()`` reports. It is tempting to conclude that the append
    order pinned here therefore decides which weight lands on which gene.
    ⚠ IT DOES NOT FOLLOW, and the evidence is genuinely mixed:

    * ``[m]`` file order is NOT preserved into ``cistron_data`` WITHIN a
      multi-gene cassette (measured on a five-gene payload).
    * ``[m]`` across single-gene cassettes, order DOES appear to be preserved
      end to end (measured on a two-cassette build).

    Those were measured by different people on different payloads and are not
    in conflict -- they are about different layers. **The general rule is not
    established**, and the decisive experiment (an end-to-end order check on a
    multi-gene cassette) has not been run.

    ⇒ ⭐ AN UNSTATED RULE IS THE WORSE OUTCOME, NOT THE REASSURING ONE. Had the
    downstream order simply been the append order, it would be predictable from
    the splice order and a positional vector could be reordered mechanically.
    As it stands the position of a given gene in a COMPOSED build cannot be
    predicted from the config, from the loci, or from the splice order.

    ⇒ ⛔ ANY per-target weight vector must be resolved BY NAME, through the ids
    ``new_gene_indices()`` returns alongside its indices. A positionally ported
    vector may be right, wrong, or accidentally right, and the counts match
    either way, so nothing fires.
    """
    manifest = _write_payload(tmp_path, "probe_pair", [LOW, HIGH])
    kb = _kb(manifest, "probe_pair")

    appended = [g["id"] for g in kb.genes if g["id"] in (LOW["gene"], HIGH["gene"])]
    assert appended == [HIGH["gene"], LOW["gene"]], (
        "new-gene rows must enter the base tables highest-locus-first; "
        "downstream positional weight vectors are indexed on this order"
    )


# --------------------------------------------------------------------------
# Correctness -- is the cassette where it was DECLARED, not merely where it
# was last time
# --------------------------------------------------------------------------
# ⛔ WHY A SEPARATE TEST, AND WHAT IT IS AND IS NOT. The control above proves
# INVARIANCE -- a cassette lands in the same place with or without company. It
# does NOT prove that place is the declared one: a build that puts every
# cassette at the same WRONG locus passes it. The two tests answer different
# questions and neither subsumes the other:
#
#     correctness(alone) + invariance(alone -> composed) => correctness(composed)
#
# ⚠ HONEST CHARACTERISATION, because "correctness test" implies more than this
# delivers: the check below re-applies the loader's documented relocation rule
# to independently-sourced HOST annotation. That is a CROSS-CHECK AGAINST
# INDEPENDENT DATA, NOT AN EXTERNAL ORACLE -- stronger than invariance, weaker
# than a true independent reference. It would not catch a change to the
# relocation POLICY; it would move with it. The requirement test that follows is
# the part that does not.
#
# ⚠ No stronger cheap oracle exists, and it is worth recording why each
# candidate fails: the DECLARED position is invalidated by legitimate
# relocation; "splits no host feature" can be satisfied BY LUCK when a
# wrong-frame position lands in an intergenic gap; and cross-checking against
# the upstream fork is COMMON-MODE, since this code is a port of that logic --
# agreement would show the port faithful, not the rule right.


def _baseline_conflict_tables(manifest):
    """Host TU + oriC/TerC rows from a build that never saw a cassette.

    Sourced from ``new_genes_option="off"`` so the coordinates are in the
    ORIGINAL frame and owe nothing to the insertion under test.
    """
    kb = KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False, remove_rrff=False,
        stable_rrna=False, new_genes_option="off",
        bundle=SourceBundle(overrides=manifest),
    )
    return list(kb.transcription_units) + [
        s for s in kb.dna_sites if s["common_name"] in ("oriC", "TerC")
    ]


def _straddlers(rows, pos):
    """Host rows spanning the insertion POINT (not the cassette's span).

    A splice at ``pos`` divides the sequence between ``pos`` and ``pos + 1``,
    so only a feature straddling that point is broken by it -- which is why the
    loader tests the point rather than the cassette's extent.
    """
    return [
        r for r in rows
        if r["left_end_pos"] not in (None, "")
        and r["right_end_pos"] not in (None, "")
        and r["left_end_pos"] < pos
        and r["right_end_pos"] >= pos
    ]


def _resolved_insert_pos(kb, gene_id):
    """Where the cassette was actually spliced, from where its gene landed.

    Gene coordinates are relative-1 within the cassette and are converted as
    ``left + insert_pos``, so the first gene's left end is ``insert_pos + 1``.
    """
    return _locus(kb, gene_id) - 1


def test_a_cassettes_resolved_position_is_explained_by_host_annotation(tmp_path):
    """The declared position, or one base past whatever host feature blocks it.

    ⛔ MUTATION TEST: change the loader's ``shift`` computation (e.g. drop the
    ``+ 1``, or relocate to the conflict's LEFT end) and this must fail.
    """
    manifest = _write_payload(tmp_path, "probe_alone", [HIGH])
    kb = _kb(manifest, "probe_alone")
    host = _baseline_conflict_tables(manifest)

    blocking = _straddlers(host, HIGH["pos"])
    expected = (
        max(r["right_end_pos"] for r in blocking) + 1 if blocking else HIGH["pos"]
    )

    assert _resolved_insert_pos(kb, HIGH["gene"]) == expected, (
        "the resolved position is not explained by the host annotation at the "
        "declared locus -- it is neither the declared position nor one base "
        "past the feature blocking it"
    )


def test_no_host_transcription_unit_is_split_by_an_insertion(tmp_path):
    """The REQUIREMENT relocation exists to satisfy, not the rule implementing it.

    ⚠ SCOPED, and the scope is the point: this covers TRANSCRIPTION UNITS --
    what the loader actually consults when ``operons_on`` is set. Gene-level
    splitting is UNCHECKED here by design: with operons on the loader never
    looks at ``genes``, so a cassette can split a gene belonging to no
    transcription unit and nothing objects. That is a pre-existing
    single-insertion defect, reported rather than encoded, so that a
    multi-insertion change is not held hostage to it.

    Unlike the rule check above this survives a change of relocation POLICY --
    it asserts the outcome, not the mechanism.
    """
    manifest = _write_payload(tmp_path, "probe_pair", [LOW, HIGH])
    kb = _kb(manifest, "probe_pair")

    host_tus = [
        t for t in kb.transcription_units
        if not str(t.get("id", "")).startswith("NG")
        and t["left_end_pos"] not in (None, "")
        and t["right_end_pos"] not in (None, "")
    ]
    for spec in (LOW, HIGH):
        pos = _resolved_insert_pos(kb, spec["gene"])
        split = [t for t in host_tus
                 if t["left_end_pos"] <= pos < t["right_end_pos"]]
        assert not split, (
            f"{spec['subdir']} was spliced at {pos}, inside host "
            f"transcription unit(s) {[t['id'] for t in split][:3]} -- "
            "relocation exists precisely to prevent this"
        )


def test_both_cassettes_hold_their_locus_not_only_the_upper_one(tmp_path):
    """Symmetric to the control above, for the cassette spliced LAST.

    The upper cassette is the discriminating case for splice ORDER. The lower
    one is spliced last and nothing moves after it, so its locus is directly
    comparable with no frame correction -- which makes it the case that would
    expose a defect in relocation rather than in ordering.
    """
    alone_manifest = _write_payload(tmp_path / "alone", "probe_solo_low", [LOW])
    composed_manifest = _write_payload(tmp_path / "composed", "probe_pair", [LOW, HIGH])

    alone = _locus(_kb(alone_manifest, "probe_solo_low"), LOW["gene"])
    composed = _locus(_kb(composed_manifest, "probe_pair"), LOW["gene"])

    assert composed == alone, (
        f"{LOW['subdir']} resolved to {alone} alone but {composed} composed; "
        "it is spliced last, so nothing should move it"
    )
