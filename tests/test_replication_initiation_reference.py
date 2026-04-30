"""Tests that lock in the molecular facts from the curated PDF.

The curated source is ``docs/references/replication_initiation_molecular_info.pdf``
with a structured Markdown mirror at
``docs/references/replication_initiation.md``. The Python module
``v2ecoli.data.replication_initiation.molecular_reference`` codifies those
facts as importable constants. These tests assert that the codified data
agrees with the PDF.

If a fact in the PDF changes, the workflow is:
  1. Update the PDF (and the Markdown mirror).
  2. Update the Python data module.
  3. Update the matching assertion here.
All three changes land in the same PR.

These tests are pure data assertions and do not exercise the simulator,
so they run in the fast (non-``sim``) test job and do not need the ParCa
cache or any sim_data fixture.
"""

from __future__ import annotations

import re

import pytest

from v2ecoli.data.replication_initiation import (
    DNAA_BOX_CONSENSUS,
    DNAA_BOX_HIGHEST_AFFINITY,
    DNAA_BOX_HIGH_AFFINITY_KD_NM,
    DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND,
    DNAA_BOX_RELAXED_MOTIF,
    ORIC,
    DNAA_PROMOTER,
    DATA,
    DARS1,
    DARS2,
    SEQA,
    RIDA,
)


# ---------------------------------------------------------------------------
# Helpers — compile IUPAC-style motifs into regexes.
# ---------------------------------------------------------------------------

_IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
    'W': '[AT]',
    'M': '[AC]',
    'V': '[ACG]',
    'H': '[ACT]',
    'N': '[ACGT]',
}


def _iupac_to_regex(motif: str) -> re.Pattern:
    return re.compile('^' + ''.join(_IUPAC[c] for c in motif) + '$')


# ---------------------------------------------------------------------------
# DnaA box motif facts
# ---------------------------------------------------------------------------

class TestDnaABoxMotifs:

    def test_consensus_is_nine_bp(self):
        assert len(DNAA_BOX_CONSENSUS) == 9

    def test_consensus_string_uses_iupac(self):
        # PDF: TTWTNCACA. Fixed bases at positions 1,2,4,6,7,8,9.
        assert DNAA_BOX_CONSENSUS == 'TTWTNCACA'

    def test_highest_affinity_is_nine_bp(self):
        assert len(DNAA_BOX_HIGHEST_AFFINITY) == 9

    def test_highest_affinity_string(self):
        assert DNAA_BOX_HIGHEST_AFFINITY == 'TTATCCACA'

    def test_highest_affinity_matches_consensus(self):
        regex = _iupac_to_regex(DNAA_BOX_CONSENSUS)
        assert regex.match(DNAA_BOX_HIGHEST_AFFINITY)

    def test_relaxed_motif_string(self):
        assert DNAA_BOX_RELAXED_MOTIF == 'HHMTHCWVH'

    def test_relaxed_motif_admits_consensus(self):
        # The HHMTHCWVH motif should admit TTATCCACA.
        regex = _iupac_to_regex(DNAA_BOX_RELAXED_MOTIF)
        assert regex.match('TTATCCACA')

    def test_high_vs_low_affinity_kd_separation(self):
        # PDF: high-affinity boxes ~1 nM, low-affinity > 100 nM.
        # Two orders of magnitude separates the classes.
        assert (DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND
                >= 100 * DNAA_BOX_HIGH_AFFINITY_KD_NM)


# ---------------------------------------------------------------------------
# oriC
# ---------------------------------------------------------------------------

class TestOriC:

    def test_length_462_bp(self):
        assert ORIC.length_bp == 462

    def test_eleven_dnaA_boxes(self):
        assert len(ORIC.dnaA_boxes) == 11

    def test_three_high_affinity_boxes(self):
        high = [b for b in ORIC.dnaA_boxes if b.affinity_class == 'high']
        assert len(high) == 3

    def test_eight_low_affinity_boxes(self):
        low = [b for b in ORIC.dnaA_boxes if b.affinity_class == 'low']
        assert len(low) == 8

    def test_high_affinity_boxes_are_R1_R2_R4(self):
        names = {b.name for b in ORIC.dnaA_boxes if b.affinity_class == 'high'}
        assert names == {'R1', 'R2', 'R4'}

    def test_low_affinity_box_names(self):
        names = {b.name for b in ORIC.dnaA_boxes if b.affinity_class == 'low'}
        assert names == {'R5M', 'tau2', 'I1', 'I2', 'I3', 'C1', 'C2', 'C3'}

    def test_R1_and_R4_are_consensus(self):
        by_name = {b.name: b for b in ORIC.dnaA_boxes}
        assert by_name['R1'].sequence == DNAA_BOX_HIGHEST_AFFINITY
        assert by_name['R4'].sequence == DNAA_BOX_HIGHEST_AFFINITY

    def test_R2_differs_from_consensus_by_one_base(self):
        by_name = {b.name: b for b in ORIC.dnaA_boxes}
        diff = sum(
            a != b for a, b in zip(by_name['R2'].sequence, DNAA_BOX_HIGHEST_AFFINITY))
        assert diff == 1

    def test_R2_is_TTATACACA(self):
        by_name = {b.name: b for b in ORIC.dnaA_boxes}
        assert by_name['R2'].sequence == 'TTATACACA'

    def test_high_affinity_boxes_bind_both_nucleotide_forms(self):
        for b in ORIC.dnaA_boxes:
            if b.affinity_class == 'high':
                assert b.nucleotide_preference == 'both', \
                    f'{b.name} should bind both ATP and ADP forms'

    def test_low_affinity_boxes_prefer_dnaA_atp(self):
        for b in ORIC.dnaA_boxes:
            if b.affinity_class == 'low':
                assert b.nucleotide_preference == 'atp', \
                    f'{b.name} should be DnaA-ATP-preferential'

    def test_two_ihf_sites(self):
        assert len(ORIC.ihf_sites) == 2

    def test_ibs_naming_and_roles(self):
        by_name = {s.name: s for s in ORIC.ihf_sites}
        assert by_name['IBS1'].role == 'primary'
        assert by_name['IBS2'].role == 'secondary'

    def test_right_arm_oligomerization_order(self):
        # PDF: ordered DnaA-ATP loading on right arm goes C1 -> I3 -> C2 -> C3.
        assert ORIC.ordered_oligomerization_right_arm == ('C1', 'I3', 'C2', 'C3')


# ---------------------------------------------------------------------------
# dnaA promoter
# ---------------------------------------------------------------------------

class TestDnaAPromoter:

    def test_length_448_bp(self):
        assert DNAA_PROMOTER.length_bp == 448

    def test_two_promoters_p1_p2(self):
        assert DNAA_PROMOTER.promoters == ('p1', 'p2')

    def test_p2_three_fold_stronger_than_p1(self):
        assert DNAA_PROMOTER.p2_to_p1_strength_ratio == pytest.approx(3.0)

    def test_promoter_separation_about_80_bp(self):
        assert DNAA_PROMOTER.promoter_separation_bp == 80

    def test_box1_is_consensus_high_affinity(self):
        by_name = {b.name: b for b in DNAA_PROMOTER.dnaA_boxes}
        assert by_name['box1'].affinity_class == 'high'
        assert by_name['box1'].sequence == DNAA_BOX_HIGHEST_AFFINITY
        assert by_name['box1'].nucleotide_preference == 'both'

    def test_box3_is_very_low_affinity(self):
        by_name = {b.name: b for b in DNAA_PROMOTER.dnaA_boxes}
        assert by_name['box3'].affinity_class == 'very_low'

    def test_atp_preferential_boxes_present(self):
        atp_only = {b.name for b in DNAA_PROMOTER.dnaA_boxes
                    if b.nucleotide_preference == 'atp'}
        # box4 (overlaps boxa) and box b, c are ATP-preferential.
        assert {'box4', 'boxb', 'boxc'} <= atp_only


# ---------------------------------------------------------------------------
# datA
# ---------------------------------------------------------------------------

class TestDatA:

    def test_length_363_bp(self):
        assert DATA.length_bp == 363

    def test_chromosomal_position_94_point_7_min(self):
        assert DATA.chromosomal_position_min == pytest.approx(94.7)

    def test_four_dnaA_boxes(self):
        assert DATA.n_dnaA_boxes == 4

    def test_essential_boxes_2_3_7(self):
        assert set(DATA.essential_dnaA_box_names) == {'box2', 'box3', 'box7'}

    def test_box4_is_stimulatory(self):
        assert 'box4' in DATA.stimulatory_dnaA_box_names

    def test_one_ihf_site(self):
        assert DATA.n_ihf_sites == 1


# ---------------------------------------------------------------------------
# DARS1 / DARS2
# ---------------------------------------------------------------------------

class TestDars:

    def test_dars1_length_632_bp(self):
        assert DARS1.length_bp == 632

    def test_dars2_length_737_bp(self):
        assert DARS2.length_bp == 737

    def test_dars1_has_core_boxes_I_II_III(self):
        assert set(DARS1.core_box_names) == {'I', 'II', 'III'}

    def test_dars2_has_core_boxes_I_II_III(self):
        assert set(DARS2.core_box_names) == {'I', 'II', 'III'}

    def test_dars2_has_extra_box_iv_v(self):
        assert set(DARS2.extra_box_names) == {'IV', 'V'}

    def test_dars1_has_no_extra_boxes(self):
        assert DARS2.extra_box_names != DARS1.extra_box_names
        assert DARS1.extra_box_names == ()

    def test_dars2_dominant_in_vivo(self):
        assert DARS2.is_dominant_in_vivo is True

    def test_dars1_not_dominant_in_vivo(self):
        assert DARS1.is_dominant_in_vivo is False

    def test_dars2_has_ihf_and_fis_sites(self):
        assert DARS2.n_ihf_sites >= 1
        assert DARS2.n_fis_sites >= 1


# ---------------------------------------------------------------------------
# SeqA sequestration
# ---------------------------------------------------------------------------

class TestSeqA:

    def test_sequestration_window_about_10_min(self):
        assert SEQA.sequestration_window_minutes == pytest.approx(10.0)

    def test_window_is_about_one_third_of_doubling_time(self):
        assert SEQA.fraction_of_doubling_time_at_rapid_growth == pytest.approx(1 / 3)

    def test_more_than_ten_gatc_sites_at_oriC(self):
        assert SEQA.n_gatc_sites_oriC_lower_bound >= 10

    def test_seqA_binds_hemimethylated_state(self):
        assert SEQA.binds_state == 'hemimethylated'


# ---------------------------------------------------------------------------
# RIDA
# ---------------------------------------------------------------------------

class TestRida:

    def test_clamp_is_dnaN(self):
        assert RIDA.clamp_protein == 'DnaN'

    def test_catalytic_partner_is_hda(self):
        assert RIDA.catalytic_partner == 'Hda'

    def test_hda_nucleotide_state_is_adp(self):
        # PDF: ADP-bound Hda forms the active complex with the loaded clamp.
        assert RIDA.hda_nucleotide_state == 'ADP'

    def test_hda_clamp_binding_motif_is_n_terminal(self):
        assert RIDA.hda_clamp_binding_motif_terminus == 'N'

    def test_reaction_converts_atp_form_to_adp_form(self):
        assert 'DnaA-ATP' in RIDA.reaction
        assert 'DnaA-ADP' in RIDA.reaction
        # Direction: ATP -> ADP, not the other way around.
        assert RIDA.reaction.index('DnaA-ATP') < RIDA.reaction.index('DnaA-ADP')


# ---------------------------------------------------------------------------
# Cross-region invariants
# ---------------------------------------------------------------------------

class TestCrossRegionConsistency:
    """Properties that must hold across multiple regions."""

    def test_consensus_TTATCCACA_appears_in_oriC_and_dnaA_promoter(self):
        oriC_seqs = {b.sequence for b in ORIC.dnaA_boxes if b.sequence}
        promoter_seqs = {b.sequence for b in DNAA_PROMOTER.dnaA_boxes if b.sequence}
        assert DNAA_BOX_HIGHEST_AFFINITY in oriC_seqs
        assert DNAA_BOX_HIGHEST_AFFINITY in promoter_seqs

    def test_all_known_box_sequences_are_nine_bp(self):
        for region, name in [(ORIC, 'oriC'), (DNAA_PROMOTER, 'dnaA promoter')]:
            for b in region.dnaA_boxes:
                if b.sequence is not None:
                    assert len(b.sequence) == 9, \
                        f'{name}:{b.name} sequence not 9 bp: {b.sequence!r}'

    def test_all_known_box_sequences_match_relaxed_motif(self):
        # The relaxed motif HHMTHCWVH is meant to admit DnaA boxes seen in
        # datA / DARS1 / DARS2; the strict consensus and R1/R4/box1 should
        # also satisfy it.
        regex = _iupac_to_regex(DNAA_BOX_RELAXED_MOTIF)
        for region, name in [(ORIC, 'oriC'), (DNAA_PROMOTER, 'dnaA promoter')]:
            for b in region.dnaA_boxes:
                if b.sequence is not None:
                    assert regex.match(b.sequence), \
                        f'{name}:{b.name} sequence {b.sequence!r} does not match relaxed motif'
