"""Tests for the genotype build-integrity card's measurement core.

The card grades a ParCa-level knockout's knowledge base. These tests do the thing
that makes such a card worth having: they check it FAILS on corrupted input. An
all-passing integrity check is indistinguishable from one that measures nothing, and
this suite's centre of gravity is the fault-injection cases below rather than the
happy path.

No ParCa fit is needed — four of the five declared readouts are properties of
``raw_data`` (see v2ecoli/library/genotype_build.py), so this runs in seconds.
"""
from __future__ import annotations

import copy

import pytest

from v2ecoli.library import genotype_build as gb

LACY = "EG10526"
LACY_SPAN = (361926, 363179)
LACY_BP = LACY_SPAN[1] - LACY_SPAN[0] + 1  # 1254


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    """(wt, ko, spans) for a lacY knockout, built through the ParCa step's own
    resolver so the test exercises the same path a study run does."""
    workdir = tmp_path_factory.mktemp("genotype_build")
    manifest, _gid, spans = gb.make_knockout_bundle([LACY], workdir)
    return gb.resolve_raw_data(None), gb.resolve_raw_data(manifest), spans


@pytest.fixture(scope="module")
def measured(arms):
    wt, ko, spans = arms
    return gb.measure_structure(wt, ko, [LACY], spans)


# --- the generator records what we grade against -----------------------------

def test_generator_records_the_deleted_span(arms):
    """Spans come from genotype.json, not re-derived — so this pins the provenance."""
    _wt, _ko, spans = arms
    assert spans[LACY] == LACY_SPAN


# --- happy path ---------------------------------------------------------------

def test_all_structural_axes_pass_on_a_real_knockout(measured):
    flags = {k: v["ok"] for k, v in measured.items() if isinstance(v, dict) and "ok" in v}
    assert flags == {"chromosome_length": True, "round_trip": True, "tombstone": True,
                     "coordinate_shift": True, "coordinate_bounds": True,
                     "functional_absence": True}


def test_shrunk_branch_is_exercised_on_the_real_knockout(measured):
    """lacY sits inside the lac operon, so transcription units straddle the deletion
    and must land in the SHRUNK branch. Zero shrunk features would mean the second
    branch of the round trip was never exercised — the study's own vacuity flag."""
    rt = measured["round_trip"]
    assert rt["ok"] is True
    assert rt["shrunk_excised_ok"] >= 1
    assert rt["by_class"]["transcription_units"]["shrunk_excised_ok"] >= 1


def test_chromosome_shortens_by_exactly_the_span(measured):
    assert measured["chromosome_length"]["observed_delta"] == LACY_BP


def test_only_the_knocked_out_gene_is_newly_tombstoned(measured):
    """The wild type already carries null-coordinate genes; only the DELTA counts.
    Asserting 'no null-coordinate genes' would fail on correct data."""
    assert measured["tombstone"]["newly_tombstoned"] == [LACY]
    assert measured["tombstone"]["resurrected"] == []
    assert measured["tombstone"]["wt_pre_existing_nulls"] > 0


def test_tombstoned_gene_keeps_its_row_but_leaves_the_functional_sets(measured):
    """D16: the id legitimately persists, so id-absence is the WRONG check."""
    m = measured["functional_absence"]["detail"][LACY]
    assert m["row_retained_in_genes"] is True
    assert m["wt_valid_gene_ids"] and m["wt_all_mRNA_cistrons"]
    assert not m["ko_valid_gene_ids"] and not m["ko_all_mRNA_cistrons"]


# --- fault injection: the part that makes this card mean something ------------

def test_round_trip_catches_a_LENGTH_PRESERVING_corruption(arms):
    """A 1 bp substitution leaves every length identical. A length check passes; the
    sequence check must not. This is the whole reason round-trip compares content."""
    wt, ko, spans = arms
    broken = copy.copy(ko)
    pos = 2_000_000
    g = ko.genome_sequence
    broken.genome_sequence = g[:pos] + ("A" if g[pos] != "A" else "T") + g[pos + 1:]

    out = gb.measure_structure(wt, broken, [LACY], spans)
    assert out["round_trip"]["ok"] is False
    # >= 1, not == 1: the corrupted base may sit inside a gene AND a transcription
    # unit / DNA site, so more than one feature class can report it.
    assert out["round_trip"]["differing"] >= 1
    # ...and the length axis is unmoved, proving it could not have caught this.
    assert out["chromosome_length"]["ok"] is True


def test_off_by_one_span_is_caught_by_the_round_trip(arms):
    """Shift the recorded span by one base: the deleted LENGTH is unchanged, so the
    chromosome-length axis passes — only the excision comparison can object. This is
    the defect class (#455 defect 4, the off-by-one splice) the two-branch form
    exists to catch at genome scale."""
    wt, ko, _spans = arms
    out = gb.measure_structure(
        wt, ko, [LACY], {LACY: (LACY_SPAN[0] + 1, LACY_SPAN[1] + 1)})
    assert out["chromosome_length"]["ok"] is True
    assert out["round_trip"]["ok"] is False


def test_out_of_range_coordinate_fails_bounds(arms):
    """Push one surviving gene past the shortened chromosome end: only the bounds
    axis is positioned to object."""
    wt, ko, spans = arms
    broken = copy.copy(ko)
    rows = [dict(g) for g in ko.genes]
    victim = next(r for r in rows if r["left_end_pos"] is not None)
    victim["right_end_pos"] = len(ko.genome_sequence) + 50
    broken.genes = rows
    out = gb.measure_structure(wt, broken, [LACY], spans)
    assert out["coordinate_bounds"]["ok"] is False
    assert out["coordinate_bounds"]["violations"] == 1


def test_wrong_expected_span_fails_length_and_shift(arms):
    """If the recorded span disagrees with the genome, both coordinate axes fail."""
    wt, ko, _spans = arms
    out = gb.measure_structure(wt, ko, [LACY], {LACY: (LACY_SPAN[0], LACY_SPAN[1] + 100)})
    assert out["chromosome_length"]["ok"] is False
    assert out["coordinate_shift"]["ok"] is False


def test_an_unperformed_knockout_fails_rather_than_passes(arms):
    """Grading the wild type against itself must FAIL: nothing was deleted, so the
    gene is still functional. Guards against a card that passes on a no-op."""
    wt, _ko, spans = arms
    out = gb.measure_structure(wt, wt, [LACY], spans)
    assert out["tombstone"]["ok"] is False
    assert out["functional_absence"]["ok"] is False
    assert out["chromosome_length"]["ok"] is False


# --- the mode trap ------------------------------------------------------------

def test_conditions_fitted_reads_cell_specs_not_conditions():
    """`conditions` is the full condition LIST and reads 51 in BOTH fast and full
    mode; only `cell_specs` distinguishes them. An axis against `conditions` would
    pass at 51 on a fast build and never discriminate."""
    fit = gb.measure_fit({"cell_specs": {"basal": {}, "with_aa": {}},
                          "conditions": list(range(51)),
                          "mechanistic_fit_status": {"mechanistic_supply": "ok"}})
    assert fit["conditions_fitted"] == 2
    assert fit["conditions_declared"] == 51


def test_absent_parca_state_leaves_fit_axes_ungraded():
    """No build -> ungraded, never a fabricated pass."""
    assert gb.measure_fit(None)["completed"] is None


def test_failed_mechanistic_fit_is_recorded_not_passed():
    fit = gb.measure_fit({"cell_specs": {}, "conditions": [],
                          "mechanistic_fit_status": {"mechanistic_supply": "error"}})
    assert fit["completed"] is False
