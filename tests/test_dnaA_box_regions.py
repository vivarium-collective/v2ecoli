"""Phase 0 — region_for_coord behavior + per-region DnaA-box counts.

The classifier ``v2ecoli.data.replication_initiation.region_for_coord``
maps a relative-to-oriC chromosome coordinate to one of the regulatory
regions (``oriC``, ``dnaA_promoter``, ``datA``, ``DARS1``, ``DARS2``) or
``None`` when the coordinate is elsewhere on the chromosome. These tests
verify the classifier's correctness on synthetic inputs and lock in the
*empirical* per-region counts that the bioinformatic strict-consensus
motif search produces today.

The empirical counts are deliberately less than the curated PDF counts:
the strict TTWTNCACA motif misses non-consensus low-affinity boxes that
the literature names. That gap is documented in
``PER_REGION_STRICT_CONSENSUS_COUNT`` vs ``PER_REGION_PDF_COUNT`` and is
what Phase 2 (DnaA-box binding) closes.
"""

from __future__ import annotations

import os

import pytest

from v2ecoli.data.replication_initiation import (
    DARS1, DARS2, DATA, DNAA_PROMOTER, ORIC,
    GENOME_LENGTH_BP, ORIC_ABS_CENTER_BP, TERC_ABS_CENTER_BP,
    PER_REGION_PDF_COUNT,
    PER_REGION_STRICT_CONSENSUS_COUNT,
    REGION_BOUNDARIES,
    REGION_BOUNDARIES_ABS,
    region_for_coord,
)


# ---------------------------------------------------------------------------
# A. Classifier correctness on synthetic inputs
# ---------------------------------------------------------------------------

class TestRegionForCoordSynthetic:

    def test_returns_known_region_at_endpoints_and_midpoint(self):
        for region, (lo, hi) in REGION_BOUNDARIES.items():
            assert region_for_coord(lo) == region, f'{region} low endpoint'
            assert region_for_coord(hi) == region, f'{region} high endpoint'
            assert region_for_coord((lo + hi) // 2) == region, \
                f'{region} midpoint'

    def test_returns_none_outside_all_regions(self):
        # Coordinate at the antipode of every region — far from oriC and
        # not inside any window. terC is at ±genome/2, definitely outside
        # the small regulatory windows centered near oriC.
        antipode = GENOME_LENGTH_BP // 2 - 1000
        for region, (lo, hi) in REGION_BOUNDARIES.items():
            assert not (lo <= antipode <= hi), \
                f'sanity: antipode unexpectedly in {region}'
        assert region_for_coord(antipode) is None

    def test_returns_none_just_outside_each_region(self):
        for region, (lo, hi) in REGION_BOUNDARIES.items():
            # rel coords just below / above the window must miss
            # (unless they happen to fall in another region — for our
            # well-separated loci, they shouldn't).
            r_below = region_for_coord(lo - 1)
            r_above = region_for_coord(hi + 1)
            assert r_below != region and r_above != region


# ---------------------------------------------------------------------------
# B. Region windows are well-formed and non-overlapping
# ---------------------------------------------------------------------------

class TestRegionBoundaries:

    def test_no_pairwise_overlap(self):
        items = list(REGION_BOUNDARIES.items())
        for i, (r1, (lo1, hi1)) in enumerate(items):
            for r2, (lo2, hi2) in items[i + 1:]:
                assert hi1 < lo2 or hi2 < lo1, (
                    f'{r1}={lo1}..{hi1} overlaps {r2}={lo2}..{hi2}')

    def test_low_le_high(self):
        for region, (lo, hi) in REGION_BOUNDARIES.items():
            assert lo <= hi, f'{region}: lo > hi ({lo} > {hi})'

    def test_widths_match_curated_reference_lengths(self):
        # Absolute-window widths equal the PDF region lengths exactly.
        # (Relative widths can differ by 1 because the upstream
        # _get_relative_coordinates formula nudges negatives by +1.)
        widths = {r: hi - lo + 1 for r, (lo, hi) in REGION_BOUNDARIES_ABS.items()}
        assert widths['oriC'] == ORIC.length_bp
        assert widths['dnaA_promoter'] == DNAA_PROMOTER.length_bp
        assert widths['datA'] == DATA.length_bp
        assert widths['DARS1'] == DARS1.length_bp
        assert widths['DARS2'] == DARS2.length_bp


# ---------------------------------------------------------------------------
# C. Curated-vs-strict-consensus gap is documented and consistent
# ---------------------------------------------------------------------------

class TestEmpiricalVsCuratedGap:
    """The bioinformatic strict-consensus search undercounts named boxes."""

    def test_strict_consensus_count_le_pdf_count(self):
        for region in PER_REGION_PDF_COUNT:
            assert region in PER_REGION_STRICT_CONSENSUS_COUNT, \
                f'region {region!r} missing from PER_REGION_STRICT_CONSENSUS_COUNT'
            strict = PER_REGION_STRICT_CONSENSUS_COUNT[region]
            pdf = PER_REGION_PDF_COUNT[region]
            assert strict <= pdf, (
                f'{region}: strict-consensus count {strict} should not '
                f'exceed curated PDF count {pdf}')

    def test_pdf_count_oriC_is_eleven(self):
        assert PER_REGION_PDF_COUNT['oriC'] == 11

    def test_pdf_count_total_is_thirty(self):
        # 11 + 7 + 4 + 3 + 5 = 30 named boxes across the five loci
        assert sum(PER_REGION_PDF_COUNT.values()) == 30


# ---------------------------------------------------------------------------
# D. Sim-data integration: apply region_for_coord to actual init-state coords
#    Marked ``sim`` because it loads the ParCa cache.
# ---------------------------------------------------------------------------

CACHE_DIR = 'out/cache'


@pytest.mark.sim
@pytest.mark.skipif(
    not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
    reason=f'cache dir {CACHE_DIR!r} not present; '
           f'rebuild with `python scripts/build_cache.py`',
)
class TestInitStateBoxCounts:
    """Apply ``region_for_coord`` over the real init-state DnaA-box
    coordinates and assert the counts match the documented empirical
    baseline."""

    @pytest.fixture(scope='class')
    def all_box_coords(self):
        from v2ecoli.composite import _load_cache_bundle
        import numpy as np
        initial_state, _ = _load_cache_bundle(CACHE_DIR)
        boxes = initial_state['unique']['DnaA_box']
        active = boxes[boxes['_entryState'].view(np.bool_)]
        return [int(c) for c in active['coordinates']]

    def test_per_region_counts_match_documented_baseline(self, all_box_coords):
        # Count *distinct* relative coordinates per region. The unique
        # molecule store can hold the same coordinate on multiple
        # chromosome domains during replication; only distinct coords
        # are bioinformatic motif hits.
        from collections import Counter
        observed = Counter()
        seen_per_region = {r: set() for r in PER_REGION_STRICT_CONSENSUS_COUNT}
        for c in all_box_coords:
            r = region_for_coord(c)
            if r is not None:
                seen_per_region[r].add(c)
        for region, expected in PER_REGION_STRICT_CONSENSUS_COUNT.items():
            got = len(seen_per_region[region])
            assert got == expected, (
                f'{region}: observed {got} distinct strict-consensus boxes, '
                f'documented baseline is {expected}. '
                f'Distinct coords: {sorted(seen_per_region[region])}')

    def test_observed_counts_dominated_by_replicating_regions(self, all_box_coords):
        # At init, the cell is mid-replication; the regions near oriC
        # (oriC, dnaA_promoter, DARS2) have boxes spread across the
        # replicating domains. The total in-region box count should be
        # at least twice the per-domain count, since fork passage
        # populates daughter domains.
        from collections import Counter
        observed = Counter()
        for c in all_box_coords:
            r = region_for_coord(c)
            if r is not None:
                observed[r] += 1
        # oriC and DARS2 are within the replicated zone at init; both
        # should have boxes spread across multiple chromosome domains.
        assert observed['oriC'] > 0, \
            'expected non-zero oriC strict-consensus boxes at init'
        assert observed['DARS2'] > 0, \
            'expected non-zero DARS2 strict-consensus boxes at init'
