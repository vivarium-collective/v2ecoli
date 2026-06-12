"""
Tests for dnaA-promoter autoregulation signal (dnaa-4).

The _promoter_fraction helper computes the bound fraction of POOL_PROMOTER_HIGH
sites and is published on the dnaa_hydrolysis port each tick.
"""

import numpy as np


def test_promoter_fraction_from_pool_state():
    from v2ecoli.steps.dnaa_box_binding import _promoter_fraction, POOL_PROMOTER_HIGH, FORM_FREE

    # 2 promoter sites (indices 0, 1), 3 non-promoter sites.
    # index 0: promoter, free; index 1: promoter, bound (form 1)
    # indices 2-4: non-promoter (mixed — should be ignored)
    pool_label = np.array([POOL_PROMOTER_HIGH, POOL_PROMOTER_HIGH, 0, 0, 0], dtype=np.int8)
    bound_form = np.array([FORM_FREE, 1, FORM_FREE, 1, 1], dtype=np.int8)

    # 1 of 2 promoter sites bound → 0.5
    assert _promoter_fraction(pool_label, bound_form) == 0.5

    # No promoter sites at all → 0.0
    assert _promoter_fraction(np.zeros(5, np.int8), bound_form) == 0.0
