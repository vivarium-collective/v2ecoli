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

    # Saturated extreme: all promoter sites bound → 1.0
    assert _promoter_fraction(np.array([POOL_PROMOTER_HIGH, POOL_PROMOTER_HIGH], np.int8), np.array([1, 1], np.int8)) == 1.0


def test_autoreg_scaling_factor():
    from v2ecoli.processes.transcript_initiation import _autoreg_factor
    assert _autoreg_factor(0.0, 0.8) == 1.0
    assert abs(_autoreg_factor(1.0, 0.8) - 0.2) < 1e-9
    assert abs(_autoreg_factor(0.5, 0.8) - 0.6) < 1e-9
    assert _autoreg_factor(1.0, 0.0) == 1.0


def test_autoreg_preserves_normalized_distribution():
    import numpy as np
    from v2ecoli.processes.transcript_initiation import _autoreg_factor
    probs = np.array([0.25, 0.25, 0.25, 0.25])      # normalized
    dnaa = np.array([False, True, False, False])     # represses index 1
    f, s = 1.0, 0.8
    probs[dnaa] *= _autoreg_factor(f, s)             # 0.25 -> 0.05
    probs /= probs.sum()                              # renormalize (the fix)
    assert abs(probs.sum() - 1.0) < 1e-12            # stays a valid distribution
    # dnaA's share dropped; the other three rose proportionally + stayed equal
    assert probs[1] < 0.25
    assert np.allclose(probs[[0, 2, 3]], probs[0])   # untargeted TUs stay equal to each other
