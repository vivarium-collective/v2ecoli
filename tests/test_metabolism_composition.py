"""Tests for the ``composition`` criterion (branch-point flux splits).

The composition criterion grades a node's outgoing flux split (a composition on
a simplex) against a reference composition, paired with a closure residual:

  - ROUTING: total-variation distance TV = ½·Σ|p−q| over the renormalized known
    branches → within_tol / drift / mismatch.
  - CLOSURE: the fraction of node influx not accounted by the known branches
    (expected small — a biomass drain); past its band it caps the verdict.

The flagship node is the glucose-6-P fate, EMP / oxidative-PPP / ED. Reference
fractions below are Crown 2015 (per 100 glucose): EMP 71.6, oxPPP 25.4, ED 1.4.
"""

from v2ecoli.library.card_criteria import grade_axis

# Crown 2015 G6P-fate reference (fractions of glucose uptake; ED ~ 1.4%).
CROWN_G6P = {"EMP": 71.6, "oxPPP": 25.4, "ED": 1.4}


def _crit(ref=None, **kw):
    return {"type": "composition", "ref_fractions": ref or CROWN_G6P, **kw}


def test_matching_composition_is_within_tol():
    # Model ~ Crown's split, glucose uptake 100, ~1.5% biomass drain (98.5 catab).
    g = grade_axis({"branches": {"EMP": 71.6, "oxPPP": 25.4, "ED": 1.4},
                    "influx": 100.0}, _crit())
    assert g["verdict"] == "within_tol"
    assert g["detail"]["tv"] < 0.05
    assert abs(g["detail"]["residual"]) < 0.05


def test_misrouted_ppp_is_mismatch_by_tv():
    # Model routes far more through PPP than EMP (Toya-5h-like split), closure ok.
    g = grade_axis({"branches": {"EMP": 40.0, "oxPPP": 58.0, "ED": 0.0},
                    "influx": 100.0}, _crit())
    assert g["verdict"] == "mismatch"
    assert g["detail"]["routing_verdict"] == "mismatch"
    assert g["detail"]["tv"] > 0.15


def test_spurious_sink_caps_verdict_via_residual():
    # Known branches match the split perfectly, but only 60% of influx is
    # accounted — 40% leaves G6P by an unmodeled route. Routing TV ≈ 0, yet the
    # closure residual gate forces a mismatch.
    g = grade_axis({"branches": {"EMP": 43.0, "oxPPP": 15.2, "ED": 0.8},
                    "influx": 100.0}, _crit())
    assert g["detail"]["tv"] < 0.05            # routing among known branches is fine
    assert g["detail"]["residual"] > 0.15
    assert g["detail"]["residual_verdict"] == "mismatch"
    assert g["verdict"] == "mismatch"


def test_ed_active_strain_moves_on_the_simplex():
    # An ED-using model (e.g. a pgi/zwf-context strain) is a different triangle
    # point — flagged relative to the WT reference, not silently normalized away.
    g = grade_axis({"branches": {"EMP": 50.0, "oxPPP": 18.0, "ED": 30.0},
                    "influx": 100.0}, _crit())
    assert g["verdict"] in ("drift", "mismatch")
    assert g["detail"]["model_fractions"]["ED"] > 0.25


def test_no_reference_is_ungraded():
    g = grade_axis({"branches": {"EMP": 70.0}, "influx": 100.0},
                   {"type": "composition"})
    assert g["verdict"] == "ungraded"


def test_zero_influx_is_ungraded():
    assert grade_axis({"branches": {"EMP": 0.0}, "influx": 0.0},
                      _crit())["verdict"] == "ungraded"


def test_tv_is_fraction_misrouted():
    # TV has a clean interpretation: moving 20% of flux from EMP to oxPPP (both
    # closure-neutral) gives TV ≈ 0.20.
    g = grade_axis({"branches": {"EMP": 51.6, "oxPPP": 45.4, "ED": 1.4},
                    "influx": 98.4}, _crit())
    assert abs(g["detail"]["tv"] - 0.20) < 0.02
