"""B2: `distribution` report card — violin/strip of per-cell values pooled
across seeds x generations, graded via v2ecoli.library.card_criteria.grade_axis
(ttest). Uses the redux_cards fixture (Task B0): 1 seed x 1 generation ->
n=1 per engine, so grade_axis's ttest branch (needs n>=2) must fall back to
'ungraded' per axis rather than crash.

Goes through the Step round-trip (see test_report_card_trajectory.py's
docstring for why: ``@as_step`` returns a Step class, not a plain callable).
"""
from conftest import make_card_state

import scripts._compare.report_cards.distribution  # noqa: F401  (registers the Step)
from _card_helpers import _run_card


def test_distribution_card_violin_and_graded():
    out = _run_card("distribution", make_card_state())
    assert "violin" in out["card_html"].lower()
    assert out["verdict"] in ("within_tol", "drift", "mismatch", "ungraded")
    assert out["axes"]
    assert all("detail" in a for a in out["axes"])
