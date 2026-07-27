"""B4: `composition` report card — grouped-bar of proteome vs RNA vs "other"
mass FRACTIONS (v2ecoli vs vEcoli), a doubling-time / final-vs-initial mass
readout for the 1-generation fixture (no division events), and an optional
steps/s perf line (skipped gracefully when no run-summary JSON is present,
which is the case for the redux_cards fixture). Mass-fraction axes are
graded via v2ecoli.library.card_criteria.grade_axis (rel_tol, 5%/10% bands).

Uses the redux_cards fixture (Task B0): 1 seed x 1 generation. rel_tol grades
a scalar (not a population stat) so this should render real graded verdicts,
not force ungraded, as long as both engines have protein/rna/dry mass data.

Goes through the Step round-trip (see test_report_card_trajectory.py's
docstring for why: ``@as_step`` returns a Step class, not a plain callable).
"""
from conftest import make_card_state

import scripts._compare.report_cards.composition  # noqa: F401  (registers the Step)
from _card_helpers import _run_card


def test_composition_card_emits_grouped_bar_and_graded_axes():
    out = _run_card("composition", make_card_state())
    html = out["card_html"].lower()
    assert "plotly" in html
    assert "bar" in html  # grouped-bar trace type marker in the Plotly JSON
    assert out["verdict"] in ("within_tol", "drift", "mismatch", "ungraded")
    assert out["axes"]
    labels = {a["label"] for a in out["axes"]}
    assert {"protein fraction", "rna fraction", "other fraction"} <= labels
