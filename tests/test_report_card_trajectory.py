"""B1: `trajectory` report card — interactive Plotly value-vs-time overlay of
v2ecoli vs vEcoli. Uses the redux_cards fixture (Task B0); reads the local
zarr stores directly (read_pbg_local), not state["observables"].

``@as_step`` turns ``update_trajectory_report_card`` into a Step CLASS (see
process_bigraph.composite.as_step) — the wrapped function is not directly
callable. Go through the same Step round-trip every other card test in this
repo uses (tests/_card_helpers.py::_run_card), also proving discoverability
via REPORT_CARD_STEPS / core.link_registry.
"""
from conftest import make_card_state

import scripts._compare.report_cards.trajectory  # noqa: F401  (registers the Step)
from _card_helpers import _run_card


def test_trajectory_card_emits_plotly_and_ungraded():
    out = _run_card("trajectory", make_card_state())
    assert "plotly" in out["card_html"].lower()
    assert out["verdict"] in ("ungraded", "within_tol", "drift", "mismatch")
    assert isinstance(out["axes"], list)
