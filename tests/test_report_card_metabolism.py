"""B3: `metabolism` report card — growth-rate v2-vs-vEcoli trace + biomass
(cell/dry mass) comparison for this condition, graded on the FINAL growth
rate via v2ecoli.library.card_criteria.grade_axis (rel_tol). Uses the
redux_cards fixture (Task B0): 1 seed x 1 generation, so the "final" growth
rate is just each engine's last emitted value — grade_axis's rel_tol branch
grades a scalar and doesn't need n>=2, so this should render a real
within_tol/drift/mismatch verdict (not force ungraded) on the fixture,
unless one side is missing data entirely.

There is no flux/exchange observable in the data (checked the fixture zarrs
directly, see tests/fixtures/redux_cards/README.md), so this card scopes to
growth-rate + biomass only; flux plots are a documented follow-up.

Goes through the Step round-trip (see test_report_card_trajectory.py's
docstring for why: ``@as_step`` returns a Step class, not a plain callable).
"""
from conftest import make_card_state

import scripts._compare.report_cards.metabolism  # noqa: F401  (registers the Step)
from _card_helpers import _run_card


def test_metabolism_card_emits_plotly_growth_and_biomass():
    out = _run_card("metabolism", make_card_state())
    html = out["card_html"].lower()
    assert "plotly" in html
    # both the growth-rate trace and the biomass grouped-bar should render
    assert "cell_mass" in html or "biomass" in html or "mass" in html
    assert out["verdict"] in ("within_tol", "drift", "mismatch", "ungraded")
    assert isinstance(out["axes"], list)
