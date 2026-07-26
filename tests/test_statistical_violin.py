"""B5: enable violin rendering on the existing `statistical` report card.

`scripts/_compare/report_card_section.py::CARD_AXES` and
`scripts/comparison_report_card.py::EXTRA_AXES` gain `"plot": "violin"` so
`v2ecoli.library.report_card._axis_plot_svg` renders each axis via
`v2ecoli.library.card_plots.violin_strip` (a matplotlib violinplot → inline
SVG; its body traces are matplotlib `PolyCollection` elements, a stable
marker string in the emitted SVG — see `card_plots.violin_strip` +
`_axis_plot_svg`'s `kind == "violin"` branch in v2ecoli/library/report_card.py).

The redux_cards fixture's `state["observables"]` is `{}` (see
tests/conftest.py::make_card_state and its README) — the `statistical` card
reads ONLY `state["observables"]`, so on this fixture every axis's per-cell
`values` list is empty, and `_axis_plot_svg`'s `measured.get("values")`
truthy-guard means violin_strip is never called (no data to plot) even
though `plot="violin"` is wired through. So this test has two parts:

  1. wiring: CARD_AXES/EXTRA_AXES all declare plot="violin" (the actual
     change), and the statistical card still renders on the thin fixture
     without crashing.
  2. real render: build_report_card called directly with synthetic
     non-empty per-cell data (bypassing the thin fixture) proves the violin
     render path is genuinely exercised end-to-end, asserting the
     PolyCollection marker appears in the HTML.
"""
from conftest import make_card_state

import scripts._compare.report_cards.statistical  # noqa: F401  (registers the Step)
from _card_helpers import _run_card


def test_statistical_card_axes_declare_violin_plot():
    from scripts._compare.report_card_section import CARD_AXES
    from scripts.comparison_report_card import EXTRA_AXES

    assert CARD_AXES
    assert all(spec.get("plot") == "violin" for spec in CARD_AXES)
    assert EXTRA_AXES
    assert all(spec.get("plot") == "violin" for spec in EXTRA_AXES)


def test_statistical_card_renders_on_thin_fixture_without_crashing():
    out = _run_card("statistical", make_card_state())
    assert out["verdict"] in ("within_tol", "drift", "mismatch", "ungraded")
    assert isinstance(out["axes"], list)


def test_statistical_card_violin_marker_renders_with_real_data():
    from scripts._compare.report_card_section import build_report_card

    left = {"cell_mass": [100.0, 102.0, 98.0, 101.0],
            "growth_rate": [0.0003, 0.00031, 0.00029, 0.0003]}
    right = {"cell_mass": [101.0, 103.0, 99.0, 100.0],
             "growth_rate": [0.00031, 0.0003, 0.0003, 0.00029]}
    _, html = build_report_card(left, right, model_ref="unit-test")
    assert "PolyCollection" in html
