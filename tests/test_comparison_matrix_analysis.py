"""Task 3 (comparison convergence Phase 2): comparison_matrix Analysis --
investigation-level configs x observables verdict matrix, reusing
``reports/_summary``'s matrix builder (``aggregate.py``'s matrix structure +
``render.py``'s ``_matrix_table``). Hermetic -- no engine run, no filesystem
reads; feeds synthetic ``report_card_verdict/v1`` dicts (the exact shape
``comparison_summary``/``comparison_cards`` emit) straight into the pure
function / registered Step.
"""
from __future__ import annotations

from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
from v2ecoli.workflow.analyses.comparison_matrix import (
    ComparisonMatrix, comparison_matrix)


def _verdict(overall: str, axes: list[dict]) -> dict:
    return {
        "schema": "report_card_verdict/v1",
        "overall": overall,
        "groups": {"observables": {"verdict": overall, "axes": axes}},
    }


# Three configs x two observables, known verdicts spanning all three bands.
CONFIG_VERDICTS = {
    "baseline": _verdict("drift", [
        {"label": "cell_mass", "verdict": "within_tol",
         "detail": {"median_rel": 0.02}},
        {"label": "growth_rate", "verdict": "drift",
         "detail": {"median_rel": 0.07}},
    ]),
    "no_regulation": _verdict("mismatch", [
        {"label": "cell_mass", "verdict": "within_tol",
         "detail": {"median_rel": 0.01}},
        {"label": "growth_rate", "verdict": "mismatch",
         "detail": {"median_rel": 0.15}},
    ]),
    "reduced_media": _verdict("within_tol", [
        {"label": "cell_mass", "verdict": "within_tol",
         "detail": {"median_rel": 0.03}},
        {"label": "growth_rate", "verdict": "within_tol",
         "detail": {"median_rel": 0.04}},
    ]),
}


# --------------------------------------------------------------------------- #
# registration -- same mechanism as the other comparison Analyses
# --------------------------------------------------------------------------- #

def test_comparison_matrix_is_registered_like_the_other_comparison_analyses():
    from v2ecoli.workflow.analyses.comparison_summary import ComparisonSummary

    assert issubclass(ComparisonMatrix, Analysis)
    assert ANALYSIS_REGISTRY["comparison_matrix"] is ComparisonMatrix
    assert ANALYSIS_REGISTRY["comparison_summary"] is ComparisonSummary


def test_comparison_matrix_discoverable_via_analyses_package_import():
    import importlib

    import v2ecoli.workflow.analyses as analyses_pkg
    importlib.reload(analyses_pkg)
    assert "comparison_matrix" in ANALYSIS_REGISTRY


# --------------------------------------------------------------------------- #
# comparison_matrix() -- matrix content
# --------------------------------------------------------------------------- #

def test_comparison_matrix_renders_a_cell_per_config_and_observable():
    out = comparison_matrix(CONFIG_VERDICTS)
    assert set(out) == {"matrix_html"}
    html = out["matrix_html"]

    # every config (row) and every observable (column) appears
    for config in CONFIG_VERDICTS:
        assert config in html
    assert "cell_mass" in html
    assert "growth_rate" in html

    # status is glyph+label, never color-alone (scripts/_compare/theme.STATUS)
    assert "✓" in html and "Within tolerance" in html   # within_tol
    assert "◐" in html and "Drift" in html               # drift
    assert "✗" in html and "Mismatch" in html            # mismatch

    # right verdict class per graded cell (status fill)
    assert html.count('class="verdict-within_tol"') == 4  # baseline+no_reg+reduced cell_mass, reduced growth_rate
    assert html.count('class="verdict-drift"') == 1       # baseline growth_rate
    assert html.count('class="verdict-mismatch"') == 1    # no_regulation growth_rate

    # median-|delta| percent alongside each graded cell
    assert "2.0%" in html    # baseline cell_mass
    assert "7.0%" in html    # baseline growth_rate
    assert "1.0%" in html    # no_regulation cell_mass
    assert "15.0%" in html   # no_regulation growth_rate
    assert "3.0%" in html    # reduced_media cell_mass
    assert "4.0%" in html    # reduced_media growth_rate

    # legend present: one glyph+label per graded status
    assert "matrix-legend" in html


def test_comparison_matrix_handles_a_config_missing_an_observable():
    """A config that never graded an observable (no axis with that label)
    gets a verdict-none cell, not a fabricated verdict or a crash."""
    verdicts = dict(CONFIG_VERDICTS)
    verdicts["metabolism_only"] = _verdict("within_tol", [
        {"label": "cell_mass", "verdict": "within_tol",
         "detail": {"median_rel": 0.01}},
        # no growth_rate axis at all
    ])
    out = comparison_matrix(verdicts)
    html = out["matrix_html"]
    assert "metabolism_only" in html
    assert 'class="verdict-none"' in html


def test_comparison_matrix_analysis_step_wraps_the_function():
    from v2ecoli.core import build_core

    step = ComparisonMatrix(
        config={"config_verdicts": CONFIG_VERDICTS}, core=build_core())
    out = step.update()
    assert set(out) == {"matrix_html"}
    assert "growth_rate" in out["matrix_html"]
    assert "baseline" in out["matrix_html"]
