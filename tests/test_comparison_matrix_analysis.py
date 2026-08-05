"""Task 3 (comparison convergence Phase 2): comparison_matrix Analysis --
investigation-level configs x observables verdict matrix, reusing
``reports/_summary``'s matrix builder (``aggregate.py``'s matrix structure +
``render.py``'s ``_matrix_table``). Hermetic -- no engine run, no filesystem
reads; feeds synthetic ``report_card_verdict/v1`` dicts (the exact shape
``comparison_summary``/``comparison_cards`` emit) straight into the pure
function / registered Step.
"""
from __future__ import annotations

import re

import pytest

from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
from v2ecoli.workflow.analyses import comparison_matrix as comparison_matrix_mod
from v2ecoli.workflow.analyses.comparison_matrix import (
    ComparisonMatrix, _MatrixStructureError, comparison_matrix)


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


# --------------------------------------------------------------------------- #
# Fix-round 1 (Important finding): delta injection must be KEYED by
# (config, observable) identity parsed back out of the reused table's own
# header + study cells, NOT a flat positional pass over graded `<td>`s -- a
# future `_matrix_table` change that reorders cells (resorted columns, an
# inserted row) but keeps the same classes/count must not silently attach
# the wrong |Delta| to the wrong cell.
# --------------------------------------------------------------------------- #

def _row_cells(html: str, config_name: str) -> list[tuple[str, str]]:
    """[(css_class, cell_text), ...] for `config_name`'s row, in DOCUMENT
    order -- an independent extraction (not reusing the module's own
    parsing helpers) so this test can't be fooled by a bug shared between
    production code and its own verification."""
    m = re.search(
        r'<td class="study">' + re.escape(config_name) + r'</td>'
        r'((?:<td class="[\w-]+">.*?</td>)+)', html, re.DOTALL)
    assert m, f"no row found for config {config_name!r} in:\n{html}"
    return re.findall(r'<td class="([\w-]+)">(.*?)</td>', m.group(1), re.DOTALL)


def test_reordered_columns_still_map_correct_delta_to_correct_cell(monkeypatch):
    """Simulate a future `_matrix_table` that emits columns in a DIFFERENT
    order than `matrix["columns"]` (e.g. resorted alphabetically) -- header
    and each row's cells reordered TOGETHER, same classes/count as the real
    renderer would produce. `_inject_deltas` must still attach each delta to
    its correct (config, observable) cell by reading that order back off the
    table, not by assuming `matrix["columns"]`'s order.

    `reduced_media`'s two cells are both `verdict-within_tol` (4.0% growth
    rate, 3.0% cell mass) -- indistinguishable by class alone, so getting
    this one right proves the mapping is positional-within-row keyed by the
    header, not a lucky class-based guess. Under the pre-fix flat positional
    splice (deltas streamed in `matrix["columns"]` order: cell_mass then
    growth_rate) this reordered table would have received cell_mass's delta
    on the growth_rate cell and vice versa for every row.
    """
    reordered_html = (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        '<th>growth_rate</th><th>cell_mass</th></tr></thead><tbody>'
        '<tr><td class="study">baseline</td>'
        '<td class="verdict-drift">◐ Drift</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td></tr>'
        '<tr><td class="study">no_regulation</td>'
        '<td class="verdict-mismatch">✗ Mismatch</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td></tr>'
        '<tr><td class="study">reduced_media</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td></tr>'
        '</tbody></table>'
    )
    monkeypatch.setattr(
        comparison_matrix_mod._render, "_matrix_table",
        lambda summary: reordered_html)

    html = comparison_matrix(CONFIG_VERDICTS)["matrix_html"]

    for config, expect_growth, expect_mass in (
        ("baseline", "7.0%", "2.0%"),
        ("no_regulation", "15.0%", "1.0%"),
        ("reduced_media", "4.0%", "3.0%"),
    ):
        (growth_klass, growth_text), (mass_klass, mass_text) = _row_cells(html, config)
        assert expect_growth in growth_text, (config, growth_text)
        assert expect_mass not in growth_text, (config, growth_text)
        assert expect_mass in mass_text, (config, mass_text)
        assert expect_growth not in mass_text, (config, mass_text)


def test_matrix_table_change_that_breaks_row_structure_raises_not_mislabels(
        monkeypatch):
    """A row whose cell count no longer matches the header (a genuine
    structural desync -- e.g. an inserted rollup column/row) must fail loud,
    not silently attach a delta to the wrong cell."""
    broken_html = (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        '<th>cell_mass</th><th>growth_rate</th></tr></thead><tbody>'
        '<tr><td class="study">baseline</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td></tr>'
        '</tbody></table>'
    )
    monkeypatch.setattr(
        comparison_matrix_mod._render, "_matrix_table",
        lambda summary: broken_html)

    with pytest.raises(_MatrixStructureError):
        comparison_matrix(CONFIG_VERDICTS)


def test_matrix_table_row_for_unknown_config_raises_not_mislabels(monkeypatch):
    """A row whose study cell doesn't match any known config (e.g. a
    transposed table, where rows are now observables) must fail loud rather
    than guess which config's deltas belong on it."""
    transposed_like_html = (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        '<th>baseline</th></tr></thead><tbody>'
        '<tr><td class="study">cell_mass</td>'
        '<td class="verdict-within_tol">✓ Within tolerance</td></tr>'
        '</tbody></table>'
    )
    monkeypatch.setattr(
        comparison_matrix_mod._render, "_matrix_table",
        lambda summary: transposed_like_html)

    with pytest.raises(_MatrixStructureError):
        comparison_matrix(CONFIG_VERDICTS)
