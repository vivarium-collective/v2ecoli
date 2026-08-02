"""Phase B Task 4 (Gap 2): ``comparison_matrix`` assembles per-config
``config_verdicts`` by reading each config study's persisted verdict from
disk, keyed by study slug, instead of requiring an explicit in-memory
``config_verdicts`` dict.

Hermetic -- fixture verdict JSON files are written under a tmp workspace's
``studies/<slug>/report_card_verdict.json`` (the Gen-1 convention
``scripts/_compare/verdict.py::write_condition_verdict`` writes -- see
``comparison_matrix.py``'s module-level Gap 2 note for why this is the
canonical location this loader reads).
"""
from __future__ import annotations

import json

from v2ecoli.workflow.analyses.comparison_matrix import (
    ComparisonMatrix, comparison_matrix)


def _write_verdict(workspace, slug, overall="within_tol", median_rel=0.01):
    study_dir = workspace / "studies" / slug
    study_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "schema": "report_card_verdict/v1",
        "overall": overall,
        "groups": {
            "mass": {
                "verdict": overall,
                "axes": [
                    {"label": "cell_mass", "verdict": overall,
                     "detail": {"median_rel": median_rel}},
                ],
            },
        },
    }
    (study_dir / "report_card_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8")
    return verdict


def test_comparison_matrix_assembles_config_verdicts_from_disk(tmp_path):
    _write_verdict(tmp_path, "basal", overall="within_tol", median_rel=0.02)
    _write_verdict(tmp_path, "with_aa", overall="drift", median_rel=0.08)

    out = comparison_matrix(config_studies=["basal", "with_aa"],
                             workspace=tmp_path)

    assert set(out) == {"matrix_html"}
    html = out["matrix_html"]
    assert "basal" in html
    assert "with_aa" in html
    assert "cell_mass" in html
    assert "✓" in html and "Within tolerance" in html
    assert "◐" in html and "Drift" in html
    assert "2.0%" in html
    assert "8.0%" in html


def test_comparison_matrix_missing_config_verdict_renders_placeholder_not_crash(
        tmp_path):
    _write_verdict(tmp_path, "basal", overall="within_tol", median_rel=0.01)
    # "with_aa" has NO report_card_verdict.json on disk at all.

    out = comparison_matrix(config_studies=["basal", "with_aa"],
                             workspace=tmp_path)

    html = out["matrix_html"]
    assert "basal" in html
    assert "with_aa" in html
    # the missing config's row has no graded cells -> verdict-none, not a
    # fabricated grade and not a raised exception.
    assert 'class="verdict-none"' in html


def test_comparison_matrix_config_studies_workspace_not_found_all_placeholder(
        tmp_path):
    """Neither config has a verdict file on disk (e.g. the persistence step
    hasn't run yet) -- no crash. With zero graded axes across every config
    there are no matrix columns, so the reused ``_matrix_table`` renders no
    table body (its own documented empty-columns behavior) -- the important
    assertion is that assembling from disk with nothing found degrades
    gracefully rather than raising."""
    out = comparison_matrix(config_studies=["basal", "with_aa"],
                             workspace=tmp_path)
    assert set(out) == {"matrix_html"}
    assert "matrix-legend" in out["matrix_html"]


def test_comparison_matrix_explicit_config_verdicts_still_works_without_config_studies():
    """Backward compatibility: the original explicit-dict call shape (no
    config_studies/workspace) is untouched."""
    config_verdicts = {
        "baseline": {
            "schema": "report_card_verdict/v1",
            "overall": "within_tol",
            "groups": {
                "mass": {
                    "verdict": "within_tol",
                    "axes": [{"label": "cell_mass", "verdict": "within_tol",
                              "detail": {"median_rel": 0.01}}],
                },
            },
        },
    }
    out = comparison_matrix(config_verdicts)
    html = out["matrix_html"]
    assert "baseline" in html
    assert "cell_mass" in html
    assert "1.0%" in html


def test_comparison_matrix_config_studies_takes_precedence_over_explicit_config_verdicts(
        tmp_path):
    """On the real composite substrate, InvestigationAnalysisStep always
    hands this Analysis a non-empty (but non-verdict-shaped) config_verdicts
    dict alongside the static config_studies param -- config_studies must
    still win, or the disk-read path would be dead code in production (see
    comparison_matrix.py's module docstring on comparison_matrix())."""
    _write_verdict(tmp_path, "basal", overall="mismatch", median_rel=0.5)

    bogus_config_verdicts = {"basal": {"study": "basal", "ran": True}}

    out = comparison_matrix(bogus_config_verdicts,
                             config_studies=["basal"], workspace=tmp_path)
    html = out["matrix_html"]
    assert "✗" in html and "Mismatch" in html
    assert "50.0%" in html


def test_comparison_matrix_analysis_step_reads_from_disk(tmp_path):
    from v2ecoli.core import build_core

    _write_verdict(tmp_path, "basal", overall="within_tol", median_rel=0.03)

    step = ComparisonMatrix(
        config={"config_studies": ["basal"], "workspace": str(tmp_path)},
        core=build_core())
    out = step.update()
    assert set(out) == {"matrix_html"}
    assert "basal" in out["matrix_html"]
    assert "3.0%" in out["matrix_html"]
