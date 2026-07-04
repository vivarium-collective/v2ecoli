"""Report rendering: dedicated ParCa prelude + provenance stamps.

Guards the report-structure work that keeps a comparison report self-describing
and always-current:
  - the ParCa fit comparison renders ONCE as a top-level prelude (not repeated
    inside every condition group),
  - the top nav distinguishes report-level sections from per-condition runs,
  - the report stamps when it was generated and when each condition last ran,
    so a stale report is visible at a glance.
"""
from scripts._compare.report import render_grouped_report


def _groups():
    return [
        {"config": "cond_basal", "config_id": "basal",
         "run_mtime": "2026-06-20 14:51",
         "sections": [{"title": "2-gen sim dynamics",
                       "rows": [{"label": "cell_mass", "left": "1",
                                 "right": "1", "verdict": "within_tol"}]}],
         "embedded_html": []},
        {"config": "cond_acetate", "config_id": "acetate",
         "run_mtime": "2026-06-20 14:51",
         "sections": [{"title": "2-gen sim dynamics",
                       "rows": [{"label": "cell_mass", "left": "1",
                                 "right": "2", "verdict": "mismatch"}]}]},
    ]


_PRELUDE = {"title": "ParCa / sim_data",
            "rows": [{"label": "RNA expression — acetate", "left": "a",
                      "right": "a", "verdict": "within_tol",
                      "group": "distributions"}]}


def test_parca_prelude_rendered_once_at_top():
    html = render_grouped_report(_groups(), title="t", prelude=_PRELUDE)
    assert 'id="parca-fit"' in html
    assert "ParCa fit (condition-independent)" in html
    # the prelude content appears exactly once (not duplicated per condition)
    assert html.count("RNA expression — acetate") == 1


def test_nav_distinguishes_sections_from_conditions():
    html = render_grouped_report(_groups(), title="t", prelude=_PRELUDE)
    assert 'class="nav-top" href="#overview"' in html      # report-level
    assert 'class="nav-top" href="#parca-fit"' in html
    assert '<span class="navlabel">Conditions</span>' in html
    assert '#cfg-basal' in html and '#cfg-acetate' in html  # per-condition


def test_report_stamps_generation_and_run_freshness():
    html = render_grouped_report(
        _groups(), title="t", prelude=_PRELUDE,
        generated_at="2026-06-20 15:10")
    assert "generated 2026-06-20 15:10" in html          # report generation
    # each condition shows when its sims last ran
    assert html.count("sims last ran: 2026-06-20 14:51") == 2


def test_generation_stamp_absent_when_not_passed():
    # the generation stamp only appears when generated_at is supplied;
    # per-condition run_mtime is independent (rendered whenever a group has it)
    html = render_grouped_report(_groups(), title="t")
    assert "generated 20" not in html


def test_run_freshness_absent_when_group_has_no_mtime():
    groups = [{"config": "x", "config_id": "x", "sections": [], "embedded_html": []}]
    html = render_grouped_report(groups, title="t")
    assert "sims last ran:" not in html
