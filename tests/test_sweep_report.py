"""Pure-render tests for scripts/sweep_report.py — no parquet/cache needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sweep_report  # noqa: E402


def _prov(n_variants=1, dirty=True):
    return {
        "generated_at": "2026-05-29T12:00:00+00:00",
        "git_sha": "abc1234def5678", "git_short": "abc1234d",
        "git_branch": "feat/workflow-framework", "git_dirty": dirty,
        "git_last_commit_msg": "test commit", "git_last_commit_author": "Eran",
        "git_last_commit_when": "2026-05-29 12:00:00 -0400",
        "host": "h", "platform": "Darwin 24 arm64", "python": "3.12.0",
        "script": "scripts/sweep_report.py",
        "sweep_dir": "/tmp/run", "n_variants": n_variants, "n_seeds": 2,
        "n_gens": 3, "n_cells": 2 * n_variants * 3,
    }


def _frac_and_rows():
    frac = {(0, 0, 0): {"protein": 0.47, "rRNA": 0.10, "DNA": 0.018}}
    div_rows = [(0, 0, 0, 380.0, 685.0, 2400.0)]
    return frac, div_rows


def test_provenance_banner_has_required_fields():
    frac, div_rows = _frac_and_rows()
    html = sweep_report.render_html(_prov(), "PLOT1B64", "PLOT2B64", frac, div_rows)
    # AGENTS.md required provenance fields
    assert "2026-05-29T12:00:00+00:00" in html                 # ISO-8601 timestamp
    assert "abc1234d" in html and "abc1234def5678" in html      # short + full SHA
    assert f"{sweep_report.GITHUB_REPO}/commit/abc1234def5678" in html  # GitHub link
    assert "feat/workflow-framework" in html                   # branch
    assert "DIRTY TREE" in html                                # dirty badge
    assert "test commit — Eran" in html                        # last commit msg+author
    assert "scripts/sweep_report.py" in html                   # generator path
    assert "Python 3.12.0" in html                             # host/python


def test_clean_tree_has_no_dirty_badge():
    frac, div_rows = _frac_and_rows()
    html = sweep_report.render_html(_prov(dirty=False), "a", "b", frac, div_rows)
    assert "DIRTY TREE" not in html


def test_docs_and_plots_present():
    frac, div_rows = _frac_and_rows()
    html = sweep_report.render_html(_prov(), "PLOT1B64", "PLOT2B64", frac, div_rows)
    assert "How to run your own sweep" in html
    assert "What this framework adds" in html
    assert "v2ecoli-workflow --config" in html
    assert "ecoli-metabolism.kcat" in html
    assert "PLOT1B64" in html and "PLOT2B64" in html
    assert "0.470" in html


def test_variant_column_only_when_multiple_variants():
    frac, div_rows = _frac_and_rows()
    assert "<th>variant</th>" not in sweep_report.render_html(_prov(1), "a", "b", frac, div_rows)
    assert "<th>variant</th>" in sweep_report.render_html(_prov(2), "a", "b", frac, div_rows)


def test_collect_provenance_real_fields():
    prov = sweep_report.collect_provenance(extra={"n_cells": 6})
    for key in ("generated_at", "git_sha", "git_short", "git_branch", "git_dirty",
                "git_last_commit_msg", "host", "platform", "python", "script"):
        assert key in prov
    assert prov["script"] == "scripts/sweep_report.py"
    assert prov["n_cells"] == 6


def test_howto_documents_seed_gen_and_variant_knobs():
    for token in ("n_init_sims", "generations", "single_daughters", "variants",
                  "prod", "zip", "add", "linspace"):
        assert token in sweep_report.HOWTO_HTML, f"missing {token!r}"
