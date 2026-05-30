"""Pure-render tests for reports/sweep_report.py — no parquet/cache needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reports"))
import sweep_report  # noqa: E402


def _prov(n_variants=1):
    return {
        "branch": "feat/x", "commit": "abc1234", "subject": "test", "date": "2026-05-29",
        "dirty": True, "generated": "2026-05-29 12:00", "host": "h", "python": "3.12.0",
        "sweep_dir": "/tmp/run", "n_variants": n_variants, "n_seeds": 2, "n_gens": 3,
        "n_cells": 2 * n_variants * 3, "seeds": [0, 1], "gens": [0, 1, 2],
    }


def _frac_and_rows():
    frac = {(0, 0, 0): {"protein": 0.47, "rRNA": 0.10, "DNA": 0.018}}
    div_rows = [(0, 0, 0, 380.0, 685.0, 2400.0)]
    return frac, div_rows


def test_render_html_has_provenance_and_docs():
    frac, div_rows = _frac_and_rows()
    html = sweep_report.render_html(_prov(), "PLOT1B64", "PLOT2B64", frac, div_rows)
    # provenance banner with git + PR
    assert 'class="prov"' in html
    assert "abc1234" in html and "feat/x" in html
    assert "uncommitted changes" in html          # dirty flag surfaced
    # both doc sections present
    assert "How to run your own sweep" in html
    assert "What this framework adds" in html
    assert "v2ecoli-workflow --config" in html     # CLI documented
    assert 'target": "ecoli-metabolism.kcat"' in html or "ecoli-metabolism.kcat" in html
    # plots + a data row embedded
    assert "PLOT1B64" in html and "PLOT2B64" in html
    assert "0.470" in html                          # protein fraction rendered


def test_variant_column_only_when_multiple_variants():
    frac, div_rows = _frac_and_rows()
    single = sweep_report.render_html(_prov(1), "a", "b", frac, div_rows)
    assert "<th>variant</th>" not in single
    multi = sweep_report.render_html(_prov(2), "a", "b", frac, div_rows)
    assert "<th>variant</th>" in multi


def test_howto_documents_seed_gen_and_variant_knobs():
    html = sweep_report.HOWTO_HTML
    for token in ("n_init_sims", "generations", "single_daughters", "variants",
                  "prod", "zip", "add", "linspace"):
        assert token in html, f"missing {token!r} in how-to docs"
