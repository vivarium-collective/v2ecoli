"""Tests for scripts/sweep_report_xarray.py — pure render (CI-safe) + a
cache-gated full build from a real xarray sweep."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sweep_report_xarray as sx  # noqa: E402

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


def _prov(n_variants=1):
    return {
        "generated_at": "2026-05-30T12:00:00+00:00",
        "git_sha": "abc1234def5678", "git_short": "abc1234d",
        "git_branch": "feat/sweep-report-xarray", "git_dirty": True,
        "git_last_commit_msg": "test commit", "git_last_commit_author": "Eran",
        "git_last_commit_when": "2026-05-30 12:00:00 -0400",
        "host": "h", "platform": "Darwin 24 arm64", "python": "3.12.0",
        "script": "scripts/sweep_report_xarray.py", "sweep_dir": "/tmp/run",
        "n_variants": n_variants, "n_seeds": 2, "n_gens": 2, "n_cells": 2 * n_variants * 2,
    }


def test_render_html_xarray_labels_rna_and_notes_zarr():
    frac = {(0, 0, 1): {"protein": 0.47, "RNA": 0.20, "DNA": 0.02}}
    div_rows = [(0, 0, 1, 350.0, 660.0, 2400.0)]
    html = sx.render_html(_prov(), "PLOT1B64", "PLOT2B64", frac, div_rows)
    # shared provenance banner + docs (reused from sweep_report)
    assert 'class="provenance"' in html
    assert "abc1234d" in html and "DIRTY TREE" in html
    assert "How to run your own sweep" in html and "What this framework adds" in html
    # xarray-specific: RNA (total) not rRNA, and the zarr source note
    assert "<th>RNA</th>" in html and "<th>rRNA</th>" not in html
    assert "xarray / zarr" in html and "hive-partitioned zarr" in html
    assert "PLOT1B64" in html and "PLOT2B64" in html
    assert "0.470" in html


def test_variant_column_only_when_multiple_variants():
    frac = {(0, 0, 1): {"protein": 0.47, "RNA": 0.20, "DNA": 0.02}}
    div_rows = [(0, 0, 1, 350.0, 660.0, 2400.0)]
    assert "<th>variant</th>" not in sx.render_html(_prov(1), "a", "b", frac, div_rows)
    assert "<th>variant</th>" in sx.render_html(_prov(2), "a", "b", frac, div_rows)


def test_discover_experiment_id_parses_store_name(tmp_path):
    (tmp_path / "two_generations_xarray_v0_s1.zarr").mkdir()
    (tmp_path / "two_generations_xarray_v0_s0.zarr").mkdir()
    assert sx.discover_experiment_id(str(tmp_path)) == "two_generations_xarray"


@pytest.mark.skipif(not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_build_report_from_real_xarray_sweep(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from v2ecoli.workflow.run import run_workflow

    zarr_dir = str(tmp_path / "zarr")
    config = {
        "experiment_id": "xrep", "n_init_sims": 1, "generations": 1,
        "single_daughters": True, "cache_dir": CACHE, "out_dir": str(tmp_path),
        "variants": {}, "max_duration_per_gen": 5.0, "time_step": 1.0,
        "emitter": "xarray", "emitter_arg": {"out_dir": zarr_dir},
    }
    assert run_workflow(config, max_sim_time=20.0)["complete"] is True

    cells = sx.load_cells(zarr_dir)
    assert cells, "no cells loaded from zarr"
    # one lineage, generation 1, with dry_mass present
    (v, s, g) = next(iter(cells))
    assert "dry_mass" in cells[(v, s, g)]

    out = tmp_path / "report.html"
    latest, archive = sx.build_report(zarr_dir, out=str(out))
    assert latest.exists() and archive.exists()
    # build_report writes UTF-8 (it embeds unit symbols, em-dashes, etc.);
    # read it back as UTF-8 rather than the locale default (ASCII on CI).
    html = latest.read_text(encoding="utf-8")
    assert 'class="provenance"' in html and "xarray / zarr" in html
    assert "data:image/png;base64" in html
