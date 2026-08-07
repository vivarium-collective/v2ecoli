import json
import os

import pytest

from v2ecoli.workflow.analysis_runner import group_for_scale


def _recs():
    return [
        {"variant": 0, "lineage_seed": 0, "generation": 0, "agent_id": "0"},
        {"variant": 0, "lineage_seed": 0, "generation": 1, "agent_id": "00"},
        {"variant": 0, "lineage_seed": 1, "generation": 0, "agent_id": "0"},
        {"variant": 1, "lineage_seed": 0, "generation": 0, "agent_id": "0"},
    ]


def test_group_single_is_per_cell():
    assert len(group_for_scale("single", _recs())) == 4


def test_group_multigeneration_by_lineage():
    g = group_for_scale("multigeneration", _recs())
    assert (0, 0) in g and len(g[(0, 0)]) == 2
    assert (0, 1) in g and (1, 0) in g


def test_group_multiseed_by_variant():
    g = group_for_scale("multiseed", _recs())
    assert set(g) == {(0,), (1,)}
    assert len(g[(0,)]) == 3


def test_group_multivariant_is_all():
    g = group_for_scale("multivariant", _recs())
    assert set(g) == {()} and len(g[()]) == 4


def test_group_multidaughter_by_parent():
    g = group_for_scale("multidaughter", _recs())
    assert any(k[3] == "0" for k in g)


def test_run_analyses_over_synthetic_records(monkeypatch):
    import v2ecoli.workflow.analysis_runner as ar
    recs = {
        (0, 0, 0, "0"): {"variant": 0, "lineage_seed": 0, "generation": 0, "agent_id": "0",
                         "divided": True, "division_time": 2400.0,
                         "newborn_dry_mass": 380.0, "final_dry_mass": 700.0,
                         "timeseries": [{"listeners": {"mass": {"dry_mass": 380.0,
                            "protein_mass": 180.0, "rRna_mass": 38.0, "dna_mass": 7.0}}}]},
        (0, 1, 0, "0"): {"variant": 0, "lineage_seed": 1, "generation": 0, "agent_id": "0",
                         "divided": True, "division_time": 2600.0,
                         "newborn_dry_mass": 382.0, "final_dry_mass": 710.0,
                         "timeseries": [{"listeners": {"mass": {"dry_mass": 382.0,
                            "protein_mass": 190.0, "rRna_mass": 40.0, "dna_mass": 7.0}}}]},
    }
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: recs)
    import tempfile
    d = tempfile.mkdtemp()
    options = {"single": {"mass_fraction_summary": {}},
               "multiseed": {"doubling_time_distribution": {}}}
    results = ar.run_analyses(d, options)
    assert len(results["single"]["mass_fraction_summary"]) == 2
    ms = list(results["multiseed"]["doubling_time_distribution"].values())[0]
    assert ms["n_cells"] == 2 and abs(ms["doubling_time_mean"] - 2500.0) < 1e-9
    assert os.path.isfile(os.path.join(d, "analysis.json"))


def test_run_analyses_unknown_name_skips(monkeypatch):
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: {})
    import tempfile
    out = ar.run_analyses(tempfile.mkdtemp(), {"single": {"nope_not_real": {}}})
    assert out["single"] == {}


_CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


@pytest.mark.skipif(not os.path.isdir(_CACHE), reason="ParCa cache not present")
def test_run_workflow_runs_analyses_end_to_end(tmp_path):
    from v2ecoli.workflow.run import run_workflow
    out = str(tmp_path / "parquet")
    config = {
        "experiment_id": "anlz", "n_init_sims": 2, "generations": 1,
        "single_daughters": True, "cache_dir": _CACHE, "out_dir": out,
        "variants": {}, "max_duration_per_gen": 5.0, "time_step": 1.0,
        "analysis_options": {
            "single": {"mass_fraction_summary": {}},
            "multiseed": {"doubling_time_distribution": {}},
        },
    }
    result = run_workflow(config, max_sim_time=30.0)
    assert result["complete"] is True
    assert os.path.isfile(os.path.join(out, "summary.json"))
    assert os.path.isfile(os.path.join(out, "analysis.json"))
    with open(os.path.join(out, "analysis.json")) as f:
        analysis = json.load(f)
    assert len(analysis["single"]["mass_fraction_summary"]) == 2
    assert analysis["multiseed"]["doubling_time_distribution"]


def test_cli_main_runs(monkeypatch, tmp_path, capsys):
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: {})
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"analysis_options": {"single": {"mass_fraction_summary": {}}}}')
    monkeypatch.setattr("sys.argv", ["v2ecoli-analyze", str(tmp_path), "--config", str(cfg)])
    ar.main()
    assert os.path.isfile(str(tmp_path / "analysis.json"))
    assert "analysis.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# S3-resident sweeps + lazy record building
#
# A 1000-seed lineage sweep is terabytes of hive parquet — it never lands on one
# node, so the runner has to read it in place and must not materialize per-cell
# timeseries unless an analysis actually consumes them.
# ---------------------------------------------------------------------------


def test_is_s3_uri_discriminates():
    from v2ecoli.workflow.analysis_runner import is_s3_uri

    assert is_s3_uri("s3://bucket/sweep")
    assert not is_s3_uri("/local/sweep")
    assert not is_s3_uri("out/batch_baseline")


def _write_cell(root, seed, gen, agent, name="400.pq"):
    import polars as pl

    d = (root / "history" / "experiment_id=e" / "variant=0"
         / f"lineage_seed={seed}" / f"generation={gen}" / f"agent_id={agent}")
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"global_time": [1.0]}).write_parquet(d / name)
    return d


def test_cell_keys_reads_partitions_without_opening_parquet(tmp_path):
    """Group enumeration is a LISTING — the four keys are all in the path."""
    from v2ecoli.workflow.analysis_runner import cell_keys

    _write_cell(tmp_path, 0, 0, "0")
    _write_cell(tmp_path, 0, 1, "00")
    _write_cell(tmp_path, 7, 3, "000")
    # a second tick-file for an existing cell must not double-count it
    _write_cell(tmp_path, 7, 3, "000", name="800.pq")

    keys = cell_keys(str(tmp_path))
    assert len(keys) == 3
    assert {k["lineage_seed"] for k in keys} == {0, 7}
    assert {k["generation"] for k in keys} == {0, 1, 3}
    assert all(set(k) == {"variant", "lineage_seed", "generation", "agent_id"}
               for k in keys)


def test_cell_keys_ignores_partial_tmp_writes(tmp_path):
    """``*.pq.tmp`` files are orphaned partial writes from an interrupted run.

    Reading one would feed a truncated table into an analysis, so a cell that
    only ever produced a .tmp is not a cell.
    """
    from v2ecoli.workflow.analysis_runner import cell_keys, history_files

    _write_cell(tmp_path, 0, 0, "0")
    d = _write_cell(tmp_path, 1, 0, "0")
    (d / "400.pq").rename(d / "400.pq.tmp")

    assert len(history_files(str(tmp_path))) == 1
    assert [k["lineage_seed"] for k in cell_keys(str(tmp_path))] == [0]


def test_duckdb_only_analyses_skip_timeseries_extraction(tmp_path, monkeypatch):
    """A DuckDB-backed Analysis must not trigger per-cell record building."""
    import v2ecoli.workflow.analysis_runner as ar

    _write_cell(tmp_path, 0, 0, "0")

    def _boom(*a, **k):
        raise AssertionError("build_cell_records must not run for DuckDB analyses")

    monkeypatch.setattr(ar, "build_cell_records", _boom)
    # Pin the sim_data lookup to "absent" so the assertion below doesn't depend
    # on whether the developer happens to have a ParCa build in the environment.
    monkeypatch.delenv("V2ECOLI_SIM_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    # central_carbon_metabolism_scatter is an Analysis (DuckDB) subclass. With
    # no sim_data present the run fails resolving it — and that is the point:
    # reaching sim_data resolution proves record building was skipped, since
    # _boom would have fired first otherwise.
    with pytest.raises(FileNotFoundError, match="no sim_data pickle"):
        ar.run_analyses(
            str(tmp_path),
            {"multiseed": {"central_carbon_metabolism_scatter": {}}})


def test_record_based_analyses_still_build_records(tmp_path, monkeypatch):
    """The record-based AnalysisStep family still gets its timeseries."""
    import v2ecoli.workflow.analysis_runner as ar

    _write_cell(tmp_path, 0, 0, "0")
    called = {}

    def _fake(sweep_dir):
        called["yes"] = True
        return {}

    monkeypatch.setattr(ar, "build_cell_records", _fake)
    # doubling_time_distribution is a record-based AnalysisStep (multiseed)
    ar.run_analyses(str(tmp_path), {"multiseed": {"doubling_time_distribution": {}}})
    assert called.get("yes")


def test_s3_sweep_requires_an_out_dir(tmp_path):
    """An s3:// sweep is read-only, so results need somewhere local to land."""
    import v2ecoli.workflow.analysis_runner as ar

    with pytest.raises(ValueError, match="out_dir is required"):
        ar.run_analyses(
            "s3://bucket/sweep",
            {"multiseed": {"central_carbon_metabolism_scatter": {}}})
