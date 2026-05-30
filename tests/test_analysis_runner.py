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
