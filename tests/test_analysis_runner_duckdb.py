from v2ecoli.workflow.analysis_runner import scale_history_sql

_FROM = "read_parquet(['x.pq'], hive_partitioning=true)"


def test_single_sql_filters_full_cell():
    sql = scale_history_sql("single", _FROM, (0, 1, 2, "00"))
    assert "variant = 0" in sql and "lineage_seed = 1" in sql
    assert "generation = 2" in sql and "agent_id = '00'" in sql


def test_multiseed_sql_filters_variant_only():
    sql = scale_history_sql("multiseed", _FROM, (3,))
    assert "variant = 3" in sql
    assert "lineage_seed" not in sql and "agent_id" not in sql


def test_multivariant_sql_is_unfiltered():
    sql = scale_history_sql("multivariant", _FROM, ())
    assert "WHERE" not in sql.upper()
    assert _FROM in sql


import glob as _glob
import json as _json
import os as _os
import pytest as _pytest


def _ref_history_dir():
    d = "out/compare_harness/v2_sim/parquet/two_generations/history"
    return d if _glob.glob(_os.path.join(d, "**", "*.pq"), recursive=True) else None


_PAIRED_SIMDATA = "out/workflow/simData.cPickle"


@_pytest.mark.skipif(_ref_history_dir() is None or not _os.path.isfile(_PAIRED_SIMDATA),
                     reason="reference sweep parquet or paired sim_data absent")
def test_proving_set_end_to_end(tmp_path):
    # Build a self-contained sweep dir: history parquet (symlinked) + the PAIRED
    # sim_data at the sweep root, so resolve_sim_data finds the correct one.
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.workflow.analysis_runner import run_analyses
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "history").symlink_to(_os.path.abspath(_ref_history_dir()))
    (sweep / "simData.cPickle").symlink_to(_os.path.abspath(_PAIRED_SIMDATA))

    opts = {
        "single": {"ptools_rna": {"n_tp": 8}, "ptools_rxns": {"n_tp": 8}},
        "multiseed": {"central_carbon_metabolism_scatter": {}},
    }
    res = run_analyses(str(sweep), opts)

    # data product present (not a recorded error) for a single-scale data analysis
    rna = res["single"]["ptools_rna"]
    assert rna, "ptools_rna produced no per-group result"
    first = next(iter(rna.values()))
    assert "error" not in first, f"ptools_rna errored: {first}"
    assert first.get("filename") == "ptools_rna.tsv"

    # reaction analysis did NOT trip the pairing assertion (correct sim_data)
    rxns = res["single"]["ptools_rxns"]
    first_rx = next(iter(rxns.values()))
    assert "error" not in first_rx, f"ptools_rxns errored (pairing?): {first_rx}"

    # multiseed VIEW analysis wrote a viz html
    assert res["multiseed"]["central_carbon_metabolism_scatter"]
    viz = _glob.glob(str(sweep / "viz" / "central_carbon_metabolism_scatter*.html"))
    assert viz, "no viz html written for ccm_scatter"

    # analysis.json written
    assert _os.path.isfile(str(sweep / "analysis.json"))
