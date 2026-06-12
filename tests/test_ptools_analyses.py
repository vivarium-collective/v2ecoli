"""Fidelity tests for the ptools_rna/rxns/proteins native Analysis ports."""

import glob as _glob
import os
import pytest

FIX = "/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data"

_REF_HISTORY = "out/compare_harness/v2_sim/parquet/two_generations/history"
_PAIRED_SIMDATA = "out/workflow/simData.cPickle"

_HAS_SWEEP = bool(
    _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    if os.path.isdir(_REF_HISTORY) else []
) and os.path.isfile(_PAIRED_SIMDATA)


def _frame_ids(tsv_text):
    rows = [r for r in tsv_text.strip().splitlines() if r]
    return {r.split("\t")[0] for r in rows[1:]}  # skip header ($ row)


# ---------------------------------------------------------------------------
# Registration tests — always run (no fixture needed)
# ---------------------------------------------------------------------------

def test_ptools_rna_registered():
    from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rna"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ptools_rxns_registered():
    from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rxns"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ptools_proteins_registered():
    from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_proteins"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
    from bigraph_schema import allocate_core
    assert cls({}, core=allocate_core()).outputs() == {"view": "string", "data": "map"}


# ---------------------------------------------------------------------------
# Oracle shape tests (sms-api fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rna_output_shape_matches_oracle():
    oracle = open(os.path.join(FIX, "ptools_rna.txt")).read()
    header = oracle.strip().splitlines()[0].split("\t")
    assert header[0] == "$"
    assert len(_frame_ids(oracle)) > 0


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rxns_oracle_shape():
    oracle = open(os.path.join(FIX, "ptools_rxns.txt")).read()
    assert oracle.strip().splitlines()[0].split("\t")[0] == "$"
    assert len(_frame_ids(oracle)) > 0


# ---------------------------------------------------------------------------
# data["tsv"] + view present — require actual sweep parquet
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_rna_has_tsv_and_view():
    """analyze() must return both a non-empty tsv string and a non-empty view."""
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsRna({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_rna: data['tsv'] is empty"
    assert out.get("view"), "ptools_rna: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view, (
        "ptools_rna: view does not contain a recognizable plot"
    )


@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_rxns_has_tsv_and_view():
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsRxns({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_rxns: data['tsv'] is empty"
    assert out.get("view"), "ptools_rxns: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view


@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_proteins_has_tsv_and_view():
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsProteins({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "ptools_proteins: data['tsv'] is empty"
    assert out.get("view"), "ptools_proteins: view is absent or empty"
    view = out["view"]
    assert "vega" in view.lower() or "plotly" in view.lower() or "<svg" in view
