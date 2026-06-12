"""Tests for the ptools_overview Cellular-Overview launcher Analysis."""

import glob as _glob
import os
import pytest

from v2ecoli.workflow.analyses.ptools_overview import (
    to_omics_dataset,
    build_mixture,
    cell_overview_url,
    overview_view_html,
)

FIX = "/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data"

_REF_HISTORY = "out/compare_harness/v2_sim/parquet/two_generations/history"
_PAIRED_SIMDATA = "out/workflow/simData.cPickle"
_HAS_SWEEP = bool(
    _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    if os.path.isdir(_REF_HISTORY) else []
) and os.path.isfile(_PAIRED_SIMDATA)

# A tiny synthetic ptools TSV (same shape the sibling analyses emit).
_TSV = "$\tt0\tt1\nEG10001\t0.0000\t0.4016\nEG10002\t2.0000\t1.3365\n"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_ptools_overview_registered():
    from v2ecoli.workflow.analyses import ptools_overview  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    from bigraph_schema import allocate_core

    cls = ANALYSIS_REGISTRY["ptools_overview"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
    assert cls({}, core=allocate_core()).outputs() == {"view": "string", "data": "map"}


# ---------------------------------------------------------------------------
# to_omics_dataset
# ---------------------------------------------------------------------------

def test_to_omics_dataset_comments_header_and_keeps_rows():
    out = to_omics_dataset(_TSV, "genes", "RNA (ptools_rna)")
    lines = out.splitlines()
    # The $ header is gone as a data row; documented as comments instead.
    assert not any(ln.startswith("$") for ln in lines)
    assert any(ln.startswith("# entity type: genes") for ln in lines)
    assert any("t0, t1" in ln for ln in lines)  # timepoint columns documented
    # Data rows preserved verbatim.
    assert "EG10001\t0.0000\t0.4016" in out
    assert "EG10002\t2.0000\t1.3365" in out


def test_to_omics_dataset_data_rows_have_no_dollar():
    out = to_omics_dataset(_TSV, "reactions")
    data_rows = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    assert data_rows and all(not ln.startswith("$") for ln in data_rows)
    # Every data row keeps its 2 timepoint values.
    assert all(len(ln.split("\t")) == 3 for ln in data_rows)


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_to_omics_dataset_on_oracle_preserves_all_frame_ids():
    oracle = open(os.path.join(FIX, "ptools_rna.txt")).read()
    src_ids = {ln.split("\t")[0] for ln in oracle.strip().splitlines()[1:] if ln}
    out = to_omics_dataset(oracle, "genes")
    out_ids = {ln.split("\t")[0] for ln in out.splitlines()
               if ln and not ln.startswith("#")}
    assert src_ids and out_ids == src_ids
    assert "EG10002" in out_ids


# ---------------------------------------------------------------------------
# build_mixture
# ---------------------------------------------------------------------------

def test_build_mixture_unions_rows_across_types():
    ds = [
        {"entity_type": "genes", "label": "RNA", "tsv": _TSV},
        {"entity_type": "reactions", "label": "Fluxes",
         "tsv": "$\tt0\tt1\nRXN-1\t9.0\t8.0\n"},
    ]
    mix = build_mixture(ds)
    assert "EG10001\t0.0000\t0.4016" in mix
    assert "RXN-1\t9.0\t8.0" in mix
    assert "a mixture" in mix
    assert build_mixture([]) == ""


# ---------------------------------------------------------------------------
# cell_overview_url + view
# ---------------------------------------------------------------------------

def test_cell_overview_url():
    assert cell_overview_url() == (
        "http://localhost:1555/overviewsWeb/celOv.shtml?orgid=ECOLI"
    )
    assert cell_overview_url("h", 9, "X") == (
        "http://h:9/overviewsWeb/celOv.shtml?orgid=X"
    )


def test_overview_view_html_has_launcher_and_downloads():
    ds = [{
        "entity_type": "genes", "label": "RNA → gene expression",
        "filename": "ptools_overview_genes.tsv",
        "content": to_omics_dataset(_TSV, "genes"),
    }]
    url = cell_overview_url("myhost", 1555)
    view = overview_view_html(ds, url, "myhost", 1555, mixture="x\n")
    assert url in view
    assert 'target="_blank"' in view
    assert 'download="ptools_overview_genes.tsv"' in view
    assert "data:text/tab-separated-values;base64," in view
    assert ">genes<" in view  # object type surfaced


def test_overview_view_html_reports_errors():
    view = overview_view_html([], cell_overview_url(), errors={"ptools_rxns": "boom"})
    assert "Skipped sources" in view and "ptools_rxns" in view


# ---------------------------------------------------------------------------
# Integration (requires a real sweep + paired sim_data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SWEEP, reason="reference sweep parquet or paired sim_data absent")
def test_ptools_overview_end_to_end():
    import duckdb
    from v2ecoli.library.sim_data import LoadSimData
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analyses.ptools_overview import PtoolsCellOverview

    files = _glob.glob(os.path.join(_REF_HISTORY, "**", "*.pq"), recursive=True)
    frm = "read_parquet([" + ",".join("'" + f + "'" for f in files) + "], hive_partitioning=true)"
    sql = (
        f"SELECT * FROM {frm} WHERE variant=0 AND lineage_seed=0"
        f" AND generation=0 AND agent_id='0' ORDER BY global_time"
    )
    sd = LoadSimData(sim_data_path=_PAIRED_SIMDATA).sim_data
    out = PtoolsCellOverview({}, core=allocate_core()).update(
        {"conn": duckdb.connect(), "history_sql": sql, "sim_data": sd,
         "variant_metadata": {"n_tp": 8}}
    )
    assert out["data"].get("tsv"), "combined omics file is empty"
    assert out["data"]["cell_overview_url"].endswith("orgid=ECOLI")
    # At least the genes dataset should come through.
    assert "genes" in out["data"]["entity_types"].values()
    view = out["view"]
    assert "celOv.shtml" in view and "download=" in view
