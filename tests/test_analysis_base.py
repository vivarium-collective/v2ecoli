from bigraph_schema import allocate_core
from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY


class _Demo(Analysis):
    name = "demo_analysis"
    scale = "single"

    def analyze(self, *, conn, history_sql, sim_data, **ctx):
        return {"view": "<p>ok</p>", "data": {"n": 1}}


def test_analysis_registers_and_dispatches():
    assert ANALYSIS_REGISTRY["demo_analysis"] is _Demo
    step = _Demo({}, core=allocate_core())
    out = step.update({"conn": None, "history_sql": "SELECT 1", "sim_data": None})
    assert out == {"view": "<p>ok</p>", "data": {"n": 1}}


def test_analysis_defaults_missing_keys():
    class _Bare(Analysis):
        name = "bare_analysis"
        scale = "single"

        def analyze(self, **ctx):
            return {"data": {"x": 2}}  # no "view"

    step = _Bare({}, core=allocate_core())
    out = step.update({})
    assert out == {"view": "", "data": {"x": 2}}


def test_analysis_inputs_declare_duckdb_ports():
    assert set(_Demo({}, core=allocate_core()).inputs()) >= {
        "conn", "history_sql", "sim_data"}
