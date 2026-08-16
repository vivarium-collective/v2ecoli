import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, resolve_owning_study


def _ws_with_study(tmp_path, slug="demo"):
    (tmp_path / "workspace.yaml").write_text(
        yaml.safe_dump({"name": "test-ws", "layout": {"studies": "workspace/studies"}}))
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug, "tests": []}))
    return sd


def test_resolve_owning_study_from_config(tmp_path):
    _ws_with_study(tmp_path, "demo")
    assert resolve_owning_study("out/workflow", {"study": "demo"}, tmp_path) == "demo"


def test_resolve_owning_study_from_out_dir_path(tmp_path):
    sd = _ws_with_study(tmp_path, "demo")
    out = sd / "runs" / "r1"
    out.mkdir(parents=True)
    assert resolve_owning_study(str(out), {}, tmp_path) == "demo"


def test_resolve_owning_study_none(tmp_path):
    assert resolve_owning_study("out/workflow", {}, tmp_path) is None


def test_study_viz_dir_and_ctx(tmp_path):
    _ws_with_study(tmp_path, "demo")
    ex = RunExtract("out/workflow", {"study": "demo"}, tmp_path)
    assert ex.study_slug == "demo"
    assert ex.study_viz_dir() == tmp_path / "workspace" / "studies" / "demo" / "viz"
    assert ex.study_ctx().study_name == "demo"


def test_extraction_is_lazy(tmp_path):
    # No run data on disk; constructing RunExtract + reading study info must NOT
    # touch the (absent) parquet/sim_data. Only conn_ctx()/records() would.
    _ws_with_study(tmp_path, "demo")
    ex = RunExtract("out/workflow", {"study": "demo"}, tmp_path)
    bag = ex.context_bag()              # must not raise though no parquet exists
    assert bag["study"].study_name == "demo"
    assert "conn" in bag                 # present as a lazy handle/None, not built


def test_close_clears_ctx(tmp_path):
    ex = RunExtract("out/x", {}, tmp_path)
    class _FakeConn:
        closed = False
        def close(self):
            self.closed = True
    fake = _FakeConn()
    ex._ctx = {"conn": fake, "from_clause": "x", "sim_data": None, "validation_data": None}
    ex.close()
    assert fake.closed is True
    assert ex._ctx == {}   # cleared, so a later conn_ctx() would re-provision
