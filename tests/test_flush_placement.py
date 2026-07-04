# tests/test_flush_placement.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, place_output


def _extract(tmp_path, slug="demo", with_study=True, out=None):
    if with_study:
        sd = tmp_path / "workspace" / "studies" / slug
        sd.mkdir(parents=True)
        (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
        return RunExtract(out or "out/x", {"study": slug}, tmp_path)
    return RunExtract(out or str(tmp_path / "out"), {}, tmp_path)


def test_report_card_placed_into_study_viz(tmp_path):
    ex = _extract(tmp_path, "demo")
    p = place_output("report_card", "tests", "<i>card</i>",
                     {"overall": "drift"}, ex)
    base = tmp_path / "workspace" / "studies" / "demo" / "viz" / "report_card"
    assert Path(p) == base / "tests.html"
    assert (base / "tests.html").read_text() == "<i>card</i>"
    assert json.loads((base / "tests.verdict.json").read_text())["overall"] == "drift"


def test_visualization_placed_into_study_viz(tmp_path):
    ex = _extract(tmp_path, "demo")
    p = place_output("visualization", "massfrac", "<div>v</div>", {}, ex)
    assert Path(p) == tmp_path / "workspace" / "studies" / "demo" / "viz" / "massfrac.html"
    assert Path(p).read_text() == "<div>v</div>"


def test_no_study_falls_back_to_out_dir(tmp_path):
    out = tmp_path / "out"
    ex = _extract(tmp_path, with_study=False, out=str(out))
    p = place_output("visualization", "massfrac", "<div>v</div>", {}, ex)
    assert Path(p) == out / "viz" / "massfrac.html"


def test_empty_view_writes_nothing(tmp_path):
    ex = _extract(tmp_path, "demo")
    assert place_output("visualization", "x", "", {}, ex) is None
