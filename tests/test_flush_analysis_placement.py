import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, place_analysis_outputs


def _extract_with_study(tmp_path, slug="demo"):
    # Declare the nested studies layout the shared viva_workspace resolver reads
    # (matches v2ecoli's real workspace.yaml).
    (tmp_path / "workspace.yaml").write_text(
        yaml.safe_dump({"name": "test-ws", "layout": {"studies": "workspace/studies"}})
    )
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    out = tmp_path / "out" / "run1"
    (out / "viz").mkdir(parents=True)
    (out / "viz" / "mass_fraction__seed_0.html").write_text("<div>mf</div>")
    (out / "ptools").mkdir(parents=True)
    (out / "ptools" / "ptools_rna__seed_0.tsv").write_text("a\tb\n")
    return RunExtract(str(out), {"study": slug}, tmp_path), sd, out


def test_copies_viz_and_ptools_into_study_dir(tmp_path):
    ex, sd, out = _extract_with_study(tmp_path)
    placed = place_analysis_outputs(ex)
    # html copied into study viz
    assert (sd / "viz" / "mass_fraction__seed_0.html").read_text() == "<div>mf</div>"
    # tsv copied into study ptools
    assert (sd / "ptools" / "ptools_rna__seed_0.tsv").read_text() == "a\tb\n"
    # raw run artifacts left in place (copy, not move)
    assert (out / "viz" / "mass_fraction__seed_0.html").is_file()
    # placed reports the html
    assert {p["name"] for p in placed} == {"mass_fraction__seed_0"}
    assert placed[0]["kind"] == "analysis"


def test_no_study_copies_nothing(tmp_path):
    out = tmp_path / "out" / "run1"
    (out / "viz").mkdir(parents=True)
    (out / "viz" / "x.html").write_text("<i></i>")
    ex = RunExtract(str(out), {}, tmp_path)   # no owning study
    assert place_analysis_outputs(ex) == []
