import subprocess
from pathlib import Path
from scripts._compare.study_spec import specs_from_configs, REPO
from scripts._compare.reference import ReferenceEngine

TOP = Path(__file__).resolve().parents[2]


def test_no_stale_investigation_id():
    hits = subprocess.run(
        ["grep", "-rl", "v2ecoli-vecoli-comparison",
         str(TOP / "workspace"), str(TOP / "docs" / "report_cards")],
        capture_output=True, text=True).stdout.strip()
    assert hits == "", f"stale id remains in:\n{hits}"


def test_new_investigation_loads():
    from scripts._compare.study_spec import _context, _invest_dir
    ctx = _context(_invest_dir("whole-cell-model-comparison"))
    assert ctx["invest_name"] == "whole-cell-model-comparison"
    assert ctx["reference"].kind == "vecoli"


def test_specs_use_top_level_registry_path():
    ctx = {"invest_name": "whole-cell-model-comparison",
           "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
           "configs": [{"name": "basal", "config": "basal"}],
           "v2_cache": "vc", "ve_cache": "vec",
           "defaults": {"seeds": 4, "gens": 1, "cards": ["parca"]}, "inv_dir": None}
    sp = specs_from_configs(ctx)[0].study_path
    assert sp == str(REPO / "workspace" / "studies" / "basal" / "study.yaml")
