import yaml
from pathlib import Path
from scripts._compare.study_spec import load_investigation
from scripts._compare.materialize import materialize_study


def test_studies_declare_modules_matching_their_cards():
    _ctx, specs = load_investigation("v2ecoli-vecoli-comparison")
    for s in specs:
        materialize_study(s)
        data = yaml.safe_load(Path(s.study_path).read_text(encoding="utf-8"))
        test_cards = sorted(t["card"] for t in data["tests"] if t.get("kind") == "report_card")
        assert test_cards == sorted(s.cards)          # one module per card
        assert all(rc.startswith("viz/report_card/") for rc in data["report_cards"])
