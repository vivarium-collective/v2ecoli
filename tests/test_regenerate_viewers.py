import json
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).parent.parent / "scripts" / "regenerate_viewers.py"

def _load_mod():
    spec = importlib.util.spec_from_file_location("regenerate_viewers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_load_manifest_returns_curated_entries(tmp_path):
    mod = _load_mod()
    mf = tmp_path / "viewers.json"
    mf.write_text(json.dumps([
        {"slug": "baseline", "id": "v2ecoli.composites.baseline.baseline",
         "title": "Baseline whole-cell", "blurb": "Full single-cell WCM."},
    ]))
    entries = mod.load_manifest(mf)
    assert entries[0].slug == "baseline"
    assert entries[0].id == "v2ecoli.composites.baseline.baseline"
    assert entries[0].title == "Baseline whole-cell"


def test_write_state_writes_slug_json(tmp_path):
    mod = _load_mod()
    data_dir = tmp_path / "data"
    state = {"membrane": {}, "metabolism": {}}
    out = mod.write_state(state, "baseline", data_dir)
    assert out == data_dir / "baseline.state.json"
    written = json.loads(out.read_text())
    assert written["state"] == state
