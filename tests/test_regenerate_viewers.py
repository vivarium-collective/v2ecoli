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


def test_loom_url_includes_id_and_relative_state(tmp_path):
    mod = _load_mod()
    e = mod.Entry(slug="baseline", id="v2ecoli.composites.baseline.baseline",
                  title="Baseline", blurb="x")
    assert mod.loom_url(e, has_view=True) == (
        "loom/index.html?static=1&id=v2ecoli.composites.baseline.baseline"
        "&stateUrl=../data/baseline.state.json"
        "&viewUrl=../data/baseline.view.json")
    assert "viewUrl" not in mod.loom_url(e, has_view=False)

def test_hub_html_lists_only_resolved_and_has_three_viewer_links():
    mod = _load_mod()
    e = mod.Entry(slug="baseline", id="v2ecoli.composites.baseline.baseline",
                  title="Baseline whole-cell", blurb="Full WCM.")
    html = mod.hub_html([
        {"entry": e, "has_view": True, "has_viz2": True, "has_svg": True},
    ])
    assert "Baseline whole-cell" in html
    assert "loom/index.html?static=1&id=v2ecoli.composites.baseline.baseline" in html
    assert 'href="viz2/baseline.html"' in html
    assert 'href="img/baseline.svg"' in html

def test_hub_html_omits_links_for_missing_artifacts():
    mod = _load_mod()
    e = mod.Entry(slug="colony", id="v2ecoli.composites.colony.colony",
                  title="Colony", blurb="")
    html = mod.hub_html([
        {"entry": e, "has_view": False, "has_viz2": False, "has_svg": False},
    ])
    assert "Colony" in html
    assert 'href="viz2/colony.html"' not in html
    assert 'href="img/colony.svg"' not in html
    assert "loom/index.html?static=1&id=v2ecoli.composites.colony.colony" in html
