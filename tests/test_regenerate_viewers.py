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


def test_build_rows_skips_unresolvable_and_records_artifacts(tmp_path):
    mod = _load_mod()
    entries = [
        mod.Entry("baseline", "v2ecoli.composites.baseline.baseline", "Baseline", ""),
        mod.Entry("broken", "v2ecoli.composites.broken.broken", "Broken", ""),
    ]
    def fake_resolve(spec_id):
        return {"x": 1} if "baseline" in spec_id else None
    def fake_svg(state, slug, out): return (out / f"{slug}.svg") if slug == "baseline" else None
    def fake_viz2(state, slug, out): return (out / f"{slug}.html") if slug == "baseline" else None

    rows = mod.build_rows(entries, VIEWERS_DIR=tmp_path,
                          resolve=fake_resolve, render_svg=fake_svg, render_viz2=fake_viz2)
    slugs = [r["entry"].slug for r in rows]
    assert slugs == ["baseline"]
    assert (tmp_path / "data" / "baseline.state.json").is_file()
    assert rows[0]["has_svg"] and rows[0]["has_viz2"]
    assert rows[0]["has_view"] is False


def test_copy_loom_bundle_copies_index_and_assets(tmp_path, monkeypatch):
    mod = _load_mod()
    src = tmp_path / "src_dist"; (src / "assets").mkdir(parents=True)
    (src / "index.html").write_text("<html>loom</html>")
    (src / "assets" / "app.js").write_text("//js")
    dest = tmp_path / "viewers" / "loom"
    mod.copy_loom_bundle(dest, src=src)
    assert (dest / "index.html").read_text() == "<html>loom</html>"
    assert (dest / "assets" / "app.js").is_file()
