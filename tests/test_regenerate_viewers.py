import json
from pathlib import Path
import importlib.util

import pytest

# regenerate_viewers.py resolves composites via vivarium_workbench, which the
# fast-tests CI job omits (--no-install-package vivarium-workbench). Skip the
# whole module when it is absent instead of failing on ModuleNotFoundError.
pytest.importorskip("vivarium_workbench")

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

    rows = mod.build_rows(entries, viewers_dir=tmp_path,
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
    (src / "assets" / "app.js.map").write_text("{}")  # source map — must be stripped
    dest = tmp_path / "viewers" / "loom"
    mod.copy_loom_bundle(dest, src=src)
    assert (dest / "index.html").read_text() == "<html>loom</html>"
    assert (dest / "assets" / "app.js").is_file()
    assert not (dest / "assets" / "app.js.map").exists()  # a read-only viewer drops maps


def test_write_state_serializes_numpy_arrays(tmp_path):
    """Composite states are full of numpy bulk-count arrays; write_state must
    serialize them via the dashboard's _json_body (ndarray -> list) rather than
    choking on a plain json.dumps."""
    import numpy as np
    mod = _load_mod()
    out = mod.write_state({"counts": np.array([1, 2, 3])}, "x", tmp_path / "data")
    written = json.loads(out.read_text())
    assert written["state"]["counts"] == [1, 2, 3]


def test_build_rows_isolates_a_failing_composite(tmp_path):
    """One composite raising mid-render must not abort the whole run."""
    mod = _load_mod()
    entries = [mod.Entry("a", "id.a", "A", ""), mod.Entry("b", "id.b", "B", "")]
    def res(spec_id): return {"x": 1}
    def boom_svg(state, slug, out):
        if slug == "a":
            raise RuntimeError("kaboom")
        return out / f"{slug}.svg"
    def ok_viz2(state, slug, out): return out / f"{slug}.html"
    rows = mod.build_rows(entries, viewers_dir=tmp_path,
                          resolve=res, render_svg=boom_svg, render_viz2=ok_viz2)
    # "a" dropped (its render raised); the run continued and produced "b".
    assert [r["entry"].slug for r in rows] == ["b"]


def test_trim_state_for_view_caps_long_arrays_keeps_structure():
    import numpy as np
    mod = _load_mod()
    state = {
        "agents": {"0": {
            "bulk": np.arange(16321),                       # huge numpy array
            "unique": {"ribosome": [{"i": n} for n in range(12507)]},  # huge list of dicts
            "process": {"_type": "process", "address": "local:Foo",
                        "inputs": {"x": ["agents", "0", "bulk"]}},  # short path list kept
        }},
    }
    trimmed = mod.trim_state_for_view(state, max_list=8)
    a0 = trimmed["agents"]["0"]
    assert len(a0["bulk"]) == 8                       # numpy array capped
    assert len(a0["unique"]["ribosome"]) == 8         # list of dicts capped
    assert a0["unique"]["ribosome"][0] == {"i": 0}    # dict structure preserved
    assert a0["process"]["_type"] == "process"        # process node untouched
    assert a0["process"]["inputs"]["x"] == ["agents", "0", "bulk"]  # short wiring path intact


def test_baseline_viewer_redirects_into_hub():
    """The legacy /baseline-viewer/ URL (and its QR) must redirect into the hub's
    baseline loom view. stateUrl is relative to the loom page (viewers/loom/),
    matching how the hub itself links baseline."""
    p = Path(__file__).parent.parent / "docs" / "baseline-viewer" / "index.html"
    html = p.read_text(encoding="utf-8")
    assert "location.replace" in html
    assert "../viewers/loom/index.html?static=1" in html
    assert "id=v2ecoli.composites.baseline.baseline" in html
    assert "stateUrl=../data/baseline.state.json" in html
    assert "viewUrl=../data/baseline.view.json" in html
