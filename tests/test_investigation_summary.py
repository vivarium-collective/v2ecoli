from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "workspace"
SLUG = "v2ecoli-vecoli-comparison"


def test_aggregate_discovers_studies_in_dag_order():
    from reports._summary.aggregate import aggregate

    summ = aggregate(SLUG, WS)
    assert summ["slug"] == SLUG
    assert summ["question"].startswith("Does v2ecoli reproduce vEcoli")
    slugs = [s["slug"] for s in summ["studies"]]
    # Same study set, asserted as a VALID DAG order (parca root first; every
    # study after its in-set prerequisites) rather than a brittle exact authored
    # sequence — the registry migrator sorts members alphabetically, so
    # aggregate() topologically re-orders and the exact order isn't fixed.
    # The single `metabolism_redux` study was superseded by 5 per-condition
    # studies (Task A2): metabolism_redux_{basal,with_aa,succinate,no_oxygen,
    # acetate}, each still gated on `parca`.
    assert set(slugs) == {
        "parca", "basal", "with_aa", "succinate", "no_oxygen", "acetate",
        "statistical",
        "metabolism_redux_basal", "metabolism_redux_with_aa",
        "metabolism_redux_succinate", "metabolism_redux_no_oxygen",
        "metabolism_redux_acetate",
    }
    assert slugs[0] == "parca"
    pos = {s: i for i, s in enumerate(slugs)}
    for s in summ["studies"]:
        for p in s["prerequisites"]:
            if p in pos:
                assert pos[p] < pos[s["slug"]], f"{p} must precede {s['slug']}"


def test_aggregate_per_study_metadata_and_rollup():
    from reports._summary.aggregate import aggregate

    summ = aggregate(SLUG, WS)
    by = {s["slug"]: s for s in summ["studies"]}
    # after the rpoBC fix, acetate reproduces vEcoli natively (was FAIL pre-fix)
    assert by["acetate"]["result"] == "PASS"
    assert by["parca"]["result"] == "PASS"
    assert by["parca"]["prerequisites"] == []
    assert by["acetate"]["prerequisites"] == ["parca"]
    assert by["statistical"]["prerequisites"] == ["basal"]
    # post-fix finding states acetate reproduces vEcoli natively
    assert "reproduces vEcoli" in (by["acetate"]["finding"] or "")
    # config + standard cards discovered for acetate; parca card for parca
    assert {c["name"] for c in by["acetate"]["cards"]} == {"config", "standard", "statistical"}
    assert {c["name"] for c in by["parca"]["cards"]} == {"parca"}
    # config card is ungraded, standard is graded
    acards = {c["name"]: c for c in by["acetate"]["cards"]}
    assert acards["config"]["graded"] is False
    assert acards["standard"]["graded"] is True
    # standard (single-seed, illustrative) still carries its pre-fix verdict;
    # the gold-standard statistical card + conclusions reflect the fix.
    assert acards["standard"]["overall"] == "mismatch"
    # post-fix: every condition study reproduces vEcoli (metabolism_redux ungraded)
    assert summ["rollup"] == {"PASS": 7, "PARTIAL": 0, "FAIL": 0}


def test_matrix_columns_lead_with_standard_observables():
    from reports._summary.aggregate import aggregate

    m = aggregate(SLUG, WS)["matrix"]
    # standard-card observables appear first, in card order
    assert m["columns"][:5] == [
        "cell mass (fg)", "dry mass (fg)", "protein mass (fg)",
        "RNA mass (fg)", "growth rate (1/s)",
    ]


def test_matrix_cell_verdicts_match_source_json():
    import json
    from reports._summary.aggregate import aggregate

    m = aggregate(SLUG, WS)["matrix"]
    rows = {r["study"]: r["cells"] for r in m["rows"]}
    # acetate growth rate is a mismatch in the source verdict.json
    src = json.loads(
        (WS / "studies" / "acetate"  # registry: studies live top-level (Spec 1)
         / "viz" / "report_card" / "standard.verdict.json").read_text(encoding='utf-8')
    )
    axis = {a["label"]: a["verdict"] for a in src["groups"]["standard"]["axes"]}
    assert rows["acetate"]["growth rate (1/s)"] == axis["growth rate (1/s)"] == "mismatch"
    assert rows["acetate"]["RNA mass (fg)"] == "drift"
    # parca has cell mass within tolerance, no growth-rate column value
    assert rows["parca"]["cell mass (fg)"] == "within_tol"
    assert rows["parca"].get("growth rate (1/s)") is None


def test_card_html_inlined_and_full_doc_flagged():
    from reports._summary.aggregate import aggregate

    by = {s["slug"]: s for s in aggregate(SLUG, WS)["studies"]}
    scards = {c["name"]: c for c in by["statistical"]["cards"]}
    # statistical.html is a full <html> document -> flagged for iframe embedding
    assert scards["statistical"]["is_full_doc"] is True
    assert "<html" in scards["statistical"]["html"].lower()
    # standard.html is a fragment (no <html>) -> inlined directly
    acards = {c["name"]: c for c in by["acetate"]["cards"]}
    assert acards["standard"]["is_full_doc"] is False
    assert acards["standard"]["html"].strip() != ""
    assert "<html" not in acards["standard"]["html"].lower()


def test_missing_card_marked_not_crashing(tmp_path):
    from reports._summary.aggregate import _card_stub

    # a study dir with a declared card whose files don't exist
    stub = _card_stub(tmp_path, "viz/report_card/ghost.html")
    assert stub["missing"] is True
    assert stub["html"] == ""
    assert stub["graded"] is False


def test_render_overview_and_matrix():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS), style_css=":root{--x:1}")
    assert "<!doctype html>" in html.lower()
    assert "Does v2ecoli reproduce vEcoli" in html
    # rollup counts present (post-fix: all condition studies PASS)
    assert "7 PASS" in html
    # matrix header + a verdict-colored cell class
    assert "growth rate (1/s)" in html
    assert "verdict-mismatch" in html
    # self-contained: no external stylesheet or script src
    assert "<link " not in html.lower()
    assert "script src" not in html.lower()


def test_render_per_study_sections_and_embedding():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS))
    # every study appears as a details section
    for slug in ("parca", "acetate", "statistical"):
        assert f'id="study-{slug}"' in html
    # fragment card (acetate standard) inlined: its <h3 ... simulation runs heading present
    assert "simulation runs" in html
    # full-doc card (statistical) embedded via iframe srcdoc
    assert "iframe" in html and "srcdoc=" in html
    # auto-height shim present exactly once
    assert html.count("scrollHeight") >= 1


def test_cli_writes_selfcontained_file(tmp_path):
    from reports.investigation_summary import main

    out = tmp_path / "summary.html"
    rc = main(["--investigation", SLUG, "--out", str(out), "--no-open"])
    assert rc == 0
    text = out.read_text(encoding='utf-8')
    assert out.stat().st_size > 5000
    # all 7 studies present, self-contained
    for slug in ("parca", "basal", "with_aa", "succinate", "no_oxygen", "acetate", "statistical"):
        assert f'id="study-{slug}"' in text
    assert "<link " not in text.lower()
    assert "script src" not in text.lower()


def test_badge_class_whitelisted():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    # real graded study produces a legitimate badge class
    html = render(aggregate(SLUG, WS))
    assert "badge mismatch" in html

    # synthetic: an attacker-controlled overall value must not land in the
    # class attribute unescaped, even though it's still shown (escaped) as text
    summary = {
        "slug": "x", "title": "x", "question": "",
        "rollup": {"PASS": 0, "PARTIAL": 0, "FAIL": 0},
        "matrix": {"columns": [], "rows": []},
        "studies": [{
            "slug": "s1", "title": "S1", "finding": None,
            "prerequisites": [],
            "cards": [{
                "name": "c1", "graded": True, "overall": '"><x',
                "html": "frag", "is_full_doc": False, "missing": False, "axes": [],
            }],
        }],
    }
    bad_html = render(summary)
    assert 'class="badge "><x"' not in bad_html
    # the badge span's own class attribute must be a safe whitelisted value,
    # not the attacker string (anchor on the badge span, not the page shell)
    import re
    badge_cls = re.search(r'<span class="badge ([^"]*)">', bad_html).group(1)
    assert badge_cls in ("", "within_tol", "drift", "mismatch")
    # ...but the value is still shown as text, HTML-escaped
    assert "&gt;&lt;x" in bad_html


def test_fragment_with_style_block_is_isolated(tmp_path):
    from reports._summary.aggregate import _card_stub

    card_path = tmp_path / "viz" / "report_card" / "frag.html"
    card_path.parent.mkdir(parents=True)
    card_path.write_text("<style>.x{color:red}</style><div>hi</div>")
    verdict_path = tmp_path / "viz" / "report_card" / "frag.verdict.json"
    verdict_path.write_text('{"overall": "within_tol"}')

    stub = _card_stub(tmp_path, "viz/report_card/frag.html")
    assert stub["is_full_doc"] is True
    assert "<html" not in stub["html"].lower()


def test_cli_unknown_investigation_returns_error():
    from reports.investigation_summary import main

    rc = main(["--investigation", "no-such-inv", "--no-open"])
    assert rc == 2


def test_aggregate_config_json_is_real_baseline_config():
    from reports._summary.aggregate import aggregate

    by = {s["slug"]: s for s in aggregate(SLUG, WS)["studies"]}
    cfg = by["acetate"]["config_json"]
    assert cfg["composite"] == "v2ecoli.composites.ecoli_baseline.ecoli_baseline"
    assert cfg["params"] == {"condition": "acetate"}
    # comparison run settings folded in
    assert cfg["seeds"] == 4  # gold standard: condition studies run 4 seeds
    assert cfg["generations"] == 4


def test_topnav_links_every_study():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS))
    assert '<nav class="topnav">' in html
    assert 'href="#top"' in html  # overview / home link
    for slug in ("parca", "basal", "with_aa", "succinate",
                 "no_oxygen", "acetate", "statistical"):
        assert f'href="#study-{slug}"' in html


def test_config_json_replaces_config_card():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS))
    # the actual baseline config JSON is shown (HTML-escaped inside <pre>)...
    assert "config (JSON)" in html
    assert "v2ecoli.composites.ecoli_baseline.ecoli_baseline" in html
    assert "&quot;composite&quot;" in html  # JSON keys escaped, not raw-injected
    # ...and the config *report card* is no longer embedded as a card block
    assert "config card" not in html
    # graded cards still embedded
    assert "standard card" in html
