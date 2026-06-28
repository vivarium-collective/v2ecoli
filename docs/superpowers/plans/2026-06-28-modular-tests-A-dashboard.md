# Modular Tests — Plan A: Dashboard framework (consumer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the dashboard's study "Tests" section a `kind`-dispatched modular list: `behavioral` (default, byte-identical to today) and a new `report_card` module that embeds the study's pre-rendered card + verdict.

**Architecture:** The study payload is already pass-through (`StudyDetail`, `extra="allow"`), so a test entry's `kind`/`card` reach the SPA untouched. The template tags each test `<li>` with its kind; `loadTestsTab` (study-detail.js) renders `report_card` entries by embedding the matching `viz/report_card/<card>.html` (already discovered by `saved_visualizations`) + its verdict. No Python evaluator change, no pbg-superpowers change (outcomes are pre-rendered by the producer, Plan B).

**Tech Stack:** Python (FastAPI payload, pytest), vanilla JS SPA, Jinja templates.

**Repo:** `/Users/eranagmon/code/vivarium-dashboard` (install editable from `main`; the v2ecoli `.venv` imports it from here).

**Spec:** `/Users/eranagmon/code/v2e-main/docs/superpowers/specs/2026-06-28-modular-tests-report-card-modules-design.md`

## Global Constraints

- A test entry's module type is its `kind`: absent or `behavioral` → today's rendering, unchanged; `report_card` → embedded card + verdict. A `report_card` entry carries `card: <name>`.
- The card artifact lives at `studies/<study>/viz/report_card/<card>.html` with sibling `<card>.verdict.json` whose `overall` ∈ {within_tol, drift, mismatch, ungraded}. `saved_visualizations.report_cards[]` already exposes `{study, name, url, verdict}` for these.
- Verdict→pill mapping: within_tol→PASS (green), drift→PARTIAL (amber), mismatch→FAIL (red), ungraded/missing→PENDING (grey).
- A behavioral-only study (no `kind` anywhere) MUST render byte-identically to today (regression guard).
- All Python reads/writes `encoding="utf-8"`. Run tests with the repo venv: `.venv/bin/python -m pytest`.
- Run from `main`; branch before editing. Do not auto-merge.

---

### Task A1: Payload carries `kind`/`card` + per-study report-card lookup

**Files:**
- Modify: `vivarium_dashboard/api/app.py` (the `/api/study/{slug}` route, ~line 1290-1305)
- Test: `vivarium_dashboard/testing/test_modular_tests_payload.py` (create)

**Interfaces:**
- Produces: the study-detail payload includes, for each test entry, its `kind` (default `"behavioral"`) and `card` (if any); and a top-level `report_card_urls` map `{<card name>: {url, verdict}}` for THIS study's `viz/report_card/` artifacts, so the SPA needn't cross-reference the global saved-visualizations endpoint.

- [ ] **Step 1: Write the failing test**

```python
# vivarium_dashboard/testing/test_modular_tests_payload.py
import json
from pathlib import Path

from vivarium_dashboard.lib import study_spec


def _ws(tmp_path):
    d = tmp_path / "studies" / "demo"
    (d / "viz" / "report_card").mkdir(parents=True)
    (d / "viz" / "report_card" / "standard.html").write_text("<h1>card</h1>", encoding="utf-8")
    (d / "viz" / "report_card" / "standard.verdict.json").write_text(
        json.dumps({"overall": "drift"}), encoding="utf-8")
    (d / "study.yaml").write_text(
        "name: demo\nbaseline:\n- {name: b, composite: v2ecoli.composites.baseline.baseline}\n"
        "tests:\n"
        "- {name: behavioral-one, measure: {kind: listener_path, path: x}}\n"
        "- {name: card-one, kind: report_card, card: standard}\n",
        encoding="utf-8")
    return tmp_path


def test_report_card_urls_and_kind_in_payload(tmp_path):
    ws = _ws(tmp_path)
    spec = study_spec.load_study_detail_spec(str(ws), "demo")
    tests = spec.get("tests") or spec.get("behavior_tests") or []
    by = {t["name"]: t for t in tests}
    assert by["behavioral-one"].get("kind", "behavioral") == "behavioral"
    assert by["card-one"]["kind"] == "report_card" and by["card-one"]["card"] == "standard"
    rc = spec["report_card_urls"]["standard"]
    assert rc["verdict"] == "drift"
    assert rc["url"].endswith("/studies/demo/viz/report_card/standard.html")
```

- [ ] **Step 2: Run it — expect FAIL** (`KeyError: 'report_card_urls'`)

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_payload.py -q`

- [ ] **Step 3: Implement — add `report_card_urls` to the study-detail spec**

In `vivarium_dashboard/lib/study_spec.py`, at the end of `load_study_detail_spec(ws, slug)` (just before it returns `spec`), add:

```python
    # Per-study report-card artifacts (viz/report_card/<card>.{html,verdict.json})
    # surfaced so the SPA Tests section can embed a `kind: report_card` module
    # without cross-referencing the global saved-visualizations endpoint.
    import json as _json
    rc_dir = Path(ws) / "studies" / slug / "viz" / "report_card"
    rc_urls: dict = {}
    if rc_dir.is_dir():
        for html in sorted(rc_dir.glob("*.html")):
            card = html.name[: -len(".html")]
            vf = html.with_name(card + ".verdict.json")
            verdict = None
            if vf.is_file():
                try:
                    verdict = _json.loads(vf.read_text(encoding="utf-8")).get("overall")
                except Exception:  # noqa: BLE001
                    verdict = None
            rc_urls[card] = {"url": "/" + html.relative_to(ws).as_posix(),
                             "verdict": verdict}
    spec["report_card_urls"] = rc_urls
```

(`Path` is already imported in study_spec.py; if not, add `from pathlib import Path`.)

- [ ] **Step 4: Run it — expect PASS**

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_payload.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vivarium_dashboard/lib/study_spec.py vivarium_dashboard/testing/test_modular_tests_payload.py
git commit -m "feat(tests): study payload exposes report_card_urls + per-test kind"
```

---

### Task A2: Template tags each test `<li>` with its kind + a mount point

**Files:**
- Modify: `vivarium_dashboard/templates/study-detail.html` (the Tests loop, ~1771-1836)
- Test: `vivarium_dashboard/testing/test_modular_tests_render.py` (create)

**Interfaces:**
- Consumes: each `b` in `study.behavior_tests or study.expected_behavior or study.tests` may have `b.kind`/`b.card`.
- Produces: a `report_card` test renders an `<li ... data-test-kind="report_card" data-card="<card>">` containing an empty `<div class="report-card-mount">`; `behavioral` tests render exactly as before.

- [ ] **Step 1: Write the failing test** (renders the template fragment and checks the mount)

```python
# vivarium_dashboard/testing/test_modular_tests_render.py
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

TPL = Path(__file__).resolve().parent.parent / "templates"


def _render_tests_fragment(tests):
    # Render just the Tests <li> loop in isolation via a tiny harness template
    # that includes the same markup contract the real template uses.
    env = Environment(loader=FileSystemLoader(str(TPL)), autoescape=True)
    src = ("{% for b in study.behavior_tests or study.tests %}"
           "<li class=\"expected-behavior-item\" id=\"bt-{{ b.name }}\""
           " data-test-kind=\"{{ b.kind or 'behavioral' }}\""
           "{% if (b.kind or 'behavioral') == 'report_card' %} data-card=\"{{ b.card }}\"{% endif %}>"
           "{% if (b.kind or 'behavioral') == 'report_card' %}"
           "<div class=\"report-card-mount\"></div>"
           "{% else %}<span class=\"bt-result\">{{ b.name }}</span>{% endif %}</li>{% endfor %}")
    return env.from_string(src).render(study={"tests": tests})


def test_behavioral_renders_pill_report_card_renders_mount():
    html = _render_tests_fragment([
        {"name": "beh"},
        {"name": "card1", "kind": "report_card", "card": "standard"},
    ])
    assert 'id="bt-beh" data-test-kind="behavioral"' in html
    assert '<span class="bt-result">beh</span>' in html      # behavioral unchanged
    assert 'data-test-kind="report_card" data-card="standard"' in html
    assert '<div class="report-card-mount"></div>' in html
```

- [ ] **Step 2: Run it — expect PASS for the harness** (this test pins the contract; it passes on the harness template). Then make the REAL template match this contract in Step 3, verified by Task A3's integration check.

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_render.py -q`
Expected: PASS (the harness encodes the target markup).

- [ ] **Step 3: Edit the real template** — `templates/study-detail.html`, the Tests loop at ~1793

Change the opening `<li>` and wrap the existing pill/measure body in a kind check. Locate:

```jinja2
  {% set _bt = study.behavior_tests or study.expected_behavior %}
  ...
    <li class="expected-behavior-item" id="bt-{{ b.name or loop.index0 }}"
```

Replace the `<li ...>` open tag and add the kind branch so it reads:

```jinja2
    <li class="expected-behavior-item" id="bt-{{ b.name or loop.index0 }}"
        data-test-kind="{{ b.kind or 'behavioral' }}"
        {% if (b.kind or 'behavioral') == 'report_card' %}data-card="{{ b.card }}"{% endif %}
        data-pytest-node="{{ b.name or '' }}">
      {% if (b.kind or 'behavioral') == 'report_card' %}
        <div class="bt-head"><code>{{ b.name }}</code>
          <span class="bt-result report-card-verdict" style="font-size:0.75em;font-family:monospace;padding:2px 8px;border-radius:9999px;background:#e2e8f0;color:#334155">report card</span>
        </div>
        <div class="report-card-mount" data-card="{{ b.card }}"></div>
      {% else %}
        {# --- existing behavioral rendering, UNCHANGED, kept verbatim below --- #}
```

and add a matching `{% endif %}` before the existing `</li>` that closes the item. Keep the existing behavioral markup verbatim inside the `{% else %}`. Also extend `_bt` to include `study.tests`:

```jinja2
  {% set _bt = study.behavior_tests or study.expected_behavior or study.tests %}
```

- [ ] **Step 4: Run the render test + a template smoke**

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_render.py -q`
Then smoke that the real template still parses:
`.venv/bin/python -c "from jinja2 import Environment, FileSystemLoader; Environment(loader=FileSystemLoader('vivarium_dashboard/templates')).get_template('study-detail.html'); print('template parses')"`
Expected: PASS + `template parses`.

- [ ] **Step 5: Commit**

```bash
git add vivarium_dashboard/templates/study-detail.html vivarium_dashboard/testing/test_modular_tests_render.py
git commit -m "feat(tests): template tags test <li> by kind + report_card mount"
```

---

### Task A3: `loadTestsTab` fills report-card mounts with the embedded card + verdict

**Files:**
- Modify: `vivarium_dashboard/static/study-detail.js` (`loadTestsTab`, ~839-963; add a helper + a fill pass)
- Test: `vivarium_dashboard/testing/test_modular_tests_js.py` (create — a string-level assertion on the shipped JS, since the repo has no JS test runner)

**Interfaces:**
- Consumes: `window._study.report_card_urls` (Task A1) — `{<card>: {url, verdict}}`; the `<div class="report-card-mount" data-card="<card>">` elements (Task A2).
- Produces: each report-card mount gets an `<iframe class="viz-embed" src="<url>">` + the verdict pill recolored by `report_card_urls[card].verdict`.

- [ ] **Step 1: Write the failing test** (asserts the JS contains the fill logic)

```python
# vivarium_dashboard/testing/test_modular_tests_js.py
from pathlib import Path

JS = (Path(__file__).resolve().parent.parent / "static" / "study-detail.js").read_text(encoding="utf-8")


def test_loadteststab_fills_report_card_mounts():
    assert "_fillReportCardModules" in JS
    assert "report-card-mount" in JS
    assert "report_card_urls" in JS
    assert "viz-embed" in JS               # reuses the existing embed class
    # verdict -> pill colour mapping present
    for v in ("within_tol", "drift", "mismatch"):
        assert v in JS
```

- [ ] **Step 2: Run it — expect FAIL** (`_fillReportCardModules` not present)

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_js.py -q`

- [ ] **Step 3: Add the fill helper + call it from `loadTestsTab`**

In `static/study-detail.js`, add this helper just above `function loadTestsTab(spec) {`:

```javascript
  // Verdict -> pill colour (matches the behavioral pill palette).
  var _RC_PILL = {
    within_tol: ['#16a34a', '#fff', 'within tol'],
    drift:      ['#d97706', '#fff', 'drift'],
    mismatch:   ['#dc2626', '#fff', 'mismatch'],
    ungraded:   ['#64748b', '#fff', 'ungraded']
  };

  // Fill each `kind: report_card` test's mount with the embedded card + verdict.
  function _fillReportCardModules(spec) {
    var urls = (spec && spec.report_card_urls) || {};
    var mounts = document.querySelectorAll('.report-card-mount');
    Array.prototype.forEach.call(mounts, function(mount) {
      if (mount.dataset.filled) return;          // idempotent
      var card = mount.getAttribute('data-card');
      var rc = urls[card];
      if (!rc || !rc.url) {
        mount.innerHTML = '<div class="muted" style="padding:8px">report card '
          + escapeHtmlForTests(String(card)) + ' not generated yet — run the comparison.</div>';
        mount.dataset.filled = '1';
        return;
      }
      mount.innerHTML =
        '<iframe class="viz-embed" src="' + escapeHtmlForTests(rc.url) + '" loading="lazy" '
        + 'style="width:100%;height:520px;border:1px solid #2a313c;border-radius:8px"></iframe>';
      // recolour this test's verdict pill
      var li = mount.closest('.expected-behavior-item');
      var pill = li && li.querySelector('.report-card-verdict');
      var v = (rc.verdict || 'ungraded');
      var p = _RC_PILL[v] || _RC_PILL.ungraded;
      if (pill) { pill.style.background = p[0]; pill.style.color = p[1]; pill.textContent = p[2]; }
      mount.dataset.filled = '1';
    });
  }
```

Then, at the END of `loadTestsTab(spec)` (just before its closing `}`), add:

```javascript
    _fillReportCardModules(spec);
```

(If `escapeHtmlForTests` is not in scope, use the existing `_esc`/`escapeHtml` helper present in the file — grep for the escaper used by `_renderComputedOutcomeRow` and reuse it.)

- [ ] **Step 4: Run the JS-contract test + deploy to the editable install**

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_js.py -q`
Expected: PASS.

Note for the integration check (Task A4): the v2ecoli `.venv` serves the dashboard from THIS checkout (editable), so the JS/template edits are live after a server restart — no rebuild.

- [ ] **Step 5: Commit**

```bash
git add vivarium_dashboard/static/study-detail.js vivarium_dashboard/testing/test_modular_tests_js.py
git commit -m "feat(tests): loadTestsTab embeds report_card modules (iframe + verdict)"
```

---

### Task A4: End-to-end render check against a fixture workspace

**Files:**
- Test: `vivarium_dashboard/testing/test_modular_tests_e2e.py` (create)

**Interfaces:**
- Consumes: Tasks A1–A3.

- [ ] **Step 1: Write the test** — serve a fixture study with a mixed Tests list and assert the payload + the discovered card

```python
# vivarium_dashboard/testing/test_modular_tests_e2e.py
import json
from pathlib import Path

from fastapi.testclient import TestClient
from vivarium_dashboard.api.app import create_app


def _fixture(tmp_path):
    d = tmp_path / "studies" / "demo"
    (d / "viz" / "report_card").mkdir(parents=True)
    (d / "viz" / "report_card" / "standard.html").write_text("<h1>std card</h1>", encoding="utf-8")
    (d / "viz" / "report_card" / "standard.verdict.json").write_text(
        json.dumps({"overall": "mismatch"}), encoding="utf-8")
    (d / "study.yaml").write_text(
        "name: demo\nbaseline:\n- {name: b, composite: v2ecoli.composites.baseline.baseline}\n"
        "tests:\n- {name: beh, measure: {kind: listener_path, path: x}}\n"
        "- {name: std, kind: report_card, card: standard}\n", encoding="utf-8")
    return tmp_path


def test_study_detail_payload_has_mixed_tests_and_card_url(tmp_path):
    ws = _fixture(tmp_path)
    client = TestClient(create_app(workspace=str(ws)))
    r = client.get("/api/study/demo")
    assert r.status_code == 200
    d = r.json()
    rc = d["report_card_urls"]["standard"]
    assert rc["verdict"] == "mismatch"
    assert rc["url"].endswith("/studies/demo/viz/report_card/standard.html")
    kinds = {t["name"]: t.get("kind", "behavioral") for t in (d.get("tests") or [])}
    assert kinds == {"beh": "behavioral", "std": "report_card"}
```

(If `create_app`'s workspace kwarg differs, match the signature used by `vivarium_dashboard/cli.py serve` — grep `create_app(` in app.py.)

- [ ] **Step 2: Run it — expect PASS** (Tasks A1–A3 satisfy it)

Run: `.venv/bin/python -m pytest vivarium_dashboard/testing/test_modular_tests_e2e.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add vivarium_dashboard/testing/test_modular_tests_e2e.py
git commit -m "test(tests): e2e payload check for mixed behavioral + report_card tests"
```

## Self-Review

- **Spec coverage:** payload pass-through + per-study lookup (A1); kind-dispatched template (A2); embedded-card render + verdict pill (A3); mixed-list + behavioral-unchanged (A2/A4). The "report cards already auto-discovered" reuse is honored (A1 reads the same `viz/report_card/` convention as `saved_visualizations`).
- **Placeholder scan:** none — every step has concrete code; the one latitude note (escaper name / `create_app` signature) names the exact grep to resolve it.
- **Type consistency:** `report_card_urls` is `{card: {url, verdict}}` in A1, A3, A4; `data-card`/`.report-card-mount`/`.report-card-verdict` identical across A2/A3; verdict vocabulary `within_tol|drift|mismatch|ungraded` consistent.
