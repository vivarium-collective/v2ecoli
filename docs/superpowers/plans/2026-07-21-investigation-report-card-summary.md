# Investigation Report-Card Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone generator that scans an investigation's studies, aggregates their report-card verdicts + rendered cards, and emits one self-contained HTML summary page.

**Architecture:** Pure aggregation (`reports/_summary/aggregate.py`: filesystem → a plain dict) is separated from rendering (`reports/_summary/render.py`: dict → HTML string). A thin CLI (`reports/investigation_summary.py`) wires them and writes/opens the file. Reworks the removed `reports/compare_report.py` scaffold for a proper investigation structure.

**Tech Stack:** Python 3.12, stdlib (`argparse`, `json`, `html`, `pathlib`, `webbrowser`), `pyyaml`, `pytest`.

## Global Constraints

- Output is **fully self-contained**: no external `<link href>` or `<script src>` to repo paths — all card HTML + CSS inlined.
- **Read-only**: never run a simulation, never write into `workspace/`.
- Default output path: `reports/summaries/<slug>_summary.html`.
- Verdict vocabulary (verbatim): `within_tol`, `drift`, `mismatch`, `ungraded`.
- Study order comes from `investigation.yaml`'s `studies:` list (already DAG-ordered); do not re-sort.
- Canonical run result = the `runs[]` entry with `canonical: true` → its `result` (`PASS`/`PARTIAL`/`FAIL`); `None` if no canonical run.
- All work on branch `feat/investigation-report-card-summary`.

---

## File Structure

- **Create** `reports/_summary/__init__.py` — package marker (empty).
- **Create** `reports/_summary/aggregate.py` — `aggregate(slug, workspace_root) -> dict`. Reads yaml + verdict.json + card html; returns the `InvestigationSummary` dict. No HTML.
- **Create** `reports/_summary/render.py` — `render(summary, style_css) -> str`. Pure dict → HTML string. No filesystem reads.
- **Create** `reports/investigation_summary.py` — CLI `main(argv=None)`.
- **Create** `tests/test_investigation_summary.py` — tests for aggregate, matrix, render, CLI.

### `InvestigationSummary` dict shape (produced by `aggregate`, consumed by `render`)

```python
{
  "slug": str,
  "title": str,
  "question": str,
  "studies": [
    {
      "slug": str,
      "title": str,
      "status": str | None,
      "result": str | None,            # "PASS" | "PARTIAL" | "FAIL" | None
      "prerequisites": list[str],
      "finding": str | None,           # first findings[].statement
      "cards": [
        {
          "name": str,                 # e.g. "standard", "config", "parca"
          "overall": str | None,       # within_tol/drift/mismatch/ungraded/None
          "graded": bool,              # overall not in (None, "ungraded")
          "html": str,                 # inlined card markup ("" if missing)
          "is_full_doc": bool,         # True if markup contains "<html"
          "axes": list[dict],          # [{"label","verdict","value","meter"}], graded groups only
          "missing": bool,             # True if html or verdict file absent
        }, ...
      ],
    }, ...
  ],
  "rollup": {"PASS": int, "PARTIAL": int, "FAIL": int},
  "matrix": {
    "columns": list[str],              # union of graded axis labels, first-appearance order
    "rows": [ {"study": str, "cells": {label: verdict_or_None}}, ... ],
  },
}
```

---

### Task 1: Aggregate — study discovery + per-study metadata

**Files:**
- Create: `reports/_summary/__init__.py`
- Create: `reports/_summary/aggregate.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Produces: `aggregate(slug: str, workspace_root: str | Path) -> dict` returning the `InvestigationSummary` dict above. This task fills `slug`, `title`, `question`, `studies[]` (each with `slug`, `title`, `status`, `result`, `prerequisites`, `finding`, and a `cards[]` list where each card has `name`, `overall`, `graded`, `missing` populated — `html`/`is_full_doc`/`axes` are added in Task 3, default to `""`/`False`/`[]` here), and `rollup`. `matrix` is added in Task 2 (default `{"columns": [], "rows": []}` here).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_investigation_summary.py
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
    assert slugs == [
        "parca", "basal", "with_aa", "succinate",
        "no_oxygen", "acetate", "statistical",
    ]


def test_aggregate_per_study_metadata_and_rollup():
    from reports._summary.aggregate import aggregate

    summ = aggregate(SLUG, WS)
    by = {s["slug"]: s for s in summ["studies"]}
    assert by["acetate"]["result"] == "FAIL"
    assert by["parca"]["result"] == "PASS"
    assert by["parca"]["prerequisites"] == []
    assert by["acetate"]["prerequisites"] == ["parca"]
    assert by["statistical"]["prerequisites"] == ["basal"]
    assert "RNA mass" in (by["acetate"]["finding"] or "")
    # config + standard cards discovered for acetate; parca card for parca
    assert {c["name"] for c in by["acetate"]["cards"]} == {"config", "standard"}
    assert {c["name"] for c in by["parca"]["cards"]} == {"parca"}
    # config card is ungraded, standard is graded
    acards = {c["name"]: c for c in by["acetate"]["cards"]}
    assert acards["config"]["graded"] is False
    assert acards["standard"]["graded"] is True
    assert acards["standard"]["overall"] == "mismatch"
    assert summ["rollup"] == {"PASS": 2, "PARTIAL": 3, "FAIL": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'reports._summary.aggregate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# reports/_summary/__init__.py
# (empty package marker)
```

```python
# reports/_summary/aggregate.py
"""Filesystem -> InvestigationSummary dict. No HTML, no sims, read-only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_UNGRADED = "ungraded"


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _canonical_result(study: dict) -> str | None:
    for run in study.get("runs", []) or []:
        if run.get("canonical"):
            return run.get("result")
    return None


def _first_finding(study: dict) -> str | None:
    for f in study.get("findings", []) or []:
        stmt = f.get("statement")
        if stmt:
            return " ".join(str(stmt).split())  # collapse folded-yaml whitespace
    return None


def _card_name(html_ref: str) -> str:
    return Path(html_ref).stem  # "viz/report_card/standard.html" -> "standard"


def _verdict_path(study_dir: Path, html_ref: str) -> Path:
    return study_dir / html_ref.replace(".html", ".verdict.json")


def _card_stub(study_dir: Path, html_ref: str) -> dict[str, Any]:
    name = _card_name(html_ref)
    vpath = _verdict_path(study_dir, html_ref)
    hpath = study_dir / html_ref
    overall = None
    missing = not hpath.exists()
    if vpath.exists():
        try:
            overall = json.loads(vpath.read_text()).get("overall")
        except (json.JSONDecodeError, OSError):
            overall = None
    else:
        missing = True
    return {
        "name": name,
        "overall": overall,
        "graded": overall not in (None, _UNGRADED),
        "html": "",
        "is_full_doc": False,
        "axes": [],
        "missing": missing,
    }


def aggregate(slug: str, workspace_root: str | Path) -> dict[str, Any]:
    ws = Path(workspace_root)
    inv_dir = ws / "investigations" / slug
    inv = _load_yaml(inv_dir / "investigation.yaml")

    studies: list[dict[str, Any]] = []
    rollup = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for study_slug in inv.get("studies", []) or []:
        study_dir = inv_dir / "studies" / study_slug
        study = _load_yaml(study_dir / "study.yaml")
        result = _canonical_result(study)
        if result in rollup:
            rollup[result] += 1
        cards = [_card_stub(study_dir, ref) for ref in study.get("report_cards", []) or []]
        studies.append({
            "slug": study_slug,
            "title": study.get("title") or study.get("name") or study_slug,
            "status": study.get("status"),
            "result": result,
            "prerequisites": (study.get("pipeline_gate", {}) or {}).get("prerequisites", []) or [],
            "finding": _first_finding(study),
            "cards": cards,
        })

    return {
        "slug": slug,
        "title": inv.get("title") or slug,
        "question": inv.get("question") or "",
        "studies": studies,
        "rollup": rollup,
        "matrix": {"columns": [], "rows": []},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add reports/_summary/__init__.py reports/_summary/aggregate.py tests/test_investigation_summary.py
git commit -m "feat(summary): aggregate investigation studies + per-study metadata"
```

---

### Task 2: Aggregate — verdict matrix

**Files:**
- Modify: `reports/_summary/aggregate.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Consumes: the `studies[]` list and each card's `graded`/`overall` from Task 1, plus each card's `axes` (populated here by reading the verdict groups directly — Task 3 also reads axes for the render side; both read the same `verdict.json`, so factor axis extraction into a shared helper `_graded_axes(study_dir, html_ref) -> list[dict]`).
- Produces: `summary["matrix"] = {"columns": [...], "rows": [...]}` per the shape above, and populates each card's `axes` list.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_investigation_summary.py
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
        (WS / "investigations" / SLUG / "studies" / "acetate"
         / "viz" / "report_card" / "standard.verdict.json").read_text()
    )
    axis = {a["label"]: a["verdict"] for a in src["groups"]["standard"]["axes"]}
    assert rows["acetate"]["growth rate (1/s)"] == axis["growth rate (1/s)"] == "mismatch"
    assert rows["acetate"]["RNA mass (fg)"] == "drift"
    # parca has cell mass within tolerance, no growth-rate column value
    assert rows["parca"]["cell mass (fg)"] == "within_tol"
    assert rows["parca"].get("growth rate (1/s)") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -k matrix -q`
Expected: FAIL — `m["columns"]` is empty (`[]`).

- [ ] **Step 3: Write minimal implementation**

Add the shared axis helper and matrix builder to `reports/_summary/aggregate.py`:

```python
def _graded_axes(study_dir: Path, html_ref: str) -> list[dict[str, Any]]:
    """Flattened axes across graded groups of one card's verdict.json."""
    vpath = _verdict_path(study_dir, html_ref)
    if not vpath.exists():
        return []
    try:
        data = json.loads(vpath.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("overall") in (None, _UNGRADED):
        return []
    axes: list[dict[str, Any]] = []
    for group in (data.get("groups") or {}).values():
        for a in group.get("axes", []) or []:
            axes.append({
                "label": a.get("label"),
                "verdict": a.get("verdict"),
                "value": a.get("value"),
                "meter": a.get("meter"),
            })
    return axes
```

In `aggregate()`, after building `cards`, attach axes and remember the html_ref so the matrix can be built. Change the card-building loop:

```python
        cards = []
        for ref in study.get("report_cards", []) or []:
            card = _card_stub(study_dir, ref)
            card["axes"] = _graded_axes(study_dir, ref)
            cards.append(card)
```

Then, before the final `return`, build the matrix from the assembled `studies`:

```python
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    for s in studies:
        cells: dict[str, str | None] = {}
        for card in s["cards"]:
            for axis in card["axes"]:
                label = axis["label"]
                if label and label not in columns:
                    columns.append(label)
                if label:
                    cells[label] = axis["verdict"]
        rows.append({"study": s["slug"], "cells": cells})
    matrix = {"columns": columns, "rows": rows}
```

And replace the placeholder `"matrix": {"columns": [], "rows": []}` in the returned dict with `"matrix": matrix`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add reports/_summary/aggregate.py tests/test_investigation_summary.py
git commit -m "feat(summary): build study x observable verdict matrix"
```

---

### Task 3: Aggregate — inline card HTML + full-doc detection

**Files:**
- Modify: `reports/_summary/aggregate.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Produces: each card's `html` (raw file contents, `""` if missing) and `is_full_doc` (`True` when the markup contains `<html`, case-insensitive) populated in `aggregate()`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_investigation_summary.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -k "full_doc or missing" -q`
Expected: FAIL — `is_full_doc` is `False` for statistical and `html` is `""`.

- [ ] **Step 3: Write minimal implementation**

In `reports/_summary/aggregate.py`, extend `_card_stub` to load the html and set `is_full_doc`:

```python
    html = ""
    if hpath.exists():
        try:
            html = hpath.read_text()
        except OSError:
            html = ""
            missing = True
    return {
        "name": name,
        "overall": overall,
        "graded": overall not in (None, _UNGRADED),
        "html": html,
        "is_full_doc": "<html" in html.lower(),
        "axes": [],
        "missing": missing,
    }
```

(Remove the old hardcoded `"html": ""`/`"is_full_doc": False` from the returned dict; keep `"axes": []` — axes are attached in `aggregate()` per Task 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add reports/_summary/aggregate.py tests/test_investigation_summary.py
git commit -m "feat(summary): inline card HTML and flag full-document cards"
```

---

### Task 4: Render — overview + verdict matrix HTML

**Files:**
- Create: `reports/_summary/render.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Consumes: the `InvestigationSummary` dict from Tasks 1–3 and a `style_css` string.
- Produces: `render(summary: dict, style_css: str = "") -> str` returning a full self-contained HTML document. Task 5 extends the same function with per-study sections; define it now to render `<head>` (with an inlined `<style>` block built from `style_css` + summary-specific rules), the overview header (title, question, rollup strip, pipeline DAG), and the verdict matrix table, and leave a clearly-marked insertion point (`# --- per-study sections (Task 5) ---`) before `</body>`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_investigation_summary.py
def test_render_overview_and_matrix():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS), style_css=":root{--x:1}")
    assert "<!doctype html>" in html.lower()
    assert "Does v2ecoli reproduce vEcoli" in html
    # rollup counts present
    assert "2 FAIL" in html and "3 PARTIAL" in html and "2 PASS" in html
    # matrix header + a verdict-colored cell class
    assert "growth rate (1/s)" in html
    assert "verdict-mismatch" in html
    # self-contained: no external stylesheet or script src
    assert "<link " not in html.lower()
    assert "script src" not in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -k render_overview -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'reports._summary.render'`.

- [ ] **Step 3: Write minimal implementation**

```python
# reports/_summary/render.py
"""InvestigationSummary dict -> self-contained HTML string. No filesystem reads."""
from __future__ import annotations

import html as _html
from typing import Any

_VERDICTS = ("within_tol", "drift", "mismatch", "ungraded")

_BASE_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;background:var(--bg,#fafafa);color:#1f2937;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:28px 24px 80px}
h1{font-size:1.6em;margin:0 0 4px}
.question{color:var(--gray,#666);font-size:1.05em;margin:0 0 18px}
.rollup{display:flex;gap:10px;margin:0 0 18px;flex-wrap:wrap}
.pill{border-radius:14px;padding:4px 12px;font-weight:600;font-size:0.9em;border:1px solid #0001}
.pill.PASS{background:#d1fae5;color:#065f46}.pill.PARTIAL{background:#fef3c7;color:#92400e}
.pill.FAIL{background:#fee2e2;color:#991b1b}
.dag{font-family:ui-monospace,Menlo,monospace;font-size:0.85em;color:#475569;
 background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;padding:10px 14px;margin:0 0 22px}
table.matrix{border-collapse:collapse;width:100%;margin:0 0 30px;font-size:0.85em}
table.matrix th,table.matrix td{border:1px solid var(--border,#e2e6eb);padding:6px 8px;text-align:center}
table.matrix th.study,table.matrix td.study{text-align:left;font-weight:600;white-space:nowrap}
td.verdict-within_tol{background:#d1fae5}td.verdict-drift{background:#fef3c7}
td.verdict-mismatch{background:#fee2e2}td.verdict-none{background:#f8fafc;color:#cbd5e1}
details.study{background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;margin:0 0 14px;padding:2px 14px}
details.study>summary{cursor:pointer;font-weight:600;padding:10px 0;list-style:none}
.badge{border-radius:10px;padding:1px 8px;font-size:0.78em;margin-left:8px}
.badge.within_tol{background:#d1fae5;color:#065f46}.badge.drift{background:#fef3c7;color:#92400e}
.badge.mismatch{background:#fee2e2;color:#991b1b}.badge.ungraded{background:#eef2f7;color:#475569}
.finding{color:var(--gray,#666);font-weight:400;font-size:0.9em;margin-left:6px}
.card-embed{margin:8px 0 14px;border-top:1px solid var(--border,#e2e6eb);padding-top:10px}
iframe.card-frame{width:100%;border:0}
.missing{color:#b91c1c;font-style:italic;font-size:0.9em}
"""


def _esc(s: Any) -> str:
    return _html.escape(str(s if s is not None else ""))


def _dag(summary: dict) -> str:
    parts = []
    for s in summary["studies"]:
        prereq = s["prerequisites"]
        arrow = f"{', '.join(prereq)} &rarr; " if prereq else ""
        parts.append(f"{arrow}<b>{_esc(s['slug'])}</b>")
    return "<br>".join(parts)


def _matrix_table(summary: dict) -> str:
    cols = summary["matrix"]["columns"]
    if not cols:
        return ""
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for row in summary["matrix"]["rows"]:
        cells = [f'<td class="study">{_esc(row["study"])}</td>']
        for c in cols:
            v = row["cells"].get(c)
            klass = f"verdict-{v}" if v in _VERDICTS else "verdict-none"
            cells.append(f'<td class="{klass}">{_esc(v or "")}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _rollup(summary: dict) -> str:
    r = summary["rollup"]
    order = [("FAIL", r["FAIL"]), ("PARTIAL", r["PARTIAL"]), ("PASS", r["PASS"])]
    pills = [f'<span class="pill {k}">{n} {k}</span>' for k, n in order]
    return f'<div class="rollup">{"".join(pills)}</div>'


def render(summary: dict, style_css: str = "") -> str:
    head = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(summary['title'])} — report-card summary</title>"
        f"<style>{style_css}\n{_BASE_CSS}</style></head><body><div class=\"wrap\">"
    )
    overview = (
        f"<h1>{_esc(summary['title'])}</h1>"
        f"<p class=\"question\">{_esc(summary['question'])}</p>"
        f"{_rollup(summary)}"
        f"<div class=\"dag\">{_dag(summary)}</div>"
        f"{_matrix_table(summary)}"
    )
    sections = ""  # --- per-study sections (Task 5) ---
    return head + overview + sections + "</div></body></html>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add reports/_summary/render.py tests/test_investigation_summary.py
git commit -m "feat(summary): render overview header + verdict matrix"
```

---

### Task 5: Render — per-study sections + card embedding

**Files:**
- Modify: `reports/_summary/render.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Consumes: each study's `cards[]` (with `html`, `is_full_doc`, `overall`, `graded`, `missing`) from Task 3.
- Produces: `sections` HTML inserted at the Task 4 marker. Fragment cards inline directly; full-document cards embed via `<iframe srcdoc="...">` with an auto-height script. Graded cards render `<details open>`; ungraded (config) cards render collapsed.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_investigation_summary.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -k per_study -q`
Expected: FAIL — no `id="study-..."` sections and no `srcdoc=`.

- [ ] **Step 3: Write minimal implementation**

In `reports/_summary/render.py`, add card + section renderers and an auto-height script, then wire them into `render()`:

```python
_IFRAME_JS = (
    "<script>window.addEventListener('load',function(){"
    "document.querySelectorAll('iframe.card-frame').forEach(function(f){"
    "try{f.style.height=(f.contentDocument.body.scrollHeight+20)+'px';}catch(e){}"
    "});});</script>"
)


def _card_html(card: dict) -> str:
    if card["missing"]:
        return f'<div class="missing">card &ldquo;{_esc(card["name"])}&rdquo; not rendered yet</div>'
    if card["is_full_doc"]:
        srcdoc = _html.escape(card["html"], quote=True)
        return f'<iframe class="card-frame" srcdoc="{srcdoc}"></iframe>'
    return card["html"]  # fragment: inline as-is (inline styles only, no collision)


def _study_section(study: dict) -> str:
    badge_verdict = None
    for c in study["cards"]:
        if c["graded"]:
            badge_verdict = c["overall"]
            break
    badge = (
        f'<span class="badge {badge_verdict}">{_esc(badge_verdict)}</span>'
        if badge_verdict else ""
    )
    finding = f'<span class="finding">{_esc(study["finding"])}</span>' if study["finding"] else ""
    embeds = []
    for c in study["cards"]:
        open_attr = " open" if c["graded"] else ""
        embeds.append(
            f'<details class="card-embed"{open_attr}>'
            f'<summary>{_esc(c["name"])} card</summary>{_card_html(c)}</details>'
        )
    return (
        f'<details class="study" id="study-{_esc(study["slug"])}" open>'
        f'<summary>{_esc(study["title"])}{badge}{finding}</summary>'
        f'{"".join(embeds)}</details>'
    )
```

Then update `render()`:

```python
    sections = "".join(_study_section(s) for s in summary["studies"])
    return head + overview + sections + _IFRAME_JS + "</div></body></html>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add reports/_summary/render.py tests/test_investigation_summary.py
git commit -m "feat(summary): per-study sections with inline + iframe card embeds"
```

---

### Task 6: CLI + end-to-end generation

**Files:**
- Create: `reports/investigation_summary.py`
- Test: `tests/test_investigation_summary.py`

**Interfaces:**
- Consumes: `aggregate()` and `render()`.
- Produces: `main(argv: list[str] | None = None) -> int` and a `__main__` guard. Reads `reports/assets/style.css` (best-effort; empty string if absent), passes its `:root` block through to `render()`, writes to `reports/summaries/<slug>_summary.html` (or `--out`), opens in a browser unless `--no-open`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_investigation_summary.py
def test_cli_writes_selfcontained_file(tmp_path):
    from reports.investigation_summary import main

    out = tmp_path / "summary.html"
    rc = main(["--investigation", SLUG, "--out", str(out), "--no-open"])
    assert rc == 0
    text = out.read_text()
    assert out.stat().st_size > 5000
    # all 7 studies present, self-contained
    for slug in ("parca", "basal", "with_aa", "succinate", "no_oxygen", "acetate", "statistical"):
        assert f'id="study-{slug}"' in text
    assert "<link " not in text.lower()
    assert "script src" not in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -k cli -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'reports.investigation_summary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# reports/investigation_summary.py
"""Generate a self-contained report-card summary for an investigation.

Usage:
    python reports/investigation_summary.py --investigation <slug> [--out PATH] [--no-open]
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports._summary.aggregate import aggregate  # noqa: E402
from reports._summary.render import render  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read_style() -> str:
    css = _REPO / "reports" / "assets" / "style.css"
    try:
        text = css.read_text()
    except OSError:
        return ""
    # keep only the :root{...} token block so the summary matches the report palette
    start = text.find(":root{")
    if start == -1:
        return ""
    end = text.find("}", start)
    return text[start:end + 1] if end != -1 else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Investigation report-card summary generator")
    ap.add_argument("--investigation", required=True, help="investigation slug")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--no-open", action="store_true", help="do not open in a browser")
    args = ap.parse_args(argv)

    ws = _REPO / "workspace"
    summary = aggregate(args.investigation, ws)
    html = render(summary, style_css=_read_style())

    out = Path(args.out) if args.out else (
        _REPO / "reports" / "summaries" / f"{args.investigation}_summary.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investigation_summary.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Generate the real summary and eyeball it**

Run: `.venv/bin/python reports/investigation_summary.py --investigation v2ecoli-vecoli-comparison`
Expected: prints `wrote reports/summaries/v2ecoli-vecoli-comparison_summary.html (...)` and opens the page. Verify: overview shows `2 FAIL · 3 PARTIAL · 2 PASS`, the matrix renders colored cells, and each study expands to show its embedded cards (statistical via iframe).

- [ ] **Step 6: Commit**

```bash
git add reports/investigation_summary.py tests/test_investigation_summary.py
git commit -m "feat(summary): CLI generator + end-to-end self-contained output"
```

Note: `reports/summaries/*.html` are generated artifacts. If the repo gitignores generated report output (check `.gitignore` for `reports/` render outputs, mirroring commit `459f69d4`), do **not** commit the generated HTML; otherwise leave it untracked.

---

## Self-Review

**Spec coverage:**
- Standalone generator + CLI + slug param → Task 6. ✓
- Default output `reports/summaries/<slug>_summary.html`, fully self-contained → Task 6 + Global Constraints; tests assert no external `<link>`/`script src`. ✓
- Data sources (investigation.yaml, study.yaml result/status/prereq/findings/report_cards, verdict.json, card html) → Tasks 1–3. ✓
- Overview (title, question, rollup, DAG) → Task 4. ✓
- Verdict matrix (union columns, first-appearance order, colored cells) → Task 2 (data) + Task 4 (render). ✓
- Per-study collapsible sections; graded open, config collapsed; inline fragments + iframe full-doc + auto-height → Task 5. ✓
- Missing-card placeholder rather than crash → Task 3 (`missing`) + Task 5 (`_card_html`). ✓
- Pure aggregate / render split → module structure. ✓
- Tests (7 studies, matrix matches source json, cards resolve) → Tasks 1–6. ✓

**Placeholder scan:** No TBD/TODO; the one `# --- per-study sections (Task 5) ---` marker is an intentional insertion anchor filled in Task 5. ✓

**Type consistency:** `aggregate(slug, workspace_root) -> dict` and `render(summary, style_css="") -> str` used identically across Tasks 4–6. Card keys (`name`, `overall`, `graded`, `html`, `is_full_doc`, `axes`, `missing`) match between aggregate (Tasks 1–3) and render (Tasks 4–5). Matrix keys (`columns`, `rows[].study`, `rows[].cells`) consistent between Task 2 and Task 4. ✓
