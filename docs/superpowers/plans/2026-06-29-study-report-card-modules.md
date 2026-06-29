# Per-study report-card modules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every v2ecoli study shows ≥1 report card on its dashboard detail page, produced by a pluggable card-module registry that reuses the existing `v2ecoli/library/report_card.py` — a universal `tests` card plus a `vs_vecoli` v2ecoli↔vEcoli equivalence card, both run-free.

**Architecture:** A small registry (`scripts/_cards/`) of card modules, each delegating all grading to the existing library. A generator CLI emits `workspace/studies/<name>/viz/report_card/<module>.{html,verdict.json}`, which the dashboard already auto-discovers (no dashboard changes). The `tests` module renders each study's own `tests:` block (verdict from recorded status); the `vs_vecoli` module stages a pre-generated comparison verdict JSON and renders it. Two new library helpers support this: a `status` criterion type and a `render_verdict_html` renderer.

**Tech Stack:** Python 3.12, the v2ecoli card library (`grade_card`/`verdict_json`/`card_criteria.grade_axis`), pytest, PyYAML.

## Global Constraints

- **Repo / branch:** worktree `/Users/eranagmon/code/v2e-report-cards`, branch `feat/study-report-card-modules` (off `origin/main`). All paths below are relative to this worktree root.
- **Test command (worktree has no venv — shadow the main install):** run every test as
  `PYTHONPATH=$PWD /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <path> -v` **from the worktree root**. This makes `import v2ecoli...` and `import scripts...` resolve to the worktree (verified). Define `V2EPY=/Users/eranagmon/code/v2ecoli/.venv/bin/python` for brevity.
- **Reuse, don't reinvent:** all grading/serialization goes through `v2ecoli/library/report_card.py` (`grade_card`, `verdict_json`) and `card_criteria.grade_axis`. No new grading math.
- **Dashboard contract (do not change the dashboard):** a card = `viz/report_card/<stem>.html` (required for discovery) + optional `<stem>.verdict.json` whose top-level `overall` drives the verdict pill. Verdict JSON schema is `report_card_verdict/v1` with `{overall, groups: {gslug: {verdict, axes: [{id,label,verdict,value,meter,detail}]}}}`.
- **Determinism (artifacts are committed):** never write a wall-clock timestamp into a committed card. Pass `generated=""` (the default) to `verdict_json`; `render_verdict_html` emits no timestamp. Re-running the generator on unchanged inputs must produce byte-identical output.
- **Graceful skip:** a module that cannot build logs a one-line skip and continues; it never aborts the whole generation run.
- **JSON safety:** verdict JSON is written with `allow_nan=False`; sanitize non-finite floats to `null` first (the published bundle's `JSON.parse` rejects `NaN`/`Infinity`).
- **Gitignore fact:** `workspace/studies/*/viz/` is **not** gitignored (only the unprefixed `studies/*/viz/` is), so generated cards under `workspace/studies/<name>/viz/report_card/` are tracked and committable.
- **Study layout:** studies live at `workspace/studies/<name>/study.yaml`; runs (when present) at `workspace/studies/<name>/runs.*.zarr`.

---

### Task 1: `status` criterion type in `card_criteria.grade_axis`

Lets an axis carry a pre-decided verdict (e.g. a study test's recorded pass/fail) through the standard grader, so `grade_card`/`verdict_json` work unchanged for the `tests` card.

**Files:**
- Modify: `v2ecoli/library/card_criteria.py` (add a branch in `grade_axis`, after `ctype = criterion.get("type")` near line 95)
- Test: `tests/test_status_criterion.py`

**Interfaces:**
- Consumes: `grade_axis(measured, criterion)` existing contract — returns `{verdict, value, criterion_str, meter, detail}`.
- Produces: criterion `{"type": "status", "criterion_str": <str>}`; the measured node is `{"verdict": <one of within_tol|drift|mismatch|ungraded>, "value": <any|None>, "meter": <str>, "detail": <dict>}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_status_criterion.py
from v2ecoli.library.card_criteria import grade_axis


def test_status_passes_through_verdict_and_fields():
    g = grade_axis(
        {"verdict": "within_tol", "value": 42.0, "meter": "ok", "detail": {"k": 1}},
        {"type": "status", "criterion_str": "in [35, 55]"},
    )
    assert g["verdict"] == "within_tol"
    assert g["value"] == 42.0
    assert g["criterion_str"] == "in [35, 55]"
    assert g["meter"] == "ok"
    assert g["detail"] == {"k": 1}


def test_status_unknown_verdict_is_ungraded():
    assert grade_axis({"verdict": "bogus"}, {"type": "status"})["verdict"] == "ungraded"


def test_status_missing_node_is_ungraded():
    g = grade_axis(None, {"type": "status"})
    assert g["verdict"] == "ungraded"
    assert g["value"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_status_criterion.py -v`
Expected: FAIL — `status` criterion returns the generic ungraded/None path (no `status` branch yet), so `test_status_passes_through_verdict_and_fields` fails on `verdict == "within_tol"`.

- [ ] **Step 3: Add the `status` branch**

In `v2ecoli/library/card_criteria.py`, immediately after the line `ctype = criterion.get("type")` inside `grade_axis`, insert:

```python
    if ctype == "status":
        # Verdict is carried by the measured node (pre-decided upstream, e.g. a
        # study test's recorded pass/fail). No numeric grading is done here.
        node = measured if isinstance(measured, dict) else {}
        v = node.get("verdict", "ungraded")
        if v not in VERDICTS:
            v = "ungraded"
        return {"verdict": v, "value": node.get("value"),
                "criterion_str": criterion.get("criterion_str", ""),
                "meter": node.get("meter", "—"), "detail": node.get("detail", {})}
```

(`VERDICTS` is already defined at module top: `("within_tol", "drift", "mismatch", "ungraded")`.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_status_criterion.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/card_criteria.py tests/test_status_criterion.py
git commit -m "feat(card): add 'status' criterion type (verdict carried by measured node)"
```

---

### Task 2: `render_verdict_html` renderer in `report_card.py`

One self-contained HTML renderer that turns a stored `report_card_verdict/v1` dict into a card (groups → sections, axes → rows). Shared by both modules; reuses the library's colour/glyph vocabulary so cards look consistent.

**Files:**
- Modify: `v2ecoli/library/report_card.py` (append a function after `render_html`, near line 772)
- Test: `tests/test_render_verdict_html.py`

**Interfaces:**
- Consumes: the verdict dict shape produced by `verdict_json(...)` and stored in `report_card_verdict.json`.
- Produces: `render_verdict_html(verdict: dict, *, title: str | None = None) -> str` — a self-contained HTML fragment (no external assets).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_verdict_html.py
from v2ecoli.library.report_card import render_verdict_html


def _vj():
    return {
        "schema": "report_card_verdict/v1", "overall": "drift",
        "reference_model": "vEcoli @ basal", "model_ref": "v2ecoli @ basal",
        "groups": {
            "standard": {"verdict": "drift", "axes": [
                {"id": "physiology.cell_mass", "label": "Cell mass",
                 "verdict": "within_tol", "value": 1.2, "meter": "Δ=+1%"},
                {"id": "physiology.growth_rate", "label": "Growth rate",
                 "verdict": "drift", "value": 0.9, "meter": "Δ=+7%"}]},
            "config": {"verdict": "within_tol", "axes": [
                {"id": "config.seeds", "label": "Seeds",
                 "verdict": "within_tol", "value": 4, "meter": ""}]},
        },
    }


def test_render_is_self_contained_with_groups_and_axes():
    html = render_verdict_html(_vj(), title="vEcoli ↔ v2ecoli (basal)")
    assert "<img" not in html and "src=" not in html        # no external assets
    assert "Cell mass" in html and "Growth rate" in html
    assert "Standard" in html and "Config" in html          # group headers, title-cased
    assert "vEcoli ↔ v2ecoli (basal)" in html               # title
    assert "overall" in html.lower()


def test_render_tolerates_missing_value_and_meter():
    vj = {"schema": "report_card_verdict/v1", "overall": "ungraded",
          "groups": {"tests": {"verdict": "ungraded", "axes": [
              {"id": "tests.t1", "label": "t1", "verdict": "ungraded"}]}}}
    html = render_verdict_html(vj)
    assert "t1" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_render_verdict_html.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_verdict_html'`.

- [ ] **Step 3: Implement the renderer**

Append to `v2ecoli/library/report_card.py` (after `render_html`):

```python
def render_verdict_html(verdict: dict, *, title: str | None = None) -> str:
    """Render a stored ``report_card_verdict/v1`` dict into a self-contained HTML
    card (no external assets): one section per group, one row per axis. Reuses the
    card colour/glyph vocabulary so it matches grade_card-rendered cards. Emits no
    timestamp (callers commit the output — keep it deterministic)."""
    import html as _html

    overall = verdict.get("overall", "ungraded")
    title = title or verdict.get("title") or "Report card"
    ref_model = verdict.get("reference_model", "")
    meas_model = verdict.get("model_ref", "")

    def chip(v: str, label: str | None = None) -> str:
        c = _COLOR.get(v, _COLOR["ungraded"])
        g = _GLYPH.get(v, _GLYPH["ungraded"])
        txt = _html.escape(label or v.replace("_", " "))
        return (f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;"
                f"background:{c};color:#fff;font-size:12px'>{g} {txt}</span>")

    def val_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, (int, float)):
            return _html.escape(f"{val:.4g}")
        return _html.escape(str(val))

    sections = []
    for gslug, grp in (verdict.get("groups") or {}).items():
        rows = []
        for ax in grp.get("axes", []):
            rows.append(
                "<tr>"
                f"<td style='padding:4px 10px'>{_html.escape(str(ax.get('label', ax.get('id', ''))))}</td>"
                f"<td style='padding:4px 10px'>{chip(ax.get('verdict', 'ungraded'))}</td>"
                f"<td style='padding:4px 10px;font-variant-numeric:tabular-nums'>{val_str(ax.get('value'))}</td>"
                f"<td style='padding:4px 10px;color:#555'>{_html.escape(str(ax.get('meter', '') or ''))}</td>"
                "</tr>")
        sections.append(
            f"<h3 style='margin:14px 0 4px'>{_html.escape(gslug.replace('_', ' ').title())} "
            f"{chip(grp.get('verdict', 'ungraded'))}</h3>"
            "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<thead><tr style='text-align:left;color:#888'>"
            "<th style='padding:4px 10px'>axis</th><th style='padding:4px 10px'>verdict</th>"
            "<th style='padding:4px 10px'>value</th><th style='padding:4px 10px'>meter</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")

    sub = " · ".join(x for x in [
        f"reference: {_html.escape(ref_model)}" if ref_model else "",
        f"measured: {_html.escape(meas_model)}" if meas_model else ""] if x)
    return (
        "<div style='font-family:system-ui,sans-serif;max-width:900px'>"
        f"<h2 style='margin:0 0 2px'>{_html.escape(title)} "
        f"{chip(overall, 'overall: ' + overall.replace('_', ' '))}</h2>"
        f"<div style='color:#888;font-size:12px;margin-bottom:8px'>{sub}</div>"
        f"{''.join(sections)}</div>")
```

(`_COLOR` and `_GLYPH` are module-level dicts already defined near the top of `report_card.py`.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_render_verdict_html.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/report_card.py tests/test_render_verdict_html.py
git commit -m "feat(card): add render_verdict_html (verdict JSON -> self-contained HTML)"
```

---

### Task 3: Card-module framework (registry + StudyContext + write/prune)

The pluggable core: a `StudyContext`, a `CardModule` protocol, write/prune helpers, and a registry. Modules register themselves on import.

**Files:**
- Create: `scripts/_cards/__init__.py`
- Create: `scripts/_cards/base.py`
- Test: `tests/test_cards_framework.py`

**Interfaces:**
- Produces:
  - `StudyContext(study_name, study_dir, spec, ws_root)` with classmethod `load(ws_root: Path, study_name: str) -> StudyContext`, method `run_zarr_paths() -> list[Path]`, property `card_dir -> Path` (`study_dir/viz/report_card`).
  - `CardModule` Protocol: attr `name: str`; `applies(ctx) -> bool`; `build(ctx) -> tuple[dict, str] | None` (returns `(verdict_json_dict, html_str)`).
  - `write_card(ctx, name, verdict, html) -> Path`; `prune(ctx, keep: set[str]) -> list[str]`.
  - `REGISTRY: dict[str, CardModule]`, `register(module) -> module`, `applicable(ctx, only=None) -> list[CardModule]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cards_framework.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._cards.base import StudyContext, write_card, prune


def _ctx(tmp_path, spec=None):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec or {"name": "demo"}))
    return StudyContext.load(tmp_path, "demo")


def test_studycontext_loads_spec_and_paths(tmp_path):
    ctx = _ctx(tmp_path, {"name": "Demo", "tests": [{"name": "t"}]})
    assert ctx.study_name == "demo"
    assert ctx.spec["name"] == "Demo"
    assert ctx.card_dir.name == "report_card"
    assert ctx.run_zarr_paths() == []


def test_write_card_writes_both_files(tmp_path):
    ctx = _ctx(tmp_path)
    p = write_card(ctx, "tests", {"overall": "within_tol"}, "<div>hi</div>")
    assert p.name == "tests.html"
    assert p.read_text() == "<div>hi</div>"
    vj = json.loads((ctx.card_dir / "tests.verdict.json").read_text())
    assert vj["overall"] == "within_tol"


def test_write_card_sanitizes_nonfinite(tmp_path):
    ctx = _ctx(tmp_path)
    write_card(ctx, "c", {"overall": "drift", "x": float("inf")}, "<i></i>")
    vj = json.loads((ctx.card_dir / "c.verdict.json").read_text())
    assert vj["x"] is None  # inf -> null (bundle-safe)


def test_prune_removes_stale_only(tmp_path):
    ctx = _ctx(tmp_path)
    write_card(ctx, "keep", {"overall": "within_tol"}, "<i></i>")
    write_card(ctx, "stale", {"overall": "within_tol"}, "<i></i>")
    pruned = prune(ctx, keep={"keep"})
    assert pruned == ["stale"]
    assert (ctx.card_dir / "keep.html").is_file()
    assert not (ctx.card_dir / "stale.html").is_file()
    assert not (ctx.card_dir / "stale.verdict.json").is_file()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_cards_framework.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._cards'`.

- [ ] **Step 3: Create `scripts/_cards/base.py`**

```python
# scripts/_cards/base.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats with None, recursively (bundle JSON.parse safe)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@dataclass
class StudyContext:
    study_name: str
    study_dir: Path
    spec: dict
    ws_root: Path

    @classmethod
    def load(cls, ws_root: Path, study_name: str) -> "StudyContext":
        sd = ws_root / "workspace" / "studies" / study_name
        spec_path = sd / "study.yaml"
        spec = {}
        if spec_path.is_file():
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        return cls(study_name=study_name, study_dir=sd, spec=spec, ws_root=ws_root)

    def run_zarr_paths(self) -> list[Path]:
        return sorted(self.study_dir.glob("runs.*.zarr"))

    @property
    def card_dir(self) -> Path:
        return self.study_dir / "viz" / "report_card"


class CardModule(Protocol):
    name: str

    def applies(self, ctx: StudyContext) -> bool: ...

    def build(self, ctx: StudyContext) -> "tuple[dict, str] | None": ...


def write_card(ctx: StudyContext, name: str, verdict: dict, html: str) -> Path:
    """Write <card>.html + <card>.verdict.json into the study's report_card dir.
    Returns the html path. Verdict is sanitized + written with allow_nan=False."""
    d = ctx.card_dir
    d.mkdir(parents=True, exist_ok=True)
    html_path = d / f"{name}.html"
    html_path.write_text(html, encoding="utf-8")
    (d / f"{name}.verdict.json").write_text(
        json.dumps(_sanitize(verdict), indent=1, allow_nan=False) + "\n",
        encoding="utf-8")
    return html_path


def prune(ctx: StudyContext, keep: set[str]) -> list[str]:
    """Delete <card>.html (+ sibling .verdict.json) under the study's report_card
    dir whose stem is not in `keep`. Returns pruned stems. Touches only that dir."""
    d = ctx.card_dir
    pruned: list[str] = []
    if not d.is_dir():
        return pruned
    for html in sorted(d.glob("*.html")):
        stem = html.name[: -len(".html")]
        if stem not in keep:
            html.unlink()
            vf = html.with_name(stem + ".verdict.json")
            if vf.is_file():
                vf.unlink()
            pruned.append(stem)
    return pruned
```

- [ ] **Step 4: Create `scripts/_cards/__init__.py`**

```python
# scripts/_cards/__init__.py
from __future__ import annotations

from .base import CardModule, StudyContext, prune, write_card  # noqa: F401

REGISTRY: dict[str, CardModule] = {}


def register(module: CardModule) -> CardModule:
    REGISTRY[module.name] = module
    return module


def applicable(ctx: StudyContext, only: str | None = None) -> list[CardModule]:
    """Modules to emit for a study. If the study spec lists `report_cards:`, only
    those names are eligible; otherwise every registered module is eligible. A
    module is emitted when eligible AND its applies(ctx) is True. `only` (a module
    name, or None/'all') narrows to a single module."""
    declared = ctx.spec.get("report_cards")
    want = None if (only in (None, "all")) else {only}
    out: list[CardModule] = []
    for nm, mod in REGISTRY.items():
        if want is not None and nm not in want:
            continue
        if declared is not None and nm not in declared:
            continue
        if mod.applies(ctx):
            out.append(mod)
    return out


# Register built-in modules (import for side effect; added in Tasks 4 & 5).
try:  # keep the framework importable before the modules exist (TDD ordering)
    from . import tests_card  # noqa: E402,F401
except Exception:  # noqa: BLE001
    pass
try:
    from . import vs_vecoli_card  # noqa: E402,F401
except Exception:  # noqa: BLE001
    pass
```

- [ ] **Step 5: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_cards_framework.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/_cards/__init__.py scripts/_cards/base.py tests/test_cards_framework.py
git commit -m "feat(cards): card-module registry + StudyContext + write/prune helpers"
```

---

### Task 4: `tests` card module (universal, run-free)

Renders each study's own `tests:` block as a card; verdict comes from each test's recorded `status`. Applies to every study that has tests → guarantees ≥1 card per study.

**Files:**
- Create: `scripts/_cards/tests_card.py`
- Test: `tests/test_tests_card.py`

**Interfaces:**
- Consumes: `StudyContext` (Task 3); `grade_card`, `verdict_json` (library); `render_verdict_html` (Task 2); `register` (Task 3).
- Produces: `TestsCard` (`name = "tests"`), registered. `build` returns `(verdict_json_dict, html_str)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tests_card.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._cards.base import StudyContext
from scripts._cards.tests_card import TestsCard


def _ctx(tmp_path, tests):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": "Demo", "tests": tests}))
    return StudyContext.load(tmp_path, "demo")


def test_one_axis_per_test_overall_is_worst(tmp_path):
    ctx = _ctx(tmp_path, [
        {"name": "doubling-time-in-band", "classification": "primary",
         "status": "passed", "pass_if": {"op": "in_range", "low": 35, "high": 55}},
        {"name": "mass-fraction", "classification": "primary",
         "status": "failed", "pass_if": {"op": "in_range", "low": 0.40, "high": 0.55}},
    ])
    m = TestsCard()
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["overall"] == "mismatch"               # worst of pass + fail
    assert "doubling-time-in-band" in html and "mass-fraction" in html
    assert "in [35, 55]" in html                        # criterion string surfaced


def test_absent_when_no_tests(tmp_path):
    assert TestsCard().applies(_ctx(tmp_path, [])) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_tests_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._cards.tests_card'`.

- [ ] **Step 3: Create `scripts/_cards/tests_card.py`**

```python
# scripts/_cards/tests_card.py
from __future__ import annotations

import re
from typing import Any

from scripts._cards import register
from scripts._cards.base import StudyContext
from v2ecoli.library.report_card import grade_card, render_verdict_html, verdict_json

_STATUS_TO_VERDICT = {
    "passed": "within_tol", "pass": "within_tol", "within_tol": "within_tol",
    "failed": "mismatch", "fail": "mismatch", "mismatch": "mismatch",
    "drift": "drift",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_") or "test"


def _criterion_str(pass_if: dict) -> str:
    op = pass_if.get("op")
    if op == "in_range":
        return f"in [{pass_if.get('low')}, {pass_if.get('high')}]"
    if op in ("gt", "ge", "lt", "le", "eq"):
        sym = {"gt": ">", "ge": "≥", "lt": "<", "le": "≤", "eq": "="}[op]
        return f"{sym} {pass_if.get('value', pass_if.get('threshold', ''))}"
    return op or ""


class TestsCard:
    name = "tests"

    def applies(self, ctx: StudyContext) -> bool:
        return bool(ctx.spec.get("tests"))

    def build(self, ctx: StudyContext):
        tests = ctx.spec.get("tests") or []
        if not tests:
            return None
        reference_axes: dict[str, Any] = {}
        card: dict[str, Any] = {"tests": {}}
        for t in tests:
            tname = t.get("name", "test")
            slug = _slug(tname)
            path = f"tests.{slug}"
            status = str(t.get("status", "")).lower()
            verdict = _STATUS_TO_VERDICT.get(status, "ungraded")
            group = (t.get("classification") or "tests").capitalize()
            crit_str = _criterion_str(t.get("pass_if") or {})
            measure = t.get("measure") or {}
            value = measure.get("value")  # present only if the study recorded one
            detail = t.get("question") or measure.get("detail") or ""
            reference_axes[path] = {
                "label": tname, "group": group,
                "criterion": {"type": "status", "criterion_str": crit_str},
            }
            card["tests"][slug] = {
                "verdict": verdict, "value": value,
                "meter": crit_str, "detail": {"text": detail},
            }
        reference = {
            "title": f"{ctx.spec.get('name', ctx.study_name)} — default tests",
            "stimulus": {"reference_model": "behavioral spec",
                         "measured_model": "v2ecoli"},
            "axes": reference_axes,
        }
        report = grade_card(card, reference)
        vjson = verdict_json(report, model_ref="v2ecoli",
                             reference_model="behavioral spec")
        vjson["title"] = reference["title"]
        html = render_verdict_html(vjson, title=reference["title"])
        return vjson, html


register(TestsCard())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_tests_card.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_cards/tests_card.py tests/test_tests_card.py
git commit -m "feat(cards): tests module — study tests -> universal run-free card"
```

---

### Task 5: `vs_vecoli` card module (equivalence, run-free staging)

Stages a pre-generated v2ecoli↔vEcoli comparison verdict JSON (the `standard`+`config` groups) into the study slot and renders it. Applies only when the study declares `report_card_refs.vs_vecoli`. Phase 2 regenerates the source verdict from fresh GovCloud runs; this module re-stages whatever exists.

**Files:**
- Create: `scripts/_cards/vs_vecoli_card.py`
- Test: `tests/test_vs_vecoli_card.py`

**Interfaces:**
- Consumes: `StudyContext` (Task 3); `render_verdict_html` (Task 2); `register` (Task 3).
- Produces: `VsVecoliCard` (`name = "vs_vecoli"`), registered. Reads `ctx.spec["report_card_refs"]["vs_vecoli"]` (a path relative to `ws_root`, or absolute) pointing at a `report_card_verdict.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vs_vecoli_card.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._cards.base import StudyContext
from scripts._cards.vs_vecoli_card import VsVecoliCard


def _ctx(tmp_path, refs=None):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    spec = {"name": "Demo"}
    if refs:
        spec["report_card_refs"] = refs
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    return StudyContext.load(tmp_path, "demo")


def _write_verdict(tmp_path):
    p = tmp_path / "docs" / "rc" / "basal" / "report_card_verdict.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "schema": "report_card_verdict/v1", "overall": "drift",
        "reference_model": "vEcoli @ basal", "model_ref": "v2ecoli @ basal",
        "groups": {"standard": {"verdict": "drift", "axes": [
            {"id": "physiology.cell_mass", "label": "Cell mass",
             "verdict": "within_tol", "value": 1.2, "meter": ""}]}}}))
    return "docs/rc/basal/report_card_verdict.json"


def test_absent_without_ref(tmp_path):
    assert VsVecoliCard().applies(_ctx(tmp_path)) is False


def test_absent_when_ref_missing_file(tmp_path):
    ctx = _ctx(tmp_path, refs={"vs_vecoli": "docs/rc/nope.json"})
    assert VsVecoliCard().applies(ctx) is False


def test_renders_from_declared_verdict(tmp_path):
    rel = _write_verdict(tmp_path)
    ctx = _ctx(tmp_path, refs={"vs_vecoli": rel})
    m = VsVecoliCard()
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["overall"] == "drift"
    assert "Cell mass" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_vs_vecoli_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._cards.vs_vecoli_card'`.

- [ ] **Step 3: Create `scripts/_cards/vs_vecoli_card.py`**

```python
# scripts/_cards/vs_vecoli_card.py
from __future__ import annotations

import json
from pathlib import Path

from scripts._cards import register
from scripts._cards.base import StudyContext
from v2ecoli.library.report_card import render_verdict_html


class VsVecoliCard:
    name = "vs_vecoli"

    def _verdict_path(self, ctx: StudyContext) -> "Path | None":
        rel = (ctx.spec.get("report_card_refs") or {}).get("vs_vecoli")
        if not rel:
            return None
        p = Path(rel) if str(rel).startswith("/") else (ctx.ws_root / rel)
        return p if p.is_file() else None

    def applies(self, ctx: StudyContext) -> bool:
        return self._verdict_path(ctx) is not None

    def build(self, ctx: StudyContext):
        vp = self._verdict_path(ctx)
        if vp is None:
            return None
        vjson = json.loads(vp.read_text(encoding="utf-8"))
        title = vjson.get("title") or (
            f"vEcoli ↔ v2ecoli — {ctx.spec.get('name', ctx.study_name)}")
        html = render_verdict_html(vjson, title=title)
        return vjson, html


register(VsVecoliCard())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_vs_vecoli_card.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_cards/vs_vecoli_card.py tests/test_vs_vecoli_card.py
git commit -m "feat(cards): vs_vecoli module — stage+render pre-generated comparison verdict"
```

---

### Task 6: Generator CLI (`scripts/study_report_cards.py`)

Iterates the registry over one or all studies, writes cards, optionally prunes. The single entry point the workspace/CI calls.

**Files:**
- Create: `scripts/study_report_cards.py`
- Test: `tests/test_study_report_cards_cli.py`

**Interfaces:**
- Consumes: `REGISTRY`, `applicable`, `write_card`, `prune`, `StudyContext` (Tasks 3–5).
- Produces: `generate_study(ws_root: Path, name: str, only: str | None, do_prune: bool) -> dict` (returns `{"study", "written": [stems]}`); `main(argv=None) -> int` with flags `--study {all|<name>}`, `--card {all|<name>}`, `--prune`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_study_report_cards_cli.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.study_report_cards as cli


def _study(tmp_path, name, spec):
    sd = tmp_path / "workspace" / "studies" / name
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    return sd


def test_generate_study_emits_tests_card(tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "classification": "primary", "status": "passed",
         "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", only=None, do_prune=True)
    assert "tests" in r["written"]
    rc = sd / "viz" / "report_card"
    assert (rc / "tests.html").is_file()
    assert json.loads((rc / "tests.verdict.json").read_text())["overall"] == "within_tol"


def test_only_filters_to_one_module(tmp_path):
    _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", only="vs_vecoli", do_prune=False)
    assert r["written"] == []          # tests excluded by --card vs_vecoli; no ref -> none


def test_prune_drops_stale_card(tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    rc = sd / "viz" / "report_card"
    rc.mkdir(parents=True)
    (rc / "old.html").write_text("<i></i>")
    cli.generate_study(tmp_path, "demo", only=None, do_prune=True)
    assert not (rc / "old.html").is_file()   # stale pruned
    assert (rc / "tests.html").is_file()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_study_report_cards_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.study_report_cards'`.

- [ ] **Step 3: Create `scripts/study_report_cards.py`**

```python
# scripts/study_report_cards.py
"""Generate per-study report cards via the modular card-module registry.

Each registered module (tests, vs_vecoli, ...) emits
``workspace/studies/<name>/viz/report_card/<module>.{html,verdict.json}``, which
the dashboard auto-discovers (no dashboard changes). The ``tests`` module is
universal and run-free; ``vs_vecoli`` stages a pre-generated v2ecoli<->vEcoli
comparison verdict (declared per study via ``report_card_refs.vs_vecoli``).

Usage:
  python scripts/study_report_cards.py --study all [--card all] [--prune]
  python scripts/study_report_cards.py --study showcase-2-baseline-figures --card vs_vecoli
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._cards import applicable, prune, write_card  # noqa: E402
from scripts._cards.base import StudyContext  # noqa: E402


def _all_studies(ws_root: Path) -> list[str]:
    sdir = ws_root / "workspace" / "studies"
    if not sdir.is_dir():
        return []
    return sorted(p.name for p in sdir.iterdir() if (p / "study.yaml").is_file())


def generate_study(ws_root: Path, name: str, only: str | None,
                   do_prune: bool) -> dict:
    ctx = StudyContext.load(ws_root, name)
    written: list[str] = []
    for mod in applicable(ctx, only=only):
        try:
            res = mod.build(ctx)
        except Exception as e:  # noqa: BLE001 — one module never aborts the run
            print(f"  ! {name}/{mod.name}: skip ({e})")
            continue
        if not res:
            continue
        vjson, html = res
        write_card(ctx, mod.name, vjson, html)
        written.append(mod.name)
        print(f"  ✓ {name}/{mod.name} [{vjson.get('overall', '?')}]")
    if do_prune:
        for s in prune(ctx, keep=set(written)):
            print(f"  - {name}/{s}: pruned")
    return {"study": name, "written": written}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="all", help="study name or 'all'")
    ap.add_argument("--card", default="all", help="module name or 'all'")
    ap.add_argument("--prune", action="store_true",
                    help="delete report_card/* not produced this run")
    args = ap.parse_args(argv)
    studies = _all_studies(REPO_ROOT) if args.study == "all" else [args.study]
    only = None if args.card == "all" else args.card
    total = 0
    for s in studies:
        total += len(generate_study(REPO_ROOT, s, only, args.prune)["written"])
    print(f"done — {total} cards across {len(studies)} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_study_report_cards_cli.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole new suite together**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_status_criterion.py tests/test_render_verdict_html.py tests/test_cards_framework.py tests/test_tests_card.py tests/test_vs_vecoli_card.py tests/test_study_report_cards_cli.py -v`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add scripts/study_report_cards.py tests/test_study_report_cards_cli.py
git commit -m "feat(cards): generator CLI (study_report_cards.py)"
```

---

### Task 7: Wire studies, generate real cards, verify, commit

Declare the comparison reference on the two equivalence studies, generate cards for all studies, verify the dashboard discovers them, and commit the artifacts. This is the integration deliverable — the cards actually appear.

**Files:**
- Modify: `workspace/studies/showcase-2-baseline-figures/study.yaml` (add `report_card_refs`)
- Modify: `workspace/studies/showcase-6-equivalence-large/study.yaml` (add `report_card_refs`)
- Create (generated, committed): `workspace/studies/*/viz/report_card/*.{html,verdict.json}`

**Interfaces:**
- Consumes: the generator CLI (Task 6); the existing comparison verdicts at `docs/report_cards/v2ecoli-vecoli-comparison/<cond>/report_card_verdict.json`.

- [ ] **Step 1: Confirm the comparison verdict sources exist**

Run: `ls docs/report_cards/v2ecoli-vecoli-comparison/basal/report_card_verdict.json docs/report_cards/v2ecoli-vecoli-comparison/basal_4x4/report_card_verdict.json`
Expected: both paths listed (these are the pre-generated comparison verdicts to stage). If `basal_4x4` is absent, use `basal` for showcase-6 too.

- [ ] **Step 2: Declare the vs_vecoli reference on showcase-2**

Add this top-level block to `workspace/studies/showcase-2-baseline-figures/study.yaml` (place it after the `status:` line; match the file's existing 2-space indentation):

```yaml
report_card_refs:
  vs_vecoli: docs/report_cards/v2ecoli-vecoli-comparison/basal/report_card_verdict.json
```

- [ ] **Step 3: Declare the vs_vecoli reference on showcase-6**

Add to `workspace/studies/showcase-6-equivalence-large/study.yaml` (use `basal_4x4` if Step 1 confirmed it, else `basal`):

```yaml
report_card_refs:
  vs_vecoli: docs/report_cards/v2ecoli-vecoli-comparison/basal_4x4/report_card_verdict.json
```

- [ ] **Step 4: Generate cards for all studies**

Run: `PYTHONPATH=$PWD $V2EPY scripts/study_report_cards.py --study all --prune`
Expected: a `✓ <study>/tests [...]` line for every study that has a `tests:` block, plus `✓ showcase-2-baseline-figures/vs_vecoli [...]` and `✓ showcase-6-equivalence-large/vs_vecoli [...]`. Ends with `done — N cards across 24 studies`.

- [ ] **Step 5: Verify the artifacts on disk**

Run:
```bash
ls workspace/studies/showcase-2-baseline-figures/viz/report_card/
PYTHONPATH=$PWD $V2EPY -c "import json,glob; [print(p, '->', json.load(open(p)).get('overall')) for p in sorted(glob.glob('workspace/studies/*/viz/report_card/*.verdict.json'))]"
```
Expected: `tests.html tests.verdict.json vs_vecoli.html vs_vecoli.verdict.json` for showcase-2; each verdict.json prints a valid `overall` (one of within_tol/drift/mismatch/ungraded). No tracebacks.

- [ ] **Step 6: Verify the dashboard discovers them**

The dashboard reader is installed in the venv. Its public function is
`build_saved_visualizations(ws_root) -> dict` (returns a `report_cards` list),
and v2ecoli's `workspace.yaml` is at the repo root with
`layout.studies: workspace/studies` — so `ws_root` is the repo root, and the
reader iterates the exact dirs the generator wrote to. Confirm:
```bash
$V2EPY -c "
from pathlib import Path
from vivarium_dashboard.lib.saved_visualizations import build_saved_visualizations
payload = build_saved_visualizations(Path('$PWD'))
rc = payload['report_cards']
print('report_cards found:', len(rc))
print('studies with cards:', sorted({c['study'] for c in rc})[:6])
names = {c['name'] for c in rc}
assert 'tests' in names, 'tests card not discovered'
assert 'vs_vecoli' in names, 'vs_vecoli card not discovered'
print('OK')
"
```
Expected: prints `report_cards found: <N>` (N ≥ number of studies with tests, +2 for the vs_vecoli cards), a study list, and `OK`. (No `PYTHONPATH=$PWD` here — we want the *installed* dashboard, not a worktree shadow; only `v2ecoli`/`scripts` need shadowing, and this call imports neither.)

- [ ] **Step 7: Confirm regeneration is deterministic (no churn)**

Run:
```bash
PYTHONPATH=$PWD $V2EPY scripts/study_report_cards.py --study all --prune
git status --porcelain workspace/studies/*/viz/report_card/ | head
```
Expected: after a second generation, `git status` shows **no** modifications to already-committed cards from this step (only the first-generation additions). If any card shows as modified on a no-input-change re-run, a non-deterministic field leaked in — fix before committing.

- [ ] **Step 8: Commit the wiring + generated cards**

```bash
git add workspace/studies/showcase-2-baseline-figures/study.yaml \
        workspace/studies/showcase-6-equivalence-large/study.yaml \
        workspace/studies/*/viz/report_card/
git commit -m "feat(studies): generate per-study report cards (tests + vs_vecoli)

Declare the v2ecoli<->vEcoli comparison reference on the two equivalence
studies and generate cards for all studies via scripts/study_report_cards.py.
Every study now shows a tests card; showcase-2 and showcase-6 also show the
vs_vecoli equivalence card. Dashboard discovery + publish staging already
exist (no dashboard changes)."
```

---

## Out of scope (Phase 2 — not in this plan)

- New GovCloud comparison runs (`comparison_harness.sh` / `launch_full_comparison.sh`) to regenerate the `vs_vecoli` source verdicts across all 5 conditions.
- Moving card generation into the publish-dashboard CI workflow (so cards regenerate instead of being committed). Phase 1 commits them for immediate visibility.
- Re-evaluating `tests` cards against live run data (Phase 1 grades from recorded `status`).
