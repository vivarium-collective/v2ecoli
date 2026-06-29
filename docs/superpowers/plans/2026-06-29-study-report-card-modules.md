# Per-study report-card modules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every v2ecoli study shows ≥1 report card on its dashboard detail page, produced by a pluggable registry of report-card Steps that reuses the existing `v2ecoli/library/report_card.py` — a universal `tests` card plus a `vs_vecoli` v2ecoli↔vEcoli equivalence card, both run-free.

**Architecture:** Report cards are **visualization-like process-bigraph Steps** — `ReportCardStep(V2Step)`, a sibling of `v2ecoli/workflow/analysis.py`'s `Analysis`, with output ports `{"view": "string" (HTML), "data": "map" (verdict)}`. Subclasses auto-register in `REPORT_CARD_REGISTRY`. A runner CLI builds a `bigraph_schema` core, runs each applicable Step over a `StudyContext`, and writes its `view`→`<name>.html` + `data`→`<name>.verdict.json` into `workspace/studies/<name>/viz/report_card/`, which the dashboard already auto-discovers (no dashboard changes). The `tests` card renders each study's own `tests:` block (verdict from recorded status); the `vs_vecoli` card stages a pre-generated comparison verdict JSON and renders it. Two library helpers support this: a `status` criterion type and a `render_verdict_html` renderer.

**Tech Stack:** Python 3.12, process-bigraph Steps (`V2Step`, `bigraph_schema.allocate_core`), the v2ecoli card library (`grade_card`/`verdict_json`/`card_criteria.grade_axis`), pytest, PyYAML.

## Global Constraints

- **Repo / branch:** worktree `/Users/eranagmon/code/v2e-report-cards`, branch `feat/study-report-card-modules` (off `origin/main`). All paths below are relative to this worktree root.
- **Test command (worktree has no venv — shadow the main install):** run every test as
  `PYTHONPATH=$PWD /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <path> -v` **from the worktree root**. This makes `import v2ecoli...` and `import scripts...` resolve to the worktree (verified). Define `V2EPY=/Users/eranagmon/code/v2ecoli/.venv/bin/python` for brevity.
- **Reuse, don't reinvent:** all grading/serialization goes through `v2ecoli/library/report_card.py` (`grade_card`, `verdict_json`) and `card_criteria.grade_axis`. No new grading math.
- **Report cards are Steps:** each card is a `ReportCardStep(V2Step)` subclass (in `v2ecoli/workflow/report_cards/`) with output ports `{"view": "string", "data": "map"}`, mirroring `v2ecoli/workflow/analysis.py`'s `Analysis`. Steps instantiate as `cls(config_dict, core=core)` where `core = bigraph_schema.allocate_core()` (built once per runner invocation; ~5s). Subclasses auto-register in `REPORT_CARD_REGISTRY` via `__init_subclass__`. Tests use the existing `core` pytest fixture (`tests/conftest.py`).
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

### Task 3: `ReportCardStep` base + registry + StudyContext (pluggable core)

Report cards as process-bigraph Steps: a `ReportCardStep(V2Step)` base with `{view, data}` output ports, a `REPORT_CARD_REGISTRY` (auto-registration via `__init_subclass__`), a `StudyContext`, `write_card`/`prune` helpers, and an `applicable()` selector. This mirrors `v2ecoli/workflow/analysis.py`'s `Analysis(V2Step)`.

**Files:**
- Create: `v2ecoli/workflow/report_cards/__init__.py`
- Test: `tests/test_report_card_step.py`

**Interfaces:**
- Consumes: `V2Step` (from `v2ecoli.steps.base`); `bigraph_schema.allocate_core` (in tests, the `core` pytest fixture from `tests/conftest.py`).
- Produces:
  - `StudyContext(study_name, study_dir, spec, ws_root)` — classmethod `load(ws_root: Path, study_name: str) -> StudyContext`; `run_zarr_paths() -> list[Path]`; property `card_dir -> Path` (= `study_dir/viz/report_card`).
  - `ReportCardStep(V2Step)` — attr `name: str`; `inputs() -> {"study": "any"}`; `outputs() -> {"view": "string", "data": "map"}`; `applies(study) -> bool` (default `True`); `build(study) -> tuple[dict, str] | None`; `update(state, interval=None) -> {"view": html, "data": verdict}`. Any subclass that sets `name` auto-registers in `REPORT_CARD_REGISTRY`.
  - `REPORT_CARD_REGISTRY: dict[str, type]`.
  - `write_card(ctx, name, verdict, html) -> Path`; `prune(ctx, keep: set[str]) -> list[str]`.
  - `applicable(ctx, core, only=None) -> list[ReportCardStep]` — instantiated Steps to emit for the study (honors the study's optional `report_cards:` allowlist and `applies()`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_card_step.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.steps.base import V2Step
from v2ecoli.workflow.report_cards import (
    REPORT_CARD_REGISTRY, ReportCardStep, StudyContext, applicable, prune, write_card)


class _DemoCard(ReportCardStep):
    name = "demo_card"

    def applies(self, study):
        return bool(study.spec.get("demo"))

    def build(self, study):
        return ({"schema": "report_card_verdict/v1", "overall": "drift"},
                "<div>demo</div>")


def _ctx(tmp_path, spec=None):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec or {"name": "demo"}))
    return StudyContext.load(tmp_path, "demo")


def test_reportcardstep_is_v2step_with_view_data_ports(core):
    step = _DemoCard({}, core=core)
    assert isinstance(step, V2Step)
    assert step.outputs() == {"view": "string", "data": "map"}
    assert step.inputs() == {"study": "any"}


def test_subclass_auto_registers():
    assert REPORT_CARD_REGISTRY.get("demo_card") is _DemoCard


def test_update_returns_view_and_data(core, tmp_path):
    ctx = _ctx(tmp_path, {"name": "demo", "demo": True})
    out = _DemoCard({}, core=core).update({"study": ctx})
    assert out["view"] == "<div>demo</div>"
    assert out["data"]["overall"] == "drift"


def test_studycontext_loads_spec_and_paths(tmp_path):
    ctx = _ctx(tmp_path, {"name": "Demo", "tests": [{"name": "t"}]})
    assert ctx.study_name == "demo"
    assert ctx.spec["name"] == "Demo"
    assert ctx.card_dir.name == "report_card"
    assert ctx.run_zarr_paths() == []


def test_write_card_writes_both_files_and_sanitizes(tmp_path):
    ctx = _ctx(tmp_path)
    p = write_card(ctx, "tests", {"overall": "drift", "x": float("inf")}, "<i>hi</i>")
    assert p.name == "tests.html"
    assert p.read_text() == "<i>hi</i>"
    vj = json.loads((ctx.card_dir / "tests.verdict.json").read_text())
    assert vj["overall"] == "drift"
    assert vj["x"] is None  # inf -> null (bundle-safe)


def test_prune_removes_stale_only(tmp_path):
    ctx = _ctx(tmp_path)
    write_card(ctx, "keep", {"overall": "within_tol"}, "<i></i>")
    write_card(ctx, "stale", {"overall": "within_tol"}, "<i></i>")
    assert prune(ctx, keep={"keep"}) == ["stale"]
    assert (ctx.card_dir / "keep.html").is_file()
    assert not (ctx.card_dir / "stale.html").is_file()
    assert not (ctx.card_dir / "stale.verdict.json").is_file()


def test_applicable_selects_by_applies_and_allowlist(core, tmp_path):
    on = _ctx(tmp_path, {"name": "demo", "demo": True})
    # only='demo_card' isolates from other registered cards; applies() True here
    assert [s.name for s in applicable(on, core, only="demo_card")] == ["demo_card"]
    off = _ctx(tmp_path, {"name": "demo"})  # no 'demo' key -> applies() False
    assert applicable(off, core, only="demo_card") == []
    # explicit report_cards allowlist excluding demo_card -> not emitted
    excl = _ctx(tmp_path, {"name": "demo", "demo": True, "report_cards": ["tests"]})
    assert applicable(excl, core, only="demo_card") == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_report_card_step.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.report_cards'`.

- [ ] **Step 3: Create `v2ecoli/workflow/report_cards/__init__.py`**

```python
# v2ecoli/workflow/report_cards/__init__.py
"""Report cards as visualization-like process-bigraph Steps.

A report card is a ``ReportCardStep`` — a sibling of
``v2ecoli/workflow/analysis.py``'s ``Analysis(V2Step)`` with the same HTML output
port. It emits a rendered ``view`` (the card HTML) plus ``data`` (the verdict_json
map). Unlike ``Analysis`` — which consumes a live DuckDB sim-output connection — a
report card's input is a ``StudyContext`` (the study's spec + dir), so cards grade
run-free. Subclasses that set ``name`` auto-register in ``REPORT_CARD_REGISTRY``.

The runner (``scripts/study_report_cards.py``) builds a ``bigraph_schema`` core,
instantiates each registered card, calls ``applies``/``build``, and writes the
``view`` → ``viz/report_card/<name>.html`` and ``data`` → ``<name>.verdict.json``
(the files the dashboard discovers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from v2ecoli.steps.base import V2Step

REPORT_CARD_REGISTRY: dict[str, type] = {}


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


class ReportCardStep(V2Step):
    """A report card as a visualization-like Step (sibling of ``Analysis``):
    emits ``view`` (HTML) + ``data`` (verdict map). Subclasses set ``name`` and
    implement ``applies(study)`` + ``build(study) -> (verdict_dict, html) | None``.
    """

    name: str = ""
    config_schema: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            REPORT_CARD_REGISTRY[cls.name] = cls

    def inputs(self):
        return {"study": "any"}

    def outputs(self):
        return {"view": "string", "data": "map"}

    def applies(self, study: "StudyContext") -> bool:
        return True

    def build(self, study: "StudyContext") -> "tuple[dict, str] | None":
        """Return ``(verdict_json_dict, html_str)`` or None. Subclasses override."""
        raise NotImplementedError

    def update(self, state, interval=None):
        study = state.get("study")
        res = self.build(study) if study is not None else None
        if not res:
            return {"view": "", "data": {}}
        verdict, html = res
        return {"view": html, "data": verdict}


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


def applicable(ctx: StudyContext, core, only: "str | None" = None) -> list:
    """Instantiated report-card Steps to emit for a study. If the study spec lists
    `report_cards:`, only those names are eligible; otherwise every registered card
    is eligible. A card is emitted when eligible AND its applies(ctx) is True.
    `only` (a name, or None/'all') narrows to a single card. `core` is a
    bigraph-schema core (built once by the caller) used to instantiate Steps."""
    declared = ctx.spec.get("report_cards")
    want = None if (only in (None, "all")) else {only}
    out = []
    for nm, cls in REPORT_CARD_REGISTRY.items():
        if want is not None and nm not in want:
            continue
        if declared is not None and nm not in declared:
            continue
        step = cls({}, core=core)
        if step.applies(ctx):
            out.append(step)
    return out


# Register built-in cards (import for side effect; added in Tasks 4 & 5). Guarded
# so the package imports cleanly before those modules exist (TDD ordering).
try:
    from . import tests_card  # noqa: E402,F401
except Exception:  # noqa: BLE001
    pass
try:
    from . import vs_vecoli_card  # noqa: E402,F401
except Exception:  # noqa: BLE001
    pass
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_report_card_step.py -v`
Expected: PASS (7 passed). (The `core` fixture builds a `bigraph_schema` core; expect a few "skipping optional dep" warnings from core allocation — those are pre-existing and not from this code.)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/report_cards/__init__.py tests/test_report_card_step.py
git commit -m "feat(cards): ReportCardStep base + REPORT_CARD_REGISTRY + StudyContext"
```

---

### Task 4: `tests` card Step (universal, run-free)

`TestsCard(ReportCardStep)` renders each study's own `tests:` block; verdict comes from each test's recorded `status`. Applies to every study with tests → guarantees ≥1 card per study.

**Files:**
- Create: `v2ecoli/workflow/report_cards/tests_card.py`
- Test: `tests/test_tests_card.py`

**Interfaces:**
- Consumes: `ReportCardStep`, `StudyContext` (Task 3); `grade_card`, `verdict_json` (library); `render_verdict_html` (Task 2).
- Produces: `TestsCard(ReportCardStep)` (`name = "tests"`), auto-registered. `build` returns `(verdict_json_dict, html_str)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tests_card.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.tests_card import TestsCard


def _ctx(tmp_path, tests):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": "Demo", "tests": tests}))
    return StudyContext.load(tmp_path, "demo")


def test_one_axis_per_test_overall_is_worst(core, tmp_path):
    ctx = _ctx(tmp_path, [
        {"name": "doubling-time-in-band", "classification": "primary",
         "status": "passed", "pass_if": {"op": "in_range", "low": 35, "high": 55}},
        {"name": "mass-fraction", "classification": "primary",
         "status": "failed", "pass_if": {"op": "in_range", "low": 0.40, "high": 0.55}},
    ])
    m = TestsCard({}, core=core)
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["overall"] == "mismatch"               # worst of pass + fail
    assert "doubling-time-in-band" in html and "mass-fraction" in html
    assert "in [35, 55]" in html                        # criterion string surfaced


def test_absent_when_no_tests(core, tmp_path):
    assert TestsCard({}, core=core).applies(_ctx(tmp_path, [])) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_tests_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.report_cards.tests_card'`.

- [ ] **Step 3: Create `v2ecoli/workflow/report_cards/tests_card.py`**

```python
# v2ecoli/workflow/report_cards/tests_card.py
from __future__ import annotations

import re
from typing import Any

from v2ecoli.library.report_card import grade_card, render_verdict_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext

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


class TestsCard(ReportCardStep):
    name = "tests"

    def applies(self, study: StudyContext) -> bool:
        return bool(study.spec.get("tests"))

    def build(self, study: StudyContext):
        tests = study.spec.get("tests") or []
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
            "title": f"{study.spec.get('name', study.study_name)} — default tests",
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
```

(Importing `from v2ecoli.workflow.report_cards import ReportCardStep` while that package's `__init__` is importing this module is safe: `ReportCardStep` is defined before the guarded `from . import tests_card` at the bottom of `__init__`, so the name resolves from the partially-initialized package.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_tests_card.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/report_cards/tests_card.py tests/test_tests_card.py
git commit -m "feat(cards): TestsCard Step — study tests -> universal run-free card"
```

---

### Task 5: `vs_vecoli` card Step (equivalence, run-free staging)

`VsVecoliCard(ReportCardStep)` stages a pre-generated v2ecoli↔vEcoli comparison verdict JSON (the `standard`+`config` groups) and renders it. Applies only when the study declares `report_card_refs.vs_vecoli`. Phase 2 regenerates the source verdict from fresh runs; this Step re-stages whatever exists.

**Files:**
- Create: `v2ecoli/workflow/report_cards/vs_vecoli_card.py`
- Test: `tests/test_vs_vecoli_card.py`

**Interfaces:**
- Consumes: `ReportCardStep`, `StudyContext` (Task 3); `render_verdict_html` (Task 2).
- Produces: `VsVecoliCard(ReportCardStep)` (`name = "vs_vecoli"`), auto-registered. Reads `study.spec["report_card_refs"]["vs_vecoli"]` (a path relative to `ws_root`, or absolute) pointing at a `report_card_verdict.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vs_vecoli_card.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.vs_vecoli_card import VsVecoliCard


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


def test_absent_without_ref(core, tmp_path):
    assert VsVecoliCard({}, core=core).applies(_ctx(tmp_path)) is False


def test_absent_when_ref_missing_file(core, tmp_path):
    ctx = _ctx(tmp_path, refs={"vs_vecoli": "docs/rc/nope.json"})
    assert VsVecoliCard({}, core=core).applies(ctx) is False


def test_renders_from_declared_verdict(core, tmp_path):
    rel = _write_verdict(tmp_path)
    ctx = _ctx(tmp_path, refs={"vs_vecoli": rel})
    m = VsVecoliCard({}, core=core)
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["overall"] == "drift"
    assert "Cell mass" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_vs_vecoli_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.report_cards.vs_vecoli_card'`.

- [ ] **Step 3: Create `v2ecoli/workflow/report_cards/vs_vecoli_card.py`**

```python
# v2ecoli/workflow/report_cards/vs_vecoli_card.py
from __future__ import annotations

import json
from pathlib import Path

from v2ecoli.library.report_card import render_verdict_html
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class VsVecoliCard(ReportCardStep):
    name = "vs_vecoli"

    def _verdict_path(self, study: StudyContext) -> "Path | None":
        rel = (study.spec.get("report_card_refs") or {}).get("vs_vecoli")
        if not rel:
            return None
        p = Path(rel) if str(rel).startswith("/") else (study.ws_root / rel)
        return p if p.is_file() else None

    def applies(self, study: StudyContext) -> bool:
        return self._verdict_path(study) is not None

    def build(self, study: StudyContext):
        vp = self._verdict_path(study)
        if vp is None:
            return None
        vjson = json.loads(vp.read_text(encoding="utf-8"))
        title = vjson.get("title") or (
            f"vEcoli ↔ v2ecoli — {study.spec.get('name', study.study_name)}")
        html = render_verdict_html(vjson, title=title)
        return vjson, html
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_vs_vecoli_card.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/report_cards/vs_vecoli_card.py tests/test_vs_vecoli_card.py
git commit -m "feat(cards): VsVecoliCard Step — stage+render pre-generated comparison verdict"
```

---

### Task 6: Runner CLI (`scripts/study_report_cards.py`)

Builds a core once, runs the registered report-card Steps over one or all studies, writes their `view`/`data` to `viz/report_card/`, optionally prunes. The entry point that makes cards show now.

> **Interim:** this runner is a thin, report-card-only driver so cards appear immediately. The next sub-project — the unified **post-simulation analysis flush** (one extraction over the run's emitters → dispatch to all post-sim Steps: Analyses, Visualizations, ReportCards) — will absorb this runner as its ReportCard dispatch. Keep it small and registry-driven so that absorption is clean; do not grow sim-output/extraction logic here (that belongs to the flush).

**Files:**
- Create: `scripts/study_report_cards.py`
- Test: `tests/test_study_report_cards_cli.py`

**Interfaces:**
- Consumes: `REPORT_CARD_REGISTRY`, `applicable`, `write_card`, `prune`, `StudyContext` (Task 3); `bigraph_schema.allocate_core`.
- Produces: `generate_study(ws_root: Path, name: str, core, only: str | None, do_prune: bool) -> dict` (returns `{"study", "written": [names]}`); `main(argv=None) -> int` with flags `--study {all|<name>}`, `--card {all|<name>}`, `--prune`.

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


def test_generate_study_emits_tests_card(core, tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "classification": "primary", "status": "passed",
         "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", core, only=None, do_prune=True)
    assert "tests" in r["written"]
    rc = sd / "viz" / "report_card"
    assert (rc / "tests.html").is_file()
    assert json.loads((rc / "tests.verdict.json").read_text())["overall"] == "within_tol"


def test_only_filters_to_one_card(core, tmp_path):
    _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", core, only="vs_vecoli", do_prune=False)
    assert r["written"] == []   # tests excluded by --card vs_vecoli; no ref -> none


def test_prune_drops_stale_card(core, tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    rc = sd / "viz" / "report_card"
    rc.mkdir(parents=True)
    (rc / "old.html").write_text("<i></i>")
    cli.generate_study(tmp_path, "demo", core, only=None, do_prune=True)
    assert not (rc / "old.html").is_file()    # stale pruned
    assert (rc / "tests.html").is_file()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_study_report_cards_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.study_report_cards'`.

- [ ] **Step 3: Create `scripts/study_report_cards.py`**

```python
# scripts/study_report_cards.py
"""Generate per-study report cards by running the report-card Steps.

Each registered ``ReportCardStep`` (tests, vs_vecoli, ...) emits a ``view`` (HTML)
+ ``data`` (verdict map); this runner writes them to
``workspace/studies/<name>/viz/report_card/<name>.{html,verdict.json}``, which the
dashboard auto-discovers (no dashboard changes). The ``tests`` card is universal
and run-free; ``vs_vecoli`` stages a pre-generated v2ecoli<->vEcoli comparison
verdict (declared per study via ``report_card_refs.vs_vecoli``).

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

from bigraph_schema import allocate_core  # noqa: E402

from v2ecoli.workflow.report_cards import (  # noqa: E402
    applicable, prune, write_card)


def _all_studies(ws_root: Path) -> list[str]:
    sdir = ws_root / "workspace" / "studies"
    if not sdir.is_dir():
        return []
    return sorted(p.name for p in sdir.iterdir() if (p / "study.yaml").is_file())


def generate_study(ws_root: Path, name: str, core, only: "str | None",
                   do_prune: bool) -> dict:
    from v2ecoli.workflow.report_cards import StudyContext
    ctx = StudyContext.load(ws_root, name)
    written: list[str] = []
    for step in applicable(ctx, core, only=only):
        try:
            res = step.build(ctx)
        except Exception as e:  # noqa: BLE001 — one card never aborts the run
            print(f"  ! {name}/{step.name}: skip ({e})")
            continue
        if not res:
            continue
        vjson, html = res
        write_card(ctx, step.name, vjson, html)
        written.append(step.name)
        print(f"  ✓ {name}/{step.name} [{vjson.get('overall', '?')}]")
    if do_prune:
        for s in prune(ctx, keep=set(written)):
            print(f"  - {name}/{s}: pruned")
    return {"study": name, "written": written}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="all", help="study name or 'all'")
    ap.add_argument("--card", default="all", help="card name or 'all'")
    ap.add_argument("--prune", action="store_true",
                    help="delete report_card/* not produced this run")
    args = ap.parse_args(argv)
    studies = _all_studies(REPO_ROOT) if args.study == "all" else [args.study]
    only = None if args.card == "all" else args.card
    core = allocate_core()
    total = 0
    for s in studies:
        total += len(generate_study(REPO_ROOT, s, core, only, args.prune)["written"])
    print(f"done — {total} cards across {len(studies)} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_study_report_cards_cli.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole new suite together**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_status_criterion.py tests/test_render_verdict_html.py tests/test_report_card_step.py tests/test_tests_card.py tests/test_vs_vecoli_card.py tests/test_study_report_cards_cli.py -v`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add scripts/study_report_cards.py tests/test_study_report_cards_cli.py
git commit -m "feat(cards): runner CLI (study_report_cards.py) — run report-card Steps"
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

## Out of scope (later — not in this plan)

- **Unified post-simulation analysis flush (the immediate next sub-project, its own spec+plan).** Formalize an extraction step (emitters → a shared extracted-run context: per-cell records + DuckDB conn + `sim_data`) and a flush orchestrator that, in one pass, dispatches that context to every registered post-sim Step sharing the `view`(HTML)+`data` contract — Analyses, Visualizations, and ReportCards — generalizing `v2ecoli/workflow/analysis_runner.run_analyses`. The Task-6 runner here is the interim ReportCard-only driver the flush will absorb.
- New GovCloud comparison runs (`comparison_harness.sh` / `launch_full_comparison.sh`) to regenerate the `vs_vecoli` source verdicts across all 5 conditions.
- Moving card generation into the publish-dashboard CI workflow (so cards regenerate instead of being committed). This plan commits them for immediate visibility.
- Re-evaluating `tests` cards against live run data (this plan grades from recorded `status`).
