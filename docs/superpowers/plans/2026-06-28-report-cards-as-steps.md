# Report cards as `as_step` types — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make each report card a process-bigraph `@as_step` Step registered in the bigraph-schema core, replacing the bespoke `@report_card` registry; the harness resolves cards via `core.access` and invokes `update(state)`.

**Architecture:** Convert the 5 card functions to `update_<card>_report_card(state)` decorated with `as_step(inputs, outputs)`; collect them into `REPORT_CARD_STEPS = {name: StepCls}`; register via `core.register_links` in `build_core`. The harness builds a typed `state` from the study's comparison data, calls `step.update(state)` → `{card_html, verdict, axes}`, and writes `viz/report_card/`. Pure Steps; dashboard + contract unchanged.

**Tech Stack:** Python 3, process-bigraph (`as_step`, already a dep), bigraph-schema core, pytest.

**Repo:** `/Users/eranagmon/code/v2e-main` (branch `feat/report-cards-as-steps`). Run tests via `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-28-report-cards-as-steps-design.md`

## Global Constraints

- `as_step(inputs, outputs, name=None, aliases=None)` from `process_bigraph.composite`; the decorated function MUST be named `update_*`; it returns a dict matching `outputs`. The result is a `Step` subclass.
- Card Step name = `<card>_report_card` (e.g. `standard_report_card`), alias = `<card>` (e.g. `standard`). The 5 cards: `config`, `parca`, `standard`, `statistical`, `config_diff`.
- Outputs of every card: `{card_html: str, verdict: str, axes: list}` — `verdict` ∈ {within_tol, drift, mismatch, ungraded}; `config`/`config_diff` output `verdict="ungraded"`, `axes=[]`.
- Registration: `core.register_links(REPORT_CARD_STEPS)` in `v2ecoli/core.py::build_core`; resolution `core.access("<card>_report_card")`.
- The Step is pure (returns outputs); the harness writes `viz/report_card/<card>.{html,verdict.json}` and the per-condition verdict. Verdict VALUES must be identical to today for the same input.
- All reads/writes `encoding="utf-8"`. Run via the repo `.venv`. Never auto-merge.

---

### Task 1: Pin the schemas + as_step/core round-trip + `_sections_to_html`

**Files:**
- Modify: `scripts/_compare/report_cards/__init__.py`
- Test: `tests/test_report_card_steps.py` (create)

**Interfaces:**
- Produces: `CARD_INPUTS` / `CARD_OUTPUTS` (validated bigraph-schema dicts); `_sections_to_html(sections) -> str`; `REPORT_CARD_STEPS: dict[str, type]` (empty for now). These are additive — the existing `@report_card`/`REGISTRY`/`render`/`CardContext` stay until Task 5.

- [ ] **Step 1: Confirm the bigraph type syntax for the loose inputs**

Run (pin the exact strings the rest of the plan uses):
```
.venv/bin/python -c "
from v2ecoli.core import build_core
c = build_core()
for t in ['string','integer','tree[any]','list[float]','overwrite[string]','overwrite[list[tree[any]]]','map']:
    try: c.access(t); print('OK ', t)
    except Exception as e: print('BAD', t, type(e).__name__)
"
```
Expected: `string`, `integer`, `tree[any]`, `list[float]`, `overwrite[string]` resolve. If `tree[any]`/`map`/`overwrite[list[tree[any]]]` print `BAD`, fall back to the simplest resolving form (`tree[any]` for nested, `tree[any]` for the axes output) and use THOSE strings verbatim in every later task. Record the chosen strings in the task report.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_report_card_steps.py
from process_bigraph.composite import as_step
from v2ecoli.core import build_core
from scripts._compare.report_cards import _sections_to_html, CARD_INPUTS, CARD_OUTPUTS


def test_sections_to_html_renders_html_and_rows():
    html = _sections_to_html([
        {"title": "A", "html": "<b>hi</b>"},
        {"title": "B", "rows": [{"label": "x", "left": "1", "right": "2",
                                 "verdict": "within_tol", "reason": "ok"}]},
    ])
    assert html.lstrip().startswith("<")
    assert "<b>hi</b>" in html and "A" in html
    assert "x" in html and "within_tol" in html        # rows rendered as a table


def test_as_step_card_round_trips_through_core():
    @as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="probe_report_card",
             aliases=["probe"])
    def update_probe_report_card(state):
        return {"card_html": f"<p>{state['name']}</p>", "verdict": "within_tol", "axes": []}
    core = build_core()
    core.register_link("probe_report_card", update_probe_report_card)
    step = core.access("probe_report_card")(config={}, core=core)
    out = step.update({"name": "basal", "condition": "basal", "seeds": 1,
                       "generations": 4, "variant": 0, "observables": {},
                       "plot_trajs": {}, "v2_bounds": [], "config": {},
                       "v2_dir": "", "ve_dir": ""})
    assert out["verdict"] == "within_tol" and "basal" in out["card_html"]
```

- [ ] **Step 3: Run it — expect FAIL** (`ImportError: CARD_INPUTS`)

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py -q`

- [ ] **Step 4: Add the schemas + helper to `__init__.py`**

Insert near the top (after `Section = dict`), using the strings PINNED in Step 1 (shown here with the expected defaults):

```python
import html as _html

# Typed contract for a report-card Step. Pragmatic: structural fields typed;
# the per-seed stat records under `observables` stay loose. Strings pinned in
# Task 1 Step 1 — substitute the validated forms if any printed BAD.
CARD_INPUTS = {
    "name": "string", "condition": "string",
    "seeds": "integer", "generations": "integer", "variant": "integer",
    "observables": "tree[any]", "plot_trajs": "tree[any]",
    "v2_bounds": "list[float]", "config": "tree[any]",
    "v2_dir": "string", "ve_dir": "string",
}
CARD_OUTPUTS = {
    "card_html": "overwrite[string]",
    "verdict": "overwrite[string]",
    "axes": "overwrite[tree[any]]",
}

REPORT_CARD_STEPS: dict[str, type] = {}   # {name: StepCls}; populated by the card modules


def _row_table(rows: list) -> str:
    cells = []
    for r in rows:
        label = _html.escape(str(r.get("label", "")))
        left = _html.escape(str(r.get("left", "")))
        right = _html.escape(str(r.get("right", "")))
        verdict = _html.escape(str(r.get("verdict", "")))
        reason = _html.escape(str(r.get("reason", "")))
        cells.append(
            f'<tr><td style="padding:2px 10px">{label}</td>'
            f'<td style="padding:2px 10px">{left}</td>'
            f'<td style="padding:2px 10px">{right}</td>'
            f'<td style="padding:2px 10px">{verdict}</td>'
            f'<td style="padding:2px 10px;color:#6b7280">{reason}</td></tr>')
    return ('<table style="border-collapse:collapse;font-size:13px">'
            '<thead><tr style="text-align:left">'
            '<th style="padding:2px 10px">observable</th><th>vEcoli</th>'
            '<th>v2ecoli</th><th>verdict</th><th>note</th></tr></thead><tbody>'
            + "".join(cells) + "</tbody></table>')


def _sections_to_html(sections: list) -> str:
    """Render a card's section dicts into one HTML fragment. A section with an
    `html` field is emitted as-is; a section with `rows` is rendered as a
    table (eval_section / parca_section produce rows)."""
    parts = []
    for sec in sections:
        if sec.get("title"):
            parts.append(f'<h3 style="margin:14px 0 6px">{_html.escape(str(sec["title"]))}</h3>')
        if sec.get("desc"):
            parts.append(f'<p style="color:#6b7280;font-size:12px">{_html.escape(str(sec["desc"]))}</p>')
        if sec.get("html"):
            parts.append(sec["html"])
        elif sec.get("rows"):
            parts.append(_row_table(sec["rows"]))
    return "".join(parts)
```

- [ ] **Step 5: Run it — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/report_cards/__init__.py tests/test_report_card_steps.py
git commit -m "feat(report-cards): pin Step schemas + _sections_to_html + core round-trip"
```

---

### Task 2: Convert `standard` + `parca` to `as_step` (rows-based graded cards)

**Files:**
- Modify: `scripts/_compare/report_cards/standard.py`, `scripts/_compare/report_cards/parca.py`
- Test: `tests/test_report_card_steps.py` (extend)

**Interfaces:**
- Consumes: `CARD_INPUTS`, `CARD_OUTPUTS`, `_sections_to_html`, `REPORT_CARD_STEPS`, `worst`.
- Produces: `update_standard_report_card`/`update_parca_report_card` Step classes registered in `REPORT_CARD_STEPS`. KEEP the existing `@report_card` wrapper in each (dual-registered, transitional — removed in Task 5) so `assemble_from_studies` + the old tests stay green.

- [ ] **Step 1: Write the failing test** (extend `tests/test_report_card_steps.py`)

```python
def _state(per_obs, name="basal", seeds=1, gens=4, variant=0, config=None):
    return {"name": name, "condition": "basal", "seeds": seeds, "generations": gens,
            "variant": variant, "observables": per_obs, "plot_trajs": {},
            "v2_bounds": [], "config": config or {}, "v2_dir": "", "ve_dir": ""}


# 5 seeds so the t-test / median have data; one within-tol observable.
_PO = {"rna_mass": [{"median_rel": 0.02, "max_rel": 0.05, "init_ve": 100.0,
                     "init_v2": 101.0, "init_t": 60.0, "ve_mean": 100.0, "v2_mean": 101.0}
                    for _ in range(5)]}


def _run_card(name, state):
    from v2ecoli.core import build_core
    from scripts._compare.report_cards import REPORT_CARD_STEPS
    core = build_core(); core.register_links(REPORT_CARD_STEPS)
    step = core.access(f"{name}_report_card")(config={}, core=core)
    return step.update(state)


def test_standard_step_grades_and_renders():
    out = _run_card("standard", _state(_PO))
    assert out["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    assert any(a["id"].startswith("standard.") for a in out["axes"])
    assert "<" in out["card_html"]


def test_parca_step_grades_initial_state():
    out = _run_card("parca", _state(_PO))
    assert out["verdict"] == "within_tol"
    assert any(a["id"].startswith("parca.") for a in out["axes"])
```

- [ ] **Step 2: Run it — expect FAIL** (`standard_report_card` not registered)

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py::test_standard_step_grades_and_renders -q`

- [ ] **Step 3: Rewrite `standard.py`** (add the Step + register; keep the old wrapper)

```python
"""`standard` card — matched-time runs + per-observable evaluation. as_step Step;
the harness invokes it via core.access('standard_report_card')."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)
from scripts._compare.verdict import worst


def _standard_sections_and_axes(name, per_obs, plot_trajs, v2_bounds):
    from scripts.comparison_report_card import runs_section, eval_section
    runs = runs_section(name, per_obs, plot_trajs, v2_bounds)
    ev = eval_section(name, per_obs)
    axes = []
    for row in ev.get("rows", []):
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({"id": f"standard.{row.get('label', '')}", "label": row.get("label", ""),
                     "verdict": v, "value": row.get("median_rel"),
                     "meter": row.get("reason", ""),
                     "detail": {"median_rel": row.get("median_rel"),
                                "max_rel": row.get("max_rel")}})
    return [runs, ev], axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="standard_report_card",
         aliases=["standard"])
def update_standard_report_card(state):
    sections, axes = _standard_sections_and_axes(
        state["name"], state["observables"], state["plot_trajs"], state["v2_bounds"])
    return {"card_html": _sections_to_html(sections),
            "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["standard_report_card"] = update_standard_report_card


# --- transitional: keep the old registry wrapper until the Task 5 cutover ---
@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    sections, axes = _standard_sections_and_axes(
        ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    ev = sections[1]
    ev["verdict"] = worst(a["verdict"] for a in axes)
    ev["verdict_axes"] = axes
    return sections
```

- [ ] **Step 4: Rewrite `parca.py`** (add the Step; keep the old wrapper)

```python
"""`parca` card — ParCa / initial-state match, graded on per-mass t~0 |Δ|.
as_step Step invoked via core.access('parca_report_card')."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)
from scripts._compare.verdict import worst


def _parca_section_and_axes(name, per_obs, plot_trajs, v2_bounds):
    from scripts.comparison_report_card import parca_section
    sec = parca_section({name: (per_obs, plot_trajs, v2_bounds)})
    sec["anchor"] = f"{name}-parca"
    sec["title"] = f"{name} — ParCa / initial-state match"
    axes = []
    for row in sec.get("rows", []):
        if "median_rel" not in row:
            continue
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({"id": f"parca.{row['label']}", "label": row["label"], "verdict": v,
                     "value": row.get("median_rel"), "meter": row.get("reason", ""),
                     "detail": {"init_rel": row.get("median_rel")}})
    return sec, axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="parca_report_card",
         aliases=["parca"])
def update_parca_report_card(state):
    sec, axes = _parca_section_and_axes(
        state["name"], state["observables"], state["plot_trajs"], state["v2_bounds"])
    return {"card_html": _sections_to_html([sec]),
            "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["parca_report_card"] = update_parca_report_card


@report_card("parca")
def parca_card(ctx: CardContext) -> Section:
    sec, axes = _parca_section_and_axes(
        ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    sec["verdict"] = worst(a["verdict"] for a in axes)
    sec["verdict_axes"] = axes
    return sec
```

- [ ] **Step 5: Run the new + existing card tests**

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py tests/test_card_verdicts.py -q`
Expected: PASS (the Step tests + the old-wrapper tests both green — dual registration).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/report_cards/standard.py scripts/_compare/report_cards/parca.py tests/test_report_card_steps.py
git commit -m "feat(report-cards): standard + parca as_step Steps (dual-registered)"
```

---

### Task 3: Convert `statistical` + `config` + `config_diff` to `as_step`

**Files:**
- Modify: `scripts/_compare/report_cards/statistical.py`, `config.py`, `config_diff.py`
- Test: `tests/test_report_card_steps.py` (extend)

**Interfaces:**
- Consumes: the Task 1 schemas/helpers; `REPORT_CARD_STEPS`.
- Produces: `update_statistical_report_card`, `update_config_report_card`, `update_config_diff_report_card` Steps in `REPORT_CARD_STEPS`; old `@report_card` wrappers kept (transitional).

- [ ] **Step 1: Write the failing test**

```python
def test_statistical_step_grades():
    out = _run_card("statistical", _state(_PO, name="statistical", seeds=4))
    assert out["verdict"] == "within_tol"
    assert out["axes"]


def test_config_step_is_ungraded_and_renders_config():
    out = _run_card("config", _state({}, name="basal", config={"condition": "basal"}))
    assert out["verdict"] == "ungraded" and out["axes"] == []
    assert "basal" in out["card_html"]
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py::test_statistical_step_grades -q`

- [ ] **Step 3: Rewrite `statistical.py`**

```python
"""`statistical` card — graded equivalence (violin/strip). as_step Step."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS)
from scripts._compare.report_card_section import build_report_card


def _statistical_html_axes(name, per_obs, variant):
    from scripts.comparison_report_card import OBSERVABLES, CARD_KEY, EXTRA_AXES, TOL
    left, right = {}, {}
    for obs in OBSERVABLES:
        ck = CARD_KEY.get(obs, obs)
        left[ck] = [s["ve_mean"] for s in per_obs.get(obs, [])]
        right[ck] = [s["v2_mean"] for s in per_obs.get(obs, [])]
    vjson, html = build_report_card(left, right, extra_axes=EXTRA_AXES,
                                    model_ref=f"v2ecoli @ {name} variant {variant}", tol_rel=TOL)
    axes = [ax for g in (vjson.get("groups") or {}).values() for ax in (g.get("axes") or [])]
    return html, vjson.get("overall"), axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="statistical_report_card",
         aliases=["statistical"])
def update_statistical_report_card(state):
    html, verdict, axes = _statistical_html_axes(
        state["name"], state["observables"], state["variant"])
    return {"card_html": html, "verdict": verdict or "ungraded", "axes": axes}


REPORT_CARD_STEPS["statistical_report_card"] = update_statistical_report_card


@report_card("statistical")
def statistical_card(ctx: CardContext) -> Section:
    html, verdict, axes = _statistical_html_axes(ctx.config_name, ctx.per_obs, ctx.variant)
    return {"title": f"{ctx.config_name} — statistical equivalence", "kind": "content",
            "anchor": f"{ctx.config_name}-statistical", "html": html,
            "verdict": verdict, "verdict_axes": axes}
```

- [ ] **Step 4: Rewrite `config.py`** — wrap the existing body in `_config_html(name, seeds, gens, config)`, then:

Keep the existing rendering logic but move it into a helper `_config_html(name, seeds, gens, config) -> str` (replace every `ctx.config_name`→`name`, `ctx.seeds`→`seeds`, `ctx.gens`→`gens`, `ctx.config`→`config`). Then add:

```python
from process_bigraph.composite import as_step
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS

@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="config_report_card",
         aliases=["config"])
def update_config_report_card(state):
    return {"card_html": _config_html(state["name"], state["seeds"],
                                      state["generations"], state["config"]),
            "verdict": "ungraded", "axes": []}

REPORT_CARD_STEPS["config_report_card"] = update_config_report_card


@report_card("config")
def config_card(ctx: CardContext) -> Section:
    return {"title": f"{ctx.config_name} — config", "kind": "content",
            "anchor": f"{ctx.config_name}-config",
            "html": _config_html(ctx.config_name, ctx.seeds, ctx.gens, ctx.config),
            "verdict": None}
```

- [ ] **Step 5: Rewrite `config_diff.py`**

```python
"""`config_diff` card — vEcoli vs v2ecoli config comparison (S3/Nextflow). as_step Step."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="config_diff_report_card",
         aliases=["config_diff"])
def update_config_diff_report_card(state):
    from scripts.comparison_report_card import config_sections_for
    secs = config_sections_for(state["name"], state["v2_dir"], state["ve_dir"])
    return {"card_html": _sections_to_html(secs), "verdict": "ungraded", "axes": []}


REPORT_CARD_STEPS["config_diff_report_card"] = update_config_diff_report_card


@report_card("config_diff")
def config_diff_card(ctx: CardContext) -> list[Section]:
    from scripts.comparison_report_card import config_sections_for
    return config_sections_for(ctx.config_name, ctx.v2_dir, ctx.ve_dir)
```

- [ ] **Step 6: Run the card tests**

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py tests/test_card_verdicts.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/_compare/report_cards/statistical.py scripts/_compare/report_cards/config.py scripts/_compare/report_cards/config_diff.py tests/test_report_card_steps.py
git commit -m "feat(report-cards): statistical + config + config_diff as_step Steps (dual)"
```

---

### Task 4: Cutover — register in `build_core`; `assemble_from_studies` + `viz_cards` use the Steps

**Files:**
- Modify: `v2ecoli/core.py` (build_core), `scripts/comparison_report_card.py` (`assemble_from_studies`), `scripts/_compare/viz_cards.py`
- Test: `tests/test_assemble_studies.py`, `tests/test_viz_cards.py` (adapt)

**Interfaces:**
- Consumes: `REPORT_CARD_STEPS`; `core.access`.
- Produces: `assemble_from_studies` resolves cards via the core and calls `update(state)`; `viz_cards.write_report_cards` takes `html` per card.

- [ ] **Step 1: Register the card Steps in `build_core`** — in `v2ecoli/core.py::build_core`, after the emitter `register_link` block, add:

```python
    # Report-card Steps (resolved by name in the comparison harness).
    try:
        from scripts._compare.report_cards import REPORT_CARD_STEPS
        core.register_links(REPORT_CARD_STEPS)
    except Exception:  # noqa: BLE001 — never let card registration break build_core
        pass
```

- [ ] **Step 2: Update `viz_cards.write_report_cards`** to take pre-rendered `html`

Change `_card_html` usage: the card item now carries `html` (not `sections`). Replace the body's html line:

```python
        hp.write_text(card.get("html") or "", encoding="utf-8")
```
and delete the now-unused `_card_html`/`_DOC` helpers and the `html as _html` import if unused. Update the existing `tests/test_viz_cards.py` to pass `{"name", "html", "verdict", "axes"}` (replace `sections` with `html`) and assert the `html` is written verbatim.

- [ ] **Step 3: Update the failing test** `tests/test_assemble_studies.py::test_assemble_from_studies_writes_viz_cards` — replace the `rc.render` monkeypatch with a core-resolved stub. The new assemble resolves `core.access`; monkeypatch `build_core` to a fake core whose `access(name)` returns a stub Step class with `update` returning `{"card_html": "<b>card</b>", "verdict": "drift", "axes": [{"id": "x", "verdict": "drift"}]}`:

```python
def test_assemble_from_studies_writes_viz_cards(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})

    class _Stub:
        def __init__(self, *a, **k): pass
        def update(self, state):
            return {"card_html": "<b>card</b>", "verdict": "drift",
                    "axes": [{"id": "x", "verdict": "drift"}]}

    class _Core:
        def access(self, name): return _Stub
        def register_links(self, d): pass
    monkeypatch.setattr(crc, "build_core", lambda: _Core())
    spec = _spec("basal", "basal", ["standard"])
    crc.assemble_from_studies([spec], {"basal": ({}, {}, [])},
                              {"basal": ("v2", "ve")}, verdict_root=str(tmp_path / "vr"),
                              studies_root=str(tmp_path / "ws/investigations"))
    card = (tmp_path / "ws/investigations/v2ecoli-vecoli-comparison/studies/basal"
            / "viz/report_card/standard.html")
    assert card.is_file() and "<b>card</b>" in card.read_text(encoding="utf-8")
    import json
    assert json.loads(card.with_name("standard.verdict.json").read_text())["overall"] == "drift"
```

- [ ] **Step 4: Run it — expect FAIL** (assemble still calls `rc.render`)

Run: `.venv/bin/python -m pytest tests/test_assemble_studies.py::test_assemble_from_studies_writes_viz_cards -q`

- [ ] **Step 5: Switch `assemble_from_studies` to core invocation**

In `scripts/comparison_report_card.py`: add `from v2ecoli.core import build_core` at the top of the function (lazy, mirrors existing lazy imports). Replace the per-card loop body so it resolves + updates Steps and collects `html`:

```python
def assemble_from_studies(specs, cond_data, conds, verdict_root=None,
                          studies_root="workspace/investigations"):
    from scripts._compare.verdict import write_condition_verdict
    from scripts._compare.viz_cards import write_report_cards
    from v2ecoli.core import build_core
    core = build_core()
    if verdict_root is None and specs:
        verdict_root = f"docs/report_cards/{specs[0].invest_name}"
    overview = overview_section(cond_data); overview["nav_group"] = "Overall"
    sections = [overview]
    for spec in specs:
        name = spec.name
        if name not in cond_data:
            print(f"[assemble] skip study {name!r}: no store under --out", flush=True)
            continue
        per_obs, plot_trajs, v2_bounds = cond_data[name]
        v2_dir, ve_dir = conds.get(name, ("", ""))
        state = {"name": name, "condition": spec.condition, "seeds": spec.seeds,
                 "generations": spec.gens, "variant": 0, "observables": per_obs,
                 "plot_trajs": plot_trajs, "v2_bounds": v2_bounds,
                 "config": {"condition": spec.condition, "seeds": spec.seeds,
                            "generations": spec.gens, "cards": spec.cards},
                 "v2_dir": v2_dir, "ve_dir": ve_dir}
        card_verdicts, viz = {}, []
        for card in spec.cards:
            step = core.access(f"{card}_report_card")(config={}, core=core)
            out = step.update(state)
            sections.append({"title": f"{name} — {card}", "kind": "content",
                             "html": out["card_html"], "nav_group": name})
            card_verdicts[card] = {"verdict": out.get("verdict", "ungraded"),
                                   "axes": out.get("axes") or []}
            viz.append({"name": card, "html": out["card_html"],
                        "verdict": card_verdicts[card]["verdict"],
                        "axes": card_verdicts[card]["axes"]})
        if verdict_root:
            write_condition_verdict(verdict_root, name, card_verdicts)
        if studies_root:
            write_report_cards(Path(studies_root) / spec.invest_name / "studies" / name, viz)
    return sections
```

- [ ] **Step 6: Run the affected tests**

Run: `.venv/bin/python -m pytest tests/test_assemble_studies.py tests/test_viz_cards.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/core.py scripts/comparison_report_card.py scripts/_compare/viz_cards.py tests/test_assemble_studies.py tests/test_viz_cards.py
git commit -m "feat(report-cards): cutover assemble + build_core to as_step card Steps"
```

---

### Task 5: Remove the old registry + final verification

**Files:**
- Modify: `scripts/_compare/report_cards/__init__.py` + the 5 card modules (drop the transitional wrappers)
- Test: `tests/test_card_verdicts.py` (port to the Step interface), full suite

**Interfaces:**
- Consumes: the as_step Steps (Tasks 2–3) + the cutover (Task 4).

- [ ] **Step 1: Port `tests/test_card_verdicts.py` to the Step interface** — replace the `from ... import CardContext, render` + `render(name, ctx)` usages with the `_run_card(name, state)` helper pattern from `tests/test_report_card_steps.py` (move `_run_card`/`_state` into a shared `tests/_card_helpers.py` or duplicate). Assert the same verdict vocabularies/values. Delete assertions that referenced `CardContext`/`render` directly.

- [ ] **Step 2: Remove the old registry from `__init__.py`** — delete `CardContext`, `Card`, `REGISTRY`, `report_card`, `get`, `all_names`, `render`, and the bottom card-module imports' reliance on them. Keep `Section`, `CARD_INPUTS`, `CARD_OUTPUTS`, `REPORT_CARD_STEPS`, `_sections_to_html`, `_row_table`, and the card-module imports (so importing the package populates `REPORT_CARD_STEPS`):

```python
from scripts._compare.report_cards import standard, statistical  # noqa: E402,F401
from scripts._compare.report_cards import parca, config_diff, config  # noqa: E402,F401
```

- [ ] **Step 3: Drop the transitional `@report_card` wrappers** from all 5 card modules (the second function in each), and their `report_card, CardContext, Section` imports. Each module keeps only its `_*` helper + `update_*` + the `REPORT_CARD_STEPS[...] =` line.

- [ ] **Step 4: Run the FULL comparison suite**

Run: `.venv/bin/python -m pytest tests/test_report_card_steps.py tests/test_card_verdicts.py tests/test_assemble_studies.py tests/test_viz_cards.py tests/test_comparison_verdict.py tests/test_study_spec.py tests/test_materialize.py tests/test_compare_cli.py tests/test_modular_tests_integration.py -q`
Expected: all PASS, output pristine. Grep that nothing still imports the removed names:
`grep -rn "@report_card\|CardContext\|report_cards.render\|REGISTRY" scripts/ tests/ | grep -v REPORT_CARD_STEPS` → expect no live references.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report_cards tests/
git commit -m "refactor(report-cards): remove the bespoke @report_card registry (as_step only)"
```

## Self-Review

- **Spec coverage:** as_step Steps + typed schemas (Task 1); 5 cards converted (Tasks 2–3); `core.register_links` in build_core + harness `core.access`/`update` + viz html (Task 4); old registry removed (Task 5). Pure Steps + harness-writes preserved; dashboard/contract untouched.
- **Placeholder scan:** none — complete code per step; the one conditional (Task 1 Step 1 pinning the type strings) is an explicit validate-and-substitute, with the fallback named.
- **Type consistency:** `state` keys (`name`/`condition`/`seeds`/`generations`/`variant`/`observables`/`plot_trajs`/`v2_bounds`/`config`/`v2_dir`/`ve_dir`) identical across the schema (Task 1), every card's `update_*` (Tasks 2–3), and the assemble state dict (Task 4). Step output `{card_html, verdict, axes}` identical across cards, viz_cards, and assemble. Names `<card>_report_card` consistent in registration + `core.access`.
