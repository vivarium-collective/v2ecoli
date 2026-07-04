# Modular Tests — Plan B: v2ecoli producer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The comparison harness emits each study's report cards to `studies/<study>/viz/report_card/<card>.{html,verdict.json}` and declares them as `report_card` test modules, so the dashboard (Plan A) renders the comparison investigation with report cards as its Tests.

**Architecture:** Extend the existing study-YAML-only renderer/materializer: `assemble_from_studies` writes a standalone HTML + verdict sidecar per study per card; `materialize_study` declares `tests:` as `report_card` modules (one per card) while keeping the real `baseline:` and the canonical run. A `scaffold` verb stands up the investigation + per-config studies.

**Tech Stack:** Python 3 (stdlib + pyyaml), pytest, the existing `scripts/_compare/` package + `report_cards/` registry.

**Repo:** `/Users/eranagmon/code/v2e-main` (run tests via `.venv/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-06-28-modular-tests-report-card-modules-design.md`. **Depends on the Plan A contract:** `viz/report_card/<card>.html` + `<card>.verdict.json` (`overall`), and the `tests: [{kind: report_card, card: <name>}]` schema.

## Global Constraints

- Each study's cards are written to `studies/<study>/viz/report_card/<card>.html` + `<card>.verdict.json` (`{schema, overall, groups}`, `overall` ∈ within_tol/drift/mismatch/ungraded).
- A study's `tests:` lists one `report_card` module per assigned card: `{name: "<card>-vs-vecoli", kind: report_card, card: <card>, classification: primary}`. The graded cards (standard/statistical) keep a `measure` so the gate still aggregates; config/parca are informational (no measure).
- Studies keep a real `baseline: [{name: v2ecoli-baseline, composite: v2ecoli.composites.baseline.baseline, params: {condition: <cond>}}]` and the canonical run with `outcomes` (the pre-rendered gate).
- `report_card_axis` gate verdict mapping unchanged (within_tol→PASS, drift→PARTIAL, mismatch→FAIL).
- All reads/writes `encoding="utf-8"`. Investigation name `v2ecoli-vecoli-comparison`. Never auto-merge.

---

### Task B1: Per-card standalone HTML + verdict writer (`scripts/_compare/viz_cards.py`)

**Files:**
- Create: `scripts/_compare/viz_cards.py`
- Test: `tests/test_viz_cards.py`

**Interfaces:**
- Produces: `write_report_cards(study_dir, cards) -> list[Path]` where `cards` is a list of `{"name": str, "sections": list[Section], "verdict": str, "axes": list}`. Writes `<study_dir>/viz/report_card/<name>.html` (standalone) + `<name>.verdict.json` (`{schema:"report_card_verdict/v1", overall:<verdict>, groups:{<name>:{verdict, axes}}}`). `Section` is the card section dict `{title?, html, ...}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viz_cards.py
import json
from scripts._compare.viz_cards import write_report_cards


def test_writes_html_and_verdict_per_card(tmp_path):
    cards = [
        {"name": "standard", "verdict": "drift",
         "axes": [{"id": "standard.rna", "verdict": "drift"}],
         "sections": [{"title": "basal — evaluation", "html": "<table>rows</table>"}]},
        {"name": "config", "verdict": "ungraded", "axes": [],
         "sections": [{"title": "basal — config", "html": "<pre>cfg</pre>"}]},
    ]
    paths = write_report_cards(tmp_path, cards)
    rc = tmp_path / "viz" / "report_card"
    assert (rc / "standard.html").is_file() and (rc / "standard.verdict.json").is_file()
    assert (rc / "config.html").is_file()
    html = (rc / "standard.html").read_text(encoding="utf-8")
    assert "<table>rows</table>" in html and "basal — evaluation" in html
    assert html.lstrip().startswith("<!DOCTYPE html>")
    vd = json.loads((rc / "standard.verdict.json").read_text(encoding="utf-8"))
    assert vd["overall"] == "drift"
    assert vd["groups"]["standard"]["verdict"] == "drift"
    assert {p.name for p in paths} >= {"standard.html", "config.html"}
```

- [ ] **Step 2: Run it — expect FAIL** (`ModuleNotFoundError`)

Run: `.venv/bin/python -m pytest tests/test_viz_cards.py -q`

- [ ] **Step 3: Write `scripts/_compare/viz_cards.py`**

```python
"""Write each study's report cards as standalone HTML + a verdict sidecar under
studies/<study>/viz/report_card/, the convention the dashboard auto-discovers
(saved_visualizations) and embeds as a `report_card` test module."""
from __future__ import annotations

import json
import html as _html
from pathlib import Path

_DOC = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>{title}</title><style>body{{font-family:-apple-system,Segoe UI,"
        "Roboto,sans-serif;margin:14px;color:#0f172a}}h3{{margin:14px 0 6px}}"
        "table{{border-collapse:collapse}}td,th{{padding:4px 8px}}</style></head>"
        "<body>{body}</body></html>")


def _card_html(name: str, sections: list) -> str:
    parts = []
    for sec in sections:
        if sec.get("title"):
            parts.append(f"<h3>{_html.escape(str(sec['title']))}</h3>")
        parts.append(sec.get("html", ""))
    return _DOC.format(title=_html.escape(name), body="".join(parts))


def write_report_cards(study_dir, cards: list) -> list:
    """Write <study_dir>/viz/report_card/<name>.{html,verdict.json} per card."""
    out = Path(study_dir) / "viz" / "report_card"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for card in cards:
        name = card["name"]
        hp = out / f"{name}.html"
        hp.write_text(_card_html(name, card.get("sections") or []), encoding="utf-8")
        written.append(hp)
        verdict = card.get("verdict") or "ungraded"
        vp = out / f"{name}.verdict.json"
        vp.write_text(json.dumps({
            "schema": "report_card_verdict/v1",
            "overall": verdict,
            "groups": {name: {"verdict": verdict, "axes": card.get("axes") or []}},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(vp)
    return written
```

- [ ] **Step 4: Run it — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_viz_cards.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/viz_cards.py tests/test_viz_cards.py
git commit -m "feat(compare): per-card standalone HTML + verdict writer (viz/report_card)"
```

---

### Task B2: `assemble_from_studies` writes per-study cards to `viz/report_card/`

**Files:**
- Modify: `scripts/comparison_report_card.py` (`assemble_from_studies`, the per-study card loop)
- Test: `tests/test_assemble_studies.py` (extend)

**Interfaces:**
- Consumes: `write_report_cards` (B1); the per-card sections + verdict already collected in the loop.
- Produces: after rendering a study's cards, `assemble_from_studies` writes them to `workspace/investigations/<invest>/studies/<name>/viz/report_card/`. New keyword `studies_root="workspace/investigations"`.

- [ ] **Step 1: Write the failing test** (extend the existing assemble test)

```python
# add to tests/test_assemble_studies.py
def test_assemble_from_studies_writes_viz_cards(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    monkeypatch.setattr(rc, "render", lambda name, ctx: [
        {"title": f"{ctx.config_name}-{name}", "kind": "content", "html": "<b>card</b>",
         "verdict": "drift", "verdict_axes": [{"id": "x", "verdict": "drift"}]}])
    spec = _spec("basal", "basal", ["standard"])
    crc.assemble_from_studies([spec], {"basal": ({}, {}, [])},
                              {"basal": ("v2", "ve")}, verdict_root=str(tmp_path / "vr"),
                              studies_root=str(tmp_path / "ws/investigations"))
    card = (tmp_path / "ws/investigations/v2ecoli-vecoli-comparison/studies/basal"
            / "viz/report_card/standard.html")
    assert card.is_file() and "<b>card</b>" in card.read_text(encoding="utf-8")
    import json
    vd = json.loads(card.with_name("standard.verdict.json").read_text(encoding="utf-8"))
    assert vd["overall"] == "drift"
```

(`_spec` already exists in this test file from the study-YAML refactor; it builds a `StudySpec` with `invest_name="v2ecoli-vecoli-comparison"`.)

- [ ] **Step 2: Run it — expect FAIL** (`assemble_from_studies` takes no `studies_root`)

Run: `.venv/bin/python -m pytest tests/test_assemble_studies.py::test_assemble_from_studies_writes_viz_cards -q`

- [ ] **Step 3: Edit `assemble_from_studies`** — collect per-card sections + write them

Add the param and, inside the per-study loop, accumulate each card's sections + write. Change the signature line and the card loop:

```python
def assemble_from_studies(specs, cond_data, conds, verdict_root=None,
                          studies_root="workspace/investigations"):
    from scripts._compare import report_cards as rc
    from scripts._compare.verdict import write_condition_verdict
    from scripts._compare.viz_cards import write_report_cards

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
        ctx = rc.CardContext(
            config_name=name, variant=0, v2_dir=v2_dir, ve_dir=ve_dir,
            seeds=spec.seeds, gens=spec.gens, per_obs=per_obs,
            plot_trajs=plot_trajs, v2_bounds=v2_bounds,
            config={"condition": spec.condition, "seeds": spec.seeds,
                    "generations": spec.gens, "cards": spec.cards})
        card_verdicts = {}
        viz_cards = []
        for card in spec.cards:
            cardv, secs = None, []
            for sec in rc.render(card, ctx):
                sec["nav_group"] = name
                sections.append(sec)
                secs.append(sec)
                if "verdict_axes" in sec:
                    cardv = {"verdict": sec.get("verdict", "ungraded"),
                             "axes": sec["verdict_axes"]}
            card_verdicts[card] = cardv or {"verdict": "ungraded", "axes": []}
            viz_cards.append({"name": card, "sections": secs,
                              "verdict": card_verdicts[card]["verdict"],
                              "axes": card_verdicts[card]["axes"]})
        if verdict_root:
            write_condition_verdict(verdict_root, name, card_verdicts)
        if studies_root:
            study_dir = Path(studies_root) / spec.invest_name / "studies" / name
            write_report_cards(study_dir, viz_cards)
    return sections
```

(`Path` is imported at the top of comparison_report_card.py.)

- [ ] **Step 4: Run the new + existing assemble tests**

Run: `.venv/bin/python -m pytest tests/test_assemble_studies.py -q`
Expected: PASS (new test + the two existing ones).

- [ ] **Step 5: Commit**

```bash
git add scripts/comparison_report_card.py tests/test_assemble_studies.py
git commit -m "feat(compare): assemble_from_studies writes per-study viz/report_card cards"
```

---

### Task B3: `materialize_study` declares `report_card` test modules

**Files:**
- Modify: `scripts/_compare/materialize.py`
- Test: `tests/test_materialize.py` (extend)

**Interfaces:**
- Consumes: `StudySpec` (name, condition, cards, graded_cards).
- Produces: `materialize_study` writes `tests:` (the modular list) with one `report_card` module per card; graded cards (standard/statistical) carry a `measure` (report_card_axis on the viz sidecar dir) so the gate aggregates; config/parca are informational. Keeps `baseline:`, `pipeline_gate`, canonical run + outcomes. The legacy `behavior_tests:` is removed in favour of `tests:`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_materialize.py
def test_materialize_declares_report_card_test_modules(tmp_path):
    sp = tmp_path / "study.yaml"
    sp.write_text(
        "name: basal\ninvestigation: v2ecoli-vecoli-comparison\ncondition: basal\n"
        "comparison: {seeds: 1, generations: 4, cards: [config, parca, standard]}\n",
        encoding="utf-8")
    spec = _spec(sp, name="basal", cards=["config", "parca", "standard"])
    import scripts._compare.materialize as M
    M.materialize_study(spec)
    import yaml
    data = yaml.safe_load(sp.read_text(encoding="utf-8"))
    tests = {t["name"]: t for t in data["tests"]}
    # one report_card module per card; all kind: report_card
    assert {t["kind"] for t in data["tests"]} == {"report_card"}
    assert tests["config-vs-vecoli"]["card"] == "config"
    assert "measure" not in tests["config-vs-vecoli"]            # informational
    assert tests["standard-vs-vecoli"]["measure"]["kind"] == "report_card_axis"
    assert "behavior_tests" not in data                          # replaced by tests
    assert data["baseline"][0]["name"] == "v2ecoli-baseline"
```

(`_spec` in this file builds a `StudySpec`; pass `study_path=sp` so materialize writes to the tmp file. If the existing `_spec` hardcodes a path, add a `study_path` kwarg or construct `StudySpec(...)` inline.)

- [ ] **Step 2: Run it — expect FAIL** (still emits `behavior_tests`)

Run: `.venv/bin/python -m pytest tests/test_materialize.py::test_materialize_declares_report_card_test_modules -q`

- [ ] **Step 3: Rewrite `materialized_fields` to emit `tests` (report_card modules)**

In `scripts/_compare/materialize.py`, replace `materialized_fields` so it returns `tests` (not `behavior_tests`) + `report_cards`:

```python
def materialized_fields(spec: StudySpec) -> dict:
    """report_cards (viz embeds) + a modular `tests` list of report_card modules
    (one per assigned card). Graded cards carry a report_card_axis measure so the
    gate aggregates; config/parca are informational (no measure)."""
    cdir = f"{card_root(spec)}/{spec.name}"          # docs/report_cards/<invest>/<name>
    tests = []
    for c in spec.cards:
        t = {"name": f"{c}-vs-vecoli", "kind": "report_card", "card": c,
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {spec.name} ({c} card)?"}
        if c in spec.graded_cards:
            t["measure"] = {"kind": "report_card_axis", "card": cdir, "group": c}
        tests.append(t)
    return {
        "report_cards": [f"viz/report_card/{c}.html" for c in spec.cards],
        "tests": tests,
    }
```

Then in `materialize_study`, after `data.update(materialized_fields(spec))`, drop any stale legacy key:

```python
    data.update(materialized_fields(spec))
    data.pop("behavior_tests", None)   # replaced by the modular `tests` list
```

(Keep the existing `baseline`, `pipeline_gate`, canonical-run + outcomes blocks unchanged — the outcomes still map the graded card's verdict for the gate.)

- [ ] **Step 4: Run materialize tests**

Run: `.venv/bin/python -m pytest tests/test_materialize.py -q`
Expected: PASS (update any existing assertion that referenced `behavior_tests` to use `tests`).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/materialize.py tests/test_materialize.py
git commit -m "feat(compare): materialize declares modular report_card test modules"
```

---

### Task B4: `v2e-compare scaffold` verb

**Files:**
- Modify: `scripts/compare_cli.py` (add the `scaffold` subcommand)
- Test: `tests/test_compare_cli.py` (extend)

**Interfaces:**
- Consumes: `study_spec.load_investigation`, `materialize.materialize_study`.
- Produces: `v2e-compare scaffold <investigation>` materializes every study in the investigation (declares its report_card modules, baseline, pipeline_gate) WITHOUT running sims — so the structure exists before a run produces the card artifacts. Returns 0.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_compare_cli.py
def test_scaffold_materializes_all_studies(monkeypatch):
    import scripts.compare_cli as cli
    seen = []
    fake_specs = [type("S", (), {"name": "basal"})(), type("S", (), {"name": "with_aa"})()]
    monkeypatch.setattr(cli.runner, "load_investigation",
                        lambda ref: ({}, fake_specs))
    monkeypatch.setattr(cli, "_materialize", lambda spec: seen.append(spec.name))
    rc = cli.main(["scaffold", "v2ecoli-vecoli-comparison"])
    assert rc == 0 and seen == ["basal", "with_aa"]
```

- [ ] **Step 2: Run it — expect FAIL** (no `scaffold` subcommand)

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py::test_scaffold_materializes_all_studies -q`

- [ ] **Step 3: Add the `scaffold` verb to `compare_cli.py`**

Add an indirection + the subparser. Near the top imports add:

```python
from scripts._compare.materialize import materialize_study as _materialize  # noqa: E402
```

In `main`, add the parser + dispatch:

```python
    psc = sub.add_parser("scaffold", help="materialize an investigation's studies "
                                          "(declare report_card modules; no sims)")
    psc.add_argument("investigation", nargs="?", default=DEFAULT_INVEST)
    ...
    if args.cmd == "scaffold":
        _ctx, specs = runner.load_investigation(args.investigation)
        for spec in specs:
            _materialize(spec)
        print(f"scaffolded {len(specs)} studies in {args.investigation}")
        return 0
```

(Ensure `runner.load_investigation` is importable on `cli.runner` — it already re-exports `load_investigation`.)

- [ ] **Step 4: Run CLI tests**

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_cli.py tests/test_compare_cli.py
git commit -m "feat(compare): v2e-compare scaffold (materialize studies, no sims)"
```

---

### Task B5: Regenerate the comparison studies + integration check

**Files:**
- Modify (generated): `workspace/investigations/v2ecoli-vecoli-comparison/studies/*/study.yaml`
- Test: `tests/test_modular_tests_integration.py` (create)

**Interfaces:**
- Consumes: B1–B4.

- [ ] **Step 1: Write the integration test** — render from synthetic stores → cards + modular tests line up

```python
# tests/test_modular_tests_integration.py
import json, yaml
from pathlib import Path
from scripts._compare.study_spec import load_investigation
from scripts._compare.materialize import materialize_study


def test_studies_declare_modules_matching_their_cards():
    _ctx, specs = load_investigation("v2ecoli-vecoli-comparison")
    for s in specs:
        materialize_study(s)
        data = yaml.safe_load(Path(s.study_path).read_text(encoding="utf-8"))
        test_cards = sorted(t["card"] for t in data["tests"] if t.get("kind") == "report_card")
        assert test_cards == sorted(s.cards)          # one module per card
        assert all(rc.startswith("viz/report_card/") for rc in data["report_cards"])
```

- [ ] **Step 2: Run it — expect PASS** (after B3 materialize)

Run: `.venv/bin/python -m pytest tests/test_modular_tests_integration.py -q`
Expected: PASS.

- [ ] **Step 3: Re-scaffold the real studies**

Run: `.venv/bin/python scripts/compare_cli.py scaffold v2ecoli-vecoli-comparison`
Expected: `scaffolded 6 studies …`. Confirm one study:
`.venv/bin/python -c "import yaml;d=yaml.safe_load(open('workspace/investigations/v2ecoli-vecoli-comparison/studies/basal/study.yaml'));print([ (t['name'],t['kind'],t['card']) for t in d['tests']])"`
Expected: three `report_card` modules (config/parca/standard).

- [ ] **Step 4: Commit the regenerated studies**

```bash
git add -f workspace/investigations/v2ecoli-vecoli-comparison
git commit -m "feat(compare): comparison studies declare modular report_card tests"
```

- [ ] **Step 5: Full comparison suite**

Run: `.venv/bin/python -m pytest tests/test_viz_cards.py tests/test_assemble_studies.py tests/test_materialize.py tests/test_compare_cli.py tests/test_modular_tests_integration.py tests/test_study_spec.py tests/test_runner.py -q`
Expected: all PASS.

## Human verification (post-implementation)

1. On a machine with the zarr stores (the mini, or after a local run), `v2e-compare run --render-only --out <stores>` → confirm `studies/<study>/viz/report_card/<card>.html` + `.verdict.json` appear.
2. Serve the dashboard against `v2e-main` (Plan A live), open a comparison study → its Tests section shows the embedded report cards with verdict pills (config/parca/standard or statistical), not generic behavioral pills.

## Self-Review

- **Spec coverage:** card emission to `viz/report_card/` (B1+B2); `tests:` as report_card modules (B3); scaffold (B4); regenerated studies + integration (B5). Baseline kept (B3). The dashboard render is Plan A.
- **Placeholder scan:** none — complete code per step; the `_spec` helper reuse names the exact existing fixture.
- **Type consistency:** `cards` item shape `{name, sections, verdict, axes}` identical in B1/B2; verdict.json `{schema, overall, groups}` identical in B1/B2 and matches Plan A's `overall` read; `tests[]` entry `{name, kind: report_card, card, measure?}` identical in B3/B5 and matches Plan A's schema.
