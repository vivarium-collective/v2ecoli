# Comparison ↔ Investigation Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v2ecoli↔vEcoli comparison harness a first-class dashboard investigation (each condition a study, each report card a gating test) and triggerable end-to-end from one CLI (`v2e-compare run` / `v2e-compare study`).

**Architecture:** Reuse the existing `report_card_axis` evaluator verbatim. Emit **one `report_card_verdict.json` per condition** where **each report card is a group** (graded card's internal axes = the group's axes; worst-of = card overall). Hand-authored studies reference the manifest; a validator (not a generator) guards drift; a one-time scaffold bootstraps them; a thin CLI sequences scaffold→run→verdict→validate.

**Tech Stack:** Python 3 (stdlib + `pyyaml`), pytest, the existing `scripts/_compare/` package and `v2ecoli/library/report_card.py`.

**Spec:** `docs/superpowers/specs/2026-06-27-comparison-investigation-unification-design.md`

## Global Constraints

- Investigation name is exactly `v2ecoli-vecoli-comparison`; verdict card root is exactly `docs/report_cards/v2ecoli-vecoli-comparison`.
- Verdict JSON uses schema string `report_card_verdict/v1` with top-level keys `schema, model_ref, reference_model, generated, overall, groups`; each group is `{verdict, axes}`; each axis is `{id, label, verdict, value, meter, detail}`.
- Severity order (worst-first): `mismatch` > `drift` > `within_tol` > `ungraded`. Matches `pbg_v2ecoli/evaluators.py::_SEVERITY`.
- Graded cards (produce a gating test) are exactly `standard` and `statistical`. `config`/`parca` render but emit an `ungraded` group and **no** gating test.
- All file reads AND writes use `encoding="utf-8"` (CI runs an ASCII locale; titles contain `↔`/`×`).
- One gating test per graded report card (the card's overall verdict). Independent studies (`pipeline_gate.prerequisites: []`). Studies reference the manifest (authored, persistent).
- Sims run serial+local by default; `--ray` (or `V2E_MODE=ray`) selects `--mode ray`. Scaffold is idempotent (never overwrites without `--force`).
- Run everything via `/Users/eranagmon/code/v2e-main/.venv/bin/python` (bare `python` lacks `unum`). Never auto-merge.

---

### Task 1: Per-condition verdict builder/writer (`scripts/_compare/verdict.py`)

**Files:**
- Create: `scripts/_compare/verdict.py`
- Test: `tests/test_comparison_verdict.py`

**Interfaces:**
- Produces: `worst(verdicts) -> str`; `build_condition_verdict(condition: str, card_verdicts: dict[str, dict]) -> dict`; `write_condition_verdict(card_root, condition: str, card_verdicts: dict) -> Path`. `card_verdicts` maps `card_name -> {"verdict": str, "axes": list[dict]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_comparison_verdict.py
import json
from scripts._compare.verdict import (
    worst, build_condition_verdict, write_condition_verdict)


def test_worst_orders_by_severity():
    assert worst(["within_tol", "mismatch", "drift"]) == "mismatch"
    assert worst(["within_tol", "drift"]) == "drift"
    assert worst([]) == "ungraded"
    assert worst(["bogus"]) == "ungraded"


def test_build_groups_per_card_and_overall_is_worst():
    cards = {
        "standard": {"verdict": "drift", "axes": [
            {"id": "standard.rna_mass", "label": "RNA mass", "verdict": "drift"}]},
        "config": {"verdict": "ungraded", "axes": []},
    }
    doc = build_condition_verdict("basal", cards)
    assert doc["schema"] == "report_card_verdict/v1"
    assert set(doc["groups"]) == {"standard", "config"}
    assert doc["groups"]["config"]["verdict"] == "ungraded"
    assert doc["overall"] == "drift"
    assert doc["model_ref"] == "v2ecoli @ basal"
    assert doc["reference_model"] == "vEcoli @ basal"


def test_group_verdict_falls_back_to_worst_axis_when_absent():
    cards = {"standard": {"axes": [
        {"id": "a", "verdict": "within_tol"}, {"id": "b", "verdict": "mismatch"}]}}
    doc = build_condition_verdict("with_aa", cards)
    assert doc["groups"]["standard"]["verdict"] == "mismatch"


def test_write_creates_per_condition_file(tmp_path):
    p = write_condition_verdict(tmp_path, "basal", {
        "standard": {"verdict": "within_tol",
                     "axes": [{"id": "x", "verdict": "within_tol"}]}})
    assert p == tmp_path / "basal" / "report_card_verdict.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["overall"] == "within_tol"


def test_verdict_feeds_report_card_axis_evaluator(tmp_path):
    # The core proof that gating needs no new code: the evaluator reads our file.
    from pbg_v2ecoli.evaluators import evaluate_report_card_group
    write_condition_verdict(tmp_path, "basal", {
        "standard": {"verdict": "mismatch",
                     "axes": [{"id": "standard.rna", "verdict": "mismatch"}]}})
    test = {"measure": {"kind": "report_card_axis",
                        "card": "basal", "group": "standard"}}
    res = evaluate_report_card_group(test, None, str(tmp_path))
    assert res["result"] == "FAIL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_comparison_verdict.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.verdict`.

- [ ] **Step 3: Write `scripts/_compare/verdict.py`**

```python
"""Per-condition report_card_verdict.json emission for the comparison harness.

Maps the unification model onto the existing report_card_axis evaluator:
ONE verdict JSON per CONDITION; each report CARD is a GROUP; a graded card's
internal axes are the group's axes (worst-of = the card's overall verdict).
"""
from __future__ import annotations

import json
from pathlib import Path

# Severity order matches pbg_v2ecoli/evaluators.py::_SEVERITY and the evaluator's
# worst-of-axes aggregation.
_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def worst(verdicts) -> str:
    """The most severe verdict in an iterable; 'ungraded' if empty/all unknown."""
    vs = [v for v in verdicts if v in _SEVERITY]
    return max(vs, key=lambda v: _SEVERITY[v]) if vs else "ungraded"


def build_condition_verdict(condition: str, card_verdicts: dict) -> dict:
    """Assemble the report_card_verdict/v1 doc for one condition.

    card_verdicts maps card_name -> {"verdict": str, "axes": list[dict]}. Cards
    with no axes (config/parca) become an 'ungraded' group. Top-level 'overall'
    is the worst across all groups.
    """
    groups: dict[str, dict] = {}
    for card_name, cv in card_verdicts.items():
        axes = (cv or {}).get("axes") or []
        gverdict = (cv or {}).get("verdict") or worst(
            a.get("verdict", "ungraded") for a in axes)
        groups[card_name] = {"verdict": gverdict, "axes": axes}
    overall = worst(g["verdict"] for g in groups.values())
    return {
        "schema": "report_card_verdict/v1",
        "model_ref": f"v2ecoli @ {condition}",
        "reference_model": f"vEcoli @ {condition}",
        "generated": "",
        "overall": overall,
        "groups": groups,
    }


def write_condition_verdict(card_root, condition: str, card_verdicts: dict) -> Path:
    """Write <card_root>/<condition>/report_card_verdict.json; return its path."""
    out = Path(card_root) / condition
    out.mkdir(parents=True, exist_ok=True)
    doc = build_condition_verdict(condition, card_verdicts)
    path = out / "report_card_verdict.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_comparison_verdict.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/verdict.py tests/test_comparison_verdict.py
git commit -m "feat(compare): per-condition report_card_verdict.json builder/writer"
```

---

### Task 2: Expose graded-card verdicts (`standard.py`, `statistical.py`)

**Files:**
- Modify: `scripts/_compare/report_cards/standard.py`
- Modify: `scripts/_compare/report_cards/statistical.py`
- Test: `tests/test_card_verdicts.py`

**Interfaces:**
- Consumes: `worst` from `scripts._compare.verdict` (Task 1).
- Produces: each graded card returns (among its sections) exactly one Section carrying `"verdict"` (str) and `"verdict_axes"` (list of axis dicts `{id,label,verdict,value,meter,detail}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_card_verdicts.py
from scripts._compare.report_cards import CardContext, render

# One within-tol observable; the rest are absent -> 'not_compared' -> 'ungraded'.
_PER_OBS = {"rna_mass": [
    {"median_rel": 0.02, "max_rel": 0.05, "init_ve": 100.0, "init_v2": 101.0,
     "init_t": 60.0}]}


def _ctx():
    return CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                       seeds=1, gens=1, per_obs=_PER_OBS)


def _verdict_section(sections):
    hits = [s for s in sections if "verdict_axes" in s]
    assert len(hits) == 1, "exactly one section must carry the verdict"
    return hits[0]


def test_standard_card_emits_verdict_and_axes():
    sec = _verdict_section(render("standard", _ctx()))
    assert sec["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    ids = {a["id"] for a in sec["verdict_axes"]}
    assert any(i.startswith("standard.") for i in ids)
    # not_compared rows map to 'ungraded', never the literal 'not_compared'.
    assert all(a["verdict"] != "not_compared" for a in sec["verdict_axes"])
    # the one matched observable is within_tol -> card overall within_tol.
    assert sec["verdict"] == "within_tol"


def test_statistical_card_emits_verdict_and_axes():
    sec = render("statistical", _ctx())[0]
    assert "verdict_axes" in sec and isinstance(sec["verdict_axes"], list)
    assert sec["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_card_verdicts.py -q`
Expected: FAIL — `assert len(hits) == 1` (standard emits no verdict section yet) / `KeyError: 'verdict_axes'` for statistical.

- [ ] **Step 3: Rewrite `scripts/_compare/report_cards/standard.py`**

```python
"""`standard` card — matched-time run trajectories + evaluation (the lighter
card). Thin wrapper over comparison_report_card.runs_section / eval_section.
The evaluation section also carries a card-level verdict + axes (one per
observable) so the comparison can gate on it via report_card_axis."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section
from scripts._compare.verdict import worst


@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    # Imported lazily: comparison_report_card imports heavy deps; importing it at
    # module load would slow registry import and risk a cycle.
    from scripts.comparison_report_card import runs_section, eval_section
    runs = runs_section(ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    ev = eval_section(ctx.config_name, ctx.per_obs)
    axes = []
    for row in ev.get("rows", []):
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({
            "id": f"standard.{row['label']}",
            "label": row["label"],
            "verdict": v,
            "value": row.get("median_rel"),
            "meter": row.get("reason", ""),
            "detail": {"median_rel": row.get("median_rel"),
                       "max_rel": row.get("max_rel")},
        })
    ev["verdict"] = worst(a["verdict"] for a in axes)
    ev["verdict_axes"] = axes
    return [runs, ev]
```

- [ ] **Step 4: Edit `scripts/_compare/report_cards/statistical.py`** — flatten the already-computed `vjson` groups into `verdict_axes`

Replace the `return` block so the Section also carries `verdict_axes`:

```python
    vjson, html = build_report_card(
        left, right, extra_axes=EXTRA_AXES,
        model_ref=f"v2ecoli @ {ctx.config_name} variant {ctx.variant}", tol_rel=TOL)
    axes = [ax for g in (vjson.get("groups") or {}).values()
            for ax in (g.get("axes") or [])]
    return {"title": f"{ctx.config_name} — statistical equivalence",
            "kind": "content", "anchor": f"{ctx.config_name}-statistical",
            "html": html, "verdict": vjson.get("overall"),
            "verdict_axes": axes}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_card_verdicts.py tests/test_report_cards.py -q`
Expected: PASS (new tests pass; existing `test_report_cards.py` still green).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/report_cards/standard.py scripts/_compare/report_cards/statistical.py tests/test_card_verdicts.py
git commit -m "feat(compare): graded cards expose verdict + per-axis verdict_axes"
```

---

### Task 3: Wire verdict emission into the render path (`assemble_from_manifest`)

**Files:**
- Modify: `scripts/comparison_report_card.py:691-734` (`assemble_from_manifest`)
- Test: `tests/test_assemble_verdict.py`

**Interfaces:**
- Consumes: `write_condition_verdict` (Task 1); the `verdict`/`verdict_axes` Section keys (Task 2).
- Produces: `assemble_from_manifest(manifest, cond_data, conds, config_names, verdict_root="docs/report_cards/v2ecoli-vecoli-comparison")` — now also writes one verdict JSON per rendered condition. `run_comparison.py` renders via this function, so both the render and run paths emit verdicts (spec Decision 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_assemble_verdict.py
import json


def test_assemble_writes_condition_verdict(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    # Stub the heavy overview + card rendering so we test only the wiring.
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    monkeypatch.setattr(rc, "render", lambda name, ctx: [
        {"title": name, "kind": "content", "html": "",
         "verdict": "drift", "verdict_axes": [{"id": "x", "verdict": "drift"}]}])
    manifest = {"defaults": {"cards": ["standard"]},
                "configs": [{"config": "configs/cond_basal_1x4.json",
                             "cards": ["standard"]}]}
    cond_data = {"basal": ({}, {}, [])}
    conds = {"basal": ("v2", "ve")}
    config_names = {"configs/cond_basal_1x4.json": "basal"}
    crc.assemble_from_manifest(manifest, cond_data, conds, config_names,
                               verdict_root=str(tmp_path))
    doc = json.loads((tmp_path / "basal" / "report_card_verdict.json").read_text(
        encoding="utf-8"))
    assert doc["groups"]["standard"]["verdict"] == "drift"
    assert doc["overall"] == "drift"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_assemble_verdict.py -q`
Expected: FAIL — `assemble_from_manifest()` takes 4 args / no verdict file written.

- [ ] **Step 3: Edit `assemble_from_manifest`** — add `verdict_root` param + per-condition collection/write

Change the signature and the config loop body:

```python
def assemble_from_manifest(manifest, cond_data, conds, config_names,
                           verdict_root="docs/report_cards/v2ecoli-vecoli-comparison"):
    """Overview + per-config assigned-card sections, mirroring the manifest.

    config_names maps a manifest config path -> the condition key used in
    cond_data/conds (the runner names stores by condition). When verdict_root is
    set, also writes one report_card_verdict.json per rendered condition (each
    report card becomes a group), so the comparison can gate via report_card_axis.
    """
    from scripts._compare import report_cards as rc
    from scripts._compare.verdict import write_condition_verdict
    default_cards = (manifest.get("defaults", {}) or {}).get("cards") or ["standard"]
    import json as _json
    import os as _os
    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def _load_cfg(p):
        for cand in (p, _os.path.join(_repo, p)):
            try:
                with open(cand, encoding="utf-8") as _fh:
                    d = _json.load(_fh)
                d["_source"] = cand
                return d
            except (OSError, ValueError):
                continue
        return {}

    overview = overview_section(cond_data); overview["nav_group"] = "Overall"
    sections = [overview]
    for entry in manifest.get("configs", []):
        name = config_names[entry["config"]]
        if name not in cond_data:
            continue
        per_obs, plot_trajs, v2_bounds = cond_data[name]
        v2_dir, ve_dir = conds.get(name, ("", ""))
        _cfg = _load_cfg(entry["config"])
        ctx = rc.CardContext(config_name=name, variant=0, v2_dir=v2_dir,
                             ve_dir=ve_dir, seeds=int(_cfg.get("n_init_sims") or 0),
                             gens=int(_cfg.get("generations") or 0), per_obs=per_obs,
                             plot_trajs=plot_trajs, v2_bounds=v2_bounds, config=_cfg)
        card_verdicts = {}
        for card in (entry.get("cards") or default_cards):
            cardv = None
            for sec in rc.render(card, ctx):
                sec["nav_group"] = name
                sections.append(sec)
                if "verdict_axes" in sec:
                    cardv = {"verdict": sec.get("verdict", "ungraded"),
                             "axes": sec["verdict_axes"]}
            card_verdicts[card] = cardv or {"verdict": "ungraded", "axes": []}
        if verdict_root:
            write_condition_verdict(verdict_root, name, card_verdicts)
    return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_assemble_verdict.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/comparison_report_card.py tests/test_assemble_verdict.py
git commit -m "feat(compare): assemble_from_manifest writes per-condition verdict JSON"
```

---

### Task 4: One-time scaffold (`scripts/scaffold_comparison_studies.py`)

**Files:**
- Create: `scripts/scaffold_comparison_studies.py`
- Test: `tests/test_scaffold_comparison_studies.py`

**Interfaces:**
- Produces: module constants `INVEST = "v2ecoli-vecoli-comparison"`, `CARD_ROOT = "docs/report_cards/v2ecoli-vecoli-comparison"`, `GRADED = {"standard", "statistical"}`; `condition_name(entry: dict) -> str`; `build_study(cond, cards, manifest_rel) -> dict`; `build_investigation(conds) -> dict`; `scaffold(manifest_path, ws_root, force=False) -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scaffold_comparison_studies.py
import json
import yaml
from scripts.scaffold_comparison_studies import (
    condition_name, build_study, scaffold, INVEST, CARD_ROOT)


def test_condition_name_uses_explicit_name_then_strips_stem():
    assert condition_name({"config": "configs/cond_basal_1x4.json"}) == "basal"
    assert condition_name({"config": "x/cond_with_aa.json"}) == "with_aa"
    assert condition_name({"config": "c/cond_basal_4x4.json",
                           "name": "basal_4x4"}) == "basal_4x4"


def test_build_study_one_test_per_graded_card():
    s = build_study("basal", ["config", "parca", "standard"], "comparison_spec.json")
    assert s["condition"] == "basal"
    assert s["comparison_manifest"] == "comparison_spec.json"
    assert s["pipeline_gate"] == {"prerequisites": [], "enables": []}
    groups = [t["measure"]["group"] for t in s["behavior_tests"]]
    assert groups == ["standard"]            # config/parca are not graded
    t = s["behavior_tests"][0]
    assert t["measure"]["kind"] == "report_card_axis"
    assert t["measure"]["card"] == f"{CARD_ROOT}/basal"


def _manifest(tmp_path):
    m = tmp_path / "spec.json"
    m.write_text(json.dumps({
        "defaults": {"cards": ["config", "parca", "standard"]},
        "configs": [
            {"config": "configs/cond_basal_1x4.json"},
            {"config": "configs/cond_basal_4x4.json", "name": "basal_4x4",
             "cards": ["config", "parca", "statistical"]}]}), encoding="utf-8")
    return m


def test_scaffold_writes_investigation_and_studies(tmp_path):
    written = scaffold(str(_manifest(tmp_path)), str(tmp_path))
    base = tmp_path / "workspace/investigations" / INVEST
    assert (base / "investigation.yaml").exists()
    assert (base / "studies/basal/study.yaml").exists()
    assert (base / "studies/basal_4x4/study.yaml").exists()
    inv = yaml.safe_load((base / "investigation.yaml").read_text(encoding="utf-8"))
    assert sorted(inv["studies"]) == ["basal", "basal_4x4"]
    s44 = yaml.safe_load(
        (base / "studies/basal_4x4/study.yaml").read_text(encoding="utf-8"))
    assert [t["measure"]["group"] for t in s44["behavior_tests"]] == ["statistical"]


def test_scaffold_is_idempotent_without_force(tmp_path):
    m = _manifest(tmp_path)
    scaffold(str(m), str(tmp_path))
    spath = tmp_path / "workspace/investigations" / INVEST / "studies/basal/study.yaml"
    spath.write_text("name: basal\nEDITED: true\n", encoding="utf-8")
    written = scaffold(str(m), str(tmp_path))            # no force
    assert spath not in written
    assert "EDITED" in spath.read_text(encoding="utf-8")  # not clobbered
    written2 = scaffold(str(m), str(tmp_path), force=True)
    assert spath in written2
    assert "EDITED" not in spath.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scaffold_comparison_studies.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.scaffold_comparison_studies`.

- [ ] **Step 3: Write `scripts/scaffold_comparison_studies.py`**

```python
#!/usr/bin/env python3
"""One-time scaffold: manifest -> investigation.yaml + per-condition study.yaml.

Idempotent: never overwrites an existing study unless --force. After scaffolding
the files are hand-owned (spec Decision 4); this is NOT part of the run/render
loop. Studies REFERENCE the manifest (comparison_manifest + condition); the
validator (scripts/validate_comparison_studies.py) guards drift.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
INVEST = "v2ecoli-vecoli-comparison"
CARD_ROOT = f"docs/report_cards/{INVEST}"
GRADED = {"standard", "statistical"}   # cards that produce a gating test


def condition_name(entry: dict) -> str:
    """A manifest entry's condition key: its explicit `name`, else the config
    filename stem with a leading 'cond_' and trailing scale suffix (_NxN) stripped."""
    if entry.get("name"):
        return entry["name"]
    stem = os.path.splitext(os.path.basename(entry["config"]))[0]
    if stem.startswith("cond_"):
        stem = stem[len("cond_"):]
    return re.sub(r"_\d+x\d+$", "", stem)


def build_study(cond: str, cards: list, manifest_rel: str) -> dict:
    graded = [c for c in cards if c in GRADED]
    return {
        "schema_version": 4,
        "name": cond,
        "investigation": INVEST,
        "title": f"v2ecoli reproduces vEcoli on {cond}",
        "status": "evaluated",
        "comparison_manifest": manifest_rel,
        "condition": cond,
        "question": f"Does v2ecoli reproduce vEcoli on the {cond} condition?",
        "report_cards": [f"{CARD_ROOT}/{cond}/index.html"],
        "behavior_tests": [
            {"name": f"{c}-vs-vecoli",
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {cond} ({c} card)?",
             "measure": {"kind": "report_card_axis",
                         "card": f"{CARD_ROOT}/{cond}", "group": c}}
            for c in graded],
        "runs": [
            {"name": f"{cond}-comparison", "kind": "analysis", "canonical": True,
             "description": f"v2e-compare study {cond}"}],
        "pipeline_gate": {"prerequisites": [], "enables": []},
    }


def build_investigation(conds: list) -> dict:
    return {
        "schema_version": 4,
        "name": INVEST,
        "title": "v2ecoli ↔ vEcoli comparison",
        "question": "Does v2ecoli reproduce vEcoli across nutrient conditions?",
        "studies": sorted(conds),
    }


def scaffold(manifest_path: str, ws_root: str, force: bool = False) -> list:
    spec = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    inv_dir = Path(ws_root) / "workspace/investigations" / INVEST
    studies_dir = inv_dir / "studies"
    try:
        manifest_rel = os.path.relpath(manifest_path, REPO)
    except ValueError:
        manifest_rel = manifest_path
    written = []
    conds = []
    for entry in spec.get("configs", []):
        cond = condition_name(entry)
        conds.append(cond)
        cards = entry.get("cards") or default_cards
        spath = studies_dir / cond / "study.yaml"
        if spath.exists() and not force:
            continue
        spath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(
            yaml.safe_dump(build_study(cond, cards, manifest_rel), sort_keys=False,
                           allow_unicode=True), encoding="utf-8")
        written.append(spath)
    inv_dir.mkdir(parents=True, exist_ok=True)
    ipath = inv_dir / "investigation.yaml"
    if force or not ipath.exists():
        ipath.write_text(
            yaml.safe_dump(build_investigation(conds), sort_keys=False,
                           allow_unicode=True), encoding="utf-8")
        written.append(ipath)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="comparison manifest JSON")
    ap.add_argument("--ws-root", default=str(REPO), help="repo/workspace root")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing studies/investigation")
    args = ap.parse_args(argv)
    written = scaffold(args.manifest, args.ws_root, force=args.force)
    for p in written:
        print(f"wrote {p}")
    if not written:
        print("nothing to write (all studies exist; use --force to overwrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scaffold_comparison_studies.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/scaffold_comparison_studies.py tests/test_scaffold_comparison_studies.py
git commit -m "feat(compare): one-time scaffold for comparison investigation/studies"
```

---

### Task 5: Drift validator (`scripts/validate_comparison_studies.py`)

**Files:**
- Create: `scripts/validate_comparison_studies.py`
- Test: `tests/test_validate_comparison_studies.py`

**Interfaces:**
- Consumes: `condition_name`, `INVEST`, `CARD_ROOT`, `GRADED` from `scripts.scaffold_comparison_studies` (Task 4).
- Produces: `validate(manifest_path, ws_root) -> list[str]` (empty = OK); `main(argv) -> int` (non-zero on any problem).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_comparison_studies.py
import json
import yaml
from scripts.scaffold_comparison_studies import scaffold, INVEST, CARD_ROOT
from scripts.validate_comparison_studies import validate


def _setup(tmp_path):
    m = tmp_path / "spec.json"
    m.write_text(json.dumps({
        "defaults": {"cards": ["config", "parca", "standard"]},
        "configs": [{"config": "configs/cond_basal_1x4.json"}]}), encoding="utf-8")
    scaffold(str(m), str(tmp_path))
    return m, (tmp_path / "workspace/investigations" / INVEST
               / "studies/basal/study.yaml")


def test_validate_passes_on_scaffolded(tmp_path):
    m, _ = _setup(tmp_path)
    assert validate(str(m), str(tmp_path)) == []


def test_validate_flags_unknown_condition(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["condition"] = "no_such_condition"
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("not in manifest" in p for p in problems)


def test_validate_flags_group_mismatch(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["behavior_tests"][0]["measure"]["group"] = "statistical"  # manifest says standard
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("graded cards" in p for p in problems)


def test_validate_flags_bad_card_path(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["behavior_tests"][0]["measure"]["card"] = "docs/report_cards/wrong/basal"
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("card" in p for p in problems)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validate_comparison_studies.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.validate_comparison_studies`.

- [ ] **Step 3: Write `scripts/validate_comparison_studies.py`**

```python
#!/usr/bin/env python3
"""Validate that comparison studies match their manifest (drift guard).

For the v2ecoli-vecoli-comparison investigation, assert per study: its
`condition` exists in the manifest; its report_card_axis behavior_test groups
exactly equal the manifest's graded cards for that condition; each test's `card`
path is the canonical <CARD_ROOT>/<condition>. Exits non-zero on any drift so
this can run in CI / pre-merge. This is how "studies reference the manifest"
stays honest without auto-generating them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.scaffold_comparison_studies import (
    REPO, INVEST, CARD_ROOT, GRADED, condition_name)


def validate(manifest_path: str, ws_root: str) -> list:
    spec = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    manifest_conds = {}
    for entry in spec.get("configs", []):
        cond = condition_name(entry)
        cards = entry.get("cards") or default_cards
        manifest_conds[cond] = sorted(c for c in cards if c in GRADED)

    problems = []
    studies_dir = Path(ws_root) / "workspace/investigations" / INVEST / "studies"
    if not studies_dir.is_dir():
        return [f"no studies dir: {studies_dir}"]
    for sdir in sorted(studies_dir.glob("*")):
        spath = sdir / "study.yaml"
        if not spath.exists():
            continue
        study = yaml.safe_load(spath.read_text(encoding="utf-8")) or {}
        cond = study.get("condition") or sdir.name
        if cond not in manifest_conds:
            problems.append(f"{sdir.name}: condition {cond!r} not in manifest")
            continue
        axis_tests = [t for t in study.get("behavior_tests", [])
                      if (t.get("measure") or {}).get("kind") == "report_card_axis"]
        groups = sorted(t["measure"].get("group", "") for t in axis_tests)
        if groups != manifest_conds[cond]:
            problems.append(
                f"{cond}: behavior_test groups {groups} != manifest graded "
                f"cards {manifest_conds[cond]}")
        expected_card = f"{CARD_ROOT}/{cond}"
        for t in axis_tests:
            if t["measure"].get("card") != expected_card:
                problems.append(
                    f"{cond}: test {t.get('name')!r} card "
                    f"{t['measure'].get('card')!r} != {expected_card!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="comparison manifest JSON")
    ap.add_argument("--ws-root", default=str(REPO), help="repo/workspace root")
    args = ap.parse_args(argv)
    problems = validate(args.manifest, args.ws_root)
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if problems:
        return 1
    print("comparison studies OK (match manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_validate_comparison_studies.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_comparison_studies.py tests/test_validate_comparison_studies.py
git commit -m "feat(compare): validator guarding studies-vs-manifest drift"
```

---

### Task 6: CLI front door (`scripts/compare_cli.py` + console script + `--condition`)

**Files:**
- Create: `scripts/compare_cli.py`
- Modify: `scripts/run_comparison.py:60-104` (add `--condition` filter)
- Modify: `pyproject.toml:74-78` (`[project.scripts]`)
- Test: `tests/test_compare_cli.py`

**Interfaces:**
- Consumes: `run_comparison.main(argv)` (now honoring `--condition`); `scaffold_comparison_studies.scaffold`; `validate_comparison_studies.validate`.
- Produces: `main(argv) -> int` with verbs `run` and `study`; helpers `_run_investigation(...)`, `_run_study(...)`, `_resolve_study(name_or_path)`. Console script `v2e-compare`.

- [ ] **Step 1: Add `--condition` to `run_comparison.py`** (a prerequisite the CLI relies on)

In `main()` after `ap.add_argument("--render-only", ...)` add:

```python
    ap.add_argument("--condition", default=None,
                    help="run/render only the config whose name or condition "
                         "matches (used by `v2e-compare study`)")
```

And immediately after `configs = spec.get("configs", [])` / the empty-check, add the filter:

```python
    if args.condition:
        configs = [e for e in configs
                   if (e.get("name") or condition_of(e["config"], fork)) == args.condition]
        if not configs:
            sys.exit(f"no config matches condition {args.condition!r}")
```

(The render subprocess still passes the whole `--manifest`; `assemble_from_manifest` already skips conditions without stores, so only the run condition renders + emits a verdict.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_compare_cli.py
import scripts.compare_cli as cli


def test_run_sequences_scaffold_then_run_then_validate(monkeypatch):
    seq = []
    monkeypatch.setattr(cli.scaffold_mod, "scaffold",
                        lambda m, r, force=False: seq.append("scaffold"))
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: seq.append(("run", argv)) or 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    rc = cli.main(["run", "comparison_spec.json"])
    assert rc == 0
    assert seq[0] == "scaffold"
    assert seq[1][0] == "run"


def test_run_returns_nonzero_on_drift(monkeypatch):
    monkeypatch.setattr(cli.scaffold_mod, "scaffold", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_comparison, "main", lambda argv: 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: ["basal: drift"])
    assert cli.main(["run", "comparison_spec.json"]) == 1


def test_run_ray_selects_ray_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.scaffold_mod, "scaffold", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: captured.update(argv=argv) or 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    cli.main(["run", "spec.json", "--ray"])
    assert "ray" in captured["argv"]
    assert "serial" not in captured["argv"]


def test_render_only_skips_scaffold(monkeypatch):
    seq = []
    monkeypatch.setattr(cli.scaffold_mod, "scaffold",
                        lambda *a, **k: seq.append("scaffold"))
    monkeypatch.setattr(cli.run_comparison, "main", lambda argv: 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    cli.main(["run", "spec.json", "--render-only"])
    assert "scaffold" not in seq


def test_study_resolves_manifest_and_condition(tmp_path, monkeypatch):
    sdir = tmp_path / "basal"
    sdir.mkdir()
    (sdir / "study.yaml").write_text(
        "comparison_manifest: comparison_spec.json\ncondition: basal\nname: basal\n",
        encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: captured.update(argv=argv) or 0)
    rc = cli._run_study(str(sdir), None, "out/x", False, False)
    assert rc == 0
    argv = captured["argv"]
    assert "--condition" in argv and "basal" in argv
    assert any(a.endswith("comparison_spec.json") for a in argv)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.compare_cli`.

- [ ] **Step 4: Write `scripts/compare_cli.py`**

```python
#!/usr/bin/env python3
"""v2e-compare — one front door for the comparison-harness investigation.

  v2e-compare run <manifest> [--ray] [--out DIR] [--render-only]
  v2e-compare study <name|path> [--ray] [--manifest M] [--out DIR] [--render-only]

`run` drives the whole investigation: scaffold studies if missing -> run both
engines per condition -> emit per-condition verdicts (via the renderer) ->
validate studies-vs-manifest -> report dashboard-ready. `study` runs ONE
condition, resolving its manifest from the study's own `comparison_manifest`
back-link (spec Decision 3). Sims run serial+local by default; --ray (or
V2E_MODE=ray) fans conditions out in parallel for the mini.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import run_comparison  # noqa: E402
from scripts import scaffold_comparison_studies as scaffold_mod  # noqa: E402
from scripts import validate_comparison_studies as validate_mod  # noqa: E402

INVEST = scaffold_mod.INVEST
CARD_ROOT = scaffold_mod.CARD_ROOT
STUDIES = REPO / "workspace/investigations" / INVEST / "studies"


def _abs_manifest(manifest: str) -> str:
    return manifest if os.path.isabs(manifest) else str(REPO / manifest)


def _run_investigation(manifest, out, ray, render_only) -> int:
    manifest = _abs_manifest(manifest)
    if not render_only:                                   # 1. scaffold if missing
        scaffold_mod.scaffold(manifest, str(REPO), force=False)
    mode = "ray" if ray else "serial"
    argv = [manifest, "--out", out, "--mode", mode]       # 2-3. run + verdict (render writes verdict)
    if render_only:
        argv.append("--render-only")
    rc = run_comparison.main(argv)
    if rc:
        return rc
    problems = validate_mod.validate(manifest, str(REPO))  # 4. validate
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"investigation ready: workspace/investigations/{INVEST}")  # 5.
    return 0


def _resolve_study(name_or_path):
    p = Path(name_or_path)
    if p.name == "study.yaml":
        spath = p
    elif p.is_dir():
        spath = p / "study.yaml"
    else:
        spath = STUDIES / name_or_path / "study.yaml"
    if not spath.exists():
        sys.exit(f"study not found: {spath}")
    return yaml.safe_load(spath.read_text(encoding="utf-8")) or {}, spath


def _run_study(name_or_path, manifest_override, out, ray, render_only) -> int:
    study, spath = _resolve_study(name_or_path)
    manifest = manifest_override or study.get("comparison_manifest")
    cond = study.get("condition") or study.get("name")
    if not manifest:
        sys.exit(f"{spath}: no comparison_manifest (pass --manifest)")
    if not cond:
        sys.exit(f"{spath}: no condition/name")
    mode = "ray" if ray else "serial"
    argv = [_abs_manifest(manifest), "--out", out, "--mode", mode,
            "--condition", cond]
    if render_only:
        argv.append("--render-only")
    rc = run_comparison.main(argv)
    if rc == 0:
        print(f"study '{cond}' done; verdict: {CARD_ROOT}/{cond}/"
              f"report_card_verdict.json")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="v2e-compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the whole comparison investigation")
    pr.add_argument("manifest")
    pr.add_argument("--out", default="out/report")
    pr.add_argument("--ray", action="store_true")
    pr.add_argument("--render-only", action="store_true")

    ps = sub.add_parser("study", help="run a single study/condition")
    ps.add_argument("name", help="study name or path")
    ps.add_argument("--manifest", default=None, help="override the study's back-link")
    ps.add_argument("--out", default="out/report")
    ps.add_argument("--ray", action="store_true")
    ps.add_argument("--render-only", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return _run_investigation(args.manifest, args.out, args.ray, args.render_only)
    return _run_study(args.name, args.manifest, args.out, args.ray, args.render_only)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Register the console script** — edit `pyproject.toml` `[project.scripts]`

Add one line under the existing entries:

```toml
[project.scripts]
v2ecoli-parca = "v2ecoli.cli.parca:main"
v2ecoli-colony = "v2ecoli.cli.colony:main"
v2ecoli-workflow = "v2ecoli.workflow.run:main"
v2ecoli-analyze = "v2ecoli.workflow.analysis_runner:main"
v2e-compare = "scripts.compare_cli:main"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py tests/test_config_translation.py -q`
Expected: PASS (the CLI tests pass; the run_comparison `--condition` change didn't break config translation).

- [ ] **Step 7: Commit**

```bash
git add scripts/compare_cli.py scripts/run_comparison.py pyproject.toml tests/test_compare_cli.py
git commit -m "feat(compare): v2e-compare CLI (run investigation / run study) + --condition filter"
```

---

### Task 7: Generate the real investigation + studies, validate, wire dashboard

**Files:**
- Create (generated, then hand-owned): `workspace/investigations/v2ecoli-vecoli-comparison/investigation.yaml` + `studies/<cond>/study.yaml` (6 conditions)
- Verify: `investigations -> workspace/investigations` root symlink (already exists)

**Interfaces:**
- Consumes: the scaffold (Task 4), validator (Task 5), CLI (Task 6).

- [ ] **Step 1: Scaffold the real studies from the 6-entry manifest**

Run: `.venv/bin/python scripts/scaffold_comparison_studies.py comparison.5cond_1x4.json`
Expected: prints `wrote .../studies/basal/study.yaml` … for the 5 `cond_*_1x4` conditions + `basal_4x4`, plus `investigation.yaml`.

- [ ] **Step 2: Validate the generated studies against the manifest**

Run: `.venv/bin/python scripts/validate_comparison_studies.py comparison.5cond_1x4.json`
Expected: prints `comparison studies OK (match manifest)` and exits 0.

- [ ] **Step 3: Confirm the dashboard scanner can see the investigation**

Run: `ls -la investigations/v2ecoli-vecoli-comparison/studies`
Expected: lists the 6 study dirs (the root `investigations` symlink already points at `workspace/investigations`, so the scanner's root-layout requirement is satisfied — no new symlink needed).

- [ ] **Step 4: Smoke the verdict path from existing stores (no new sims)**

Render-only from the existing 4×4×5 mediafix stores (these already exist under `out/smoke5`; if absent on this machine, skip this step and note it for the human verification pass):

Run: `.venv/bin/python scripts/compare_cli.py run comparison.5cond_1x4.json --out out/smoke5 --render-only`
Expected: writes `docs/report_cards/v2ecoli-vecoli-comparison/basal/report_card_verdict.json` (+ other rendered conditions); validator passes; prints `investigation ready: …`. Confirm with:
`.venv/bin/python -c "import json;d=json.load(open('docs/report_cards/v2ecoli-vecoli-comparison/basal/report_card_verdict.json'));print(d['overall'], list(d['groups']))"`
Expected: prints a verdict (e.g. `within_tol ['config', 'parca', 'standard']`).

- [ ] **Step 5: Commit the authored investigation + studies**

```bash
git add -f workspace/investigations/v2ecoli-vecoli-comparison docs/report_cards/v2ecoli-vecoli-comparison
git commit -m "feat(compare): authored v2ecoli-vecoli-comparison investigation + per-condition studies"
```

(`-f` because `docs/report_cards/**` and parts of `workspace/**` may be gitignored — match the existing convention used by the ketchup/beulig cards.)

- [ ] **Step 6: Run the full test suite for the touched areas**

Run: `.venv/bin/python -m pytest tests/test_comparison_verdict.py tests/test_card_verdicts.py tests/test_assemble_verdict.py tests/test_scaffold_comparison_studies.py tests/test_validate_comparison_studies.py tests/test_compare_cli.py -q`
Expected: all PASS.

---

## Human verification (post-implementation, not a coding task)

These need the dashboard running and are for the human to confirm (the design's Phase 5):

1. Start the read-only dashboard against `v2e-main` and open the **v2ecoli-vecoli-comparison** investigation; confirm it lists the 6 condition studies.
2. Confirm each study renders its report cards and a pass/drift/fail pill per graded card (driven by `report_card_axis` reading the verdict JSON).
3. Optionally run a real single-condition pass end-to-end on the mini: `V2E_MODE=ray .venv/bin/python scripts/compare_cli.py study basal` (heavy; needs `V2E_VECOLI_DIR` + ParCa caches).

## Self-Review

- **Spec coverage:** Component 0 (CLI)→Task 6; Component 1 (verdict.py)→Task 1; card payloads→Task 2; both-paths wiring→Task 3 (run_comparison renders via assemble_from_manifest, so its render subprocess emits verdicts — Decision 5 satisfied); Component 3 (scaffold)→Task 4; Component 4 (validator)→Task 5; authored studies + Phase 4 gating proof→Task 1's evaluator round-trip test + Task 7; Component 5 (dashboard symlink)→already exists, Task 7 Step 3 + human verification. All spec sections map to a task.
- **Placeholder scan:** none — every code step has complete code; the one conditional ("if `out/smoke5` absent, skip") is an explicit machine-state branch, not a TODO.
- **Type consistency:** `card_verdicts` shape `{card: {verdict, axes}}` is identical in Tasks 1, 3; axis dict keys `{id,label,verdict,value,meter,detail}` match Task 2 producers and Task 1 consumer; `condition_name`/`INVEST`/`CARD_ROOT`/`GRADED` defined in Task 4 and imported unchanged in Tasks 5–6; `report_card_axis` measure `{kind,card,group}` matches the evaluator at `pbg_v2ecoli/evaluators.py`.
