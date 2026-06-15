# Workspace-Pluggable Evaluators — Report Cards as Acceptance Evidence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a workspace grade study tests with its own evaluator (no framework-core change), and make the v2ecoli report card the first such evaluator so its per-group verdicts flow into `computed_outcomes → gate → acceptance criteria`.

**Architecture:** A generic `CUSTOM_EVALUATORS` seam in `pbg-superpowers/study_evaluator.py` dispatches any non-native `measure.kind` to an evaluator registered by the workspace's `pbg_<name>` package via a `register_evaluators(registry)` hook (discovered exactly like `build_core()`). `pbg_v2ecoli` registers a `report_card_axis` evaluator that reads a machine-readable `report_card_verdict.json` (emitted by the card renderer) and aggregates one card group → one outcome.

**Tech Stack:** Python 3.12, pytest, ruamel.yaml (round-trip), polars/duckdb (RunReader — not needed by this feature), the existing `v2ecoli.library.report_card` grader.

**Spec:** `docs/superpowers/specs/2026-06-13-workspace-evaluator-extension-report-cards.md`

**Repos / worktrees (three working trees):**
- `pbg-superpowers` at `/Users/eranagmon/code/pbg-superpowers` — Tasks 1–2. Branch first: `feat/pluggable-workspace-evaluators`.
- v2ecoli worktree at `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large` (branch `worktree-showcase-4-equivalence-large`) — Tasks 3, 4, 5, 6 (the `reports/`, `pbg_v2ecoli/`, and `workspace/studies/` changes). This is the editable worktree the mini runs from.

> **Editable-install note:** the framework seam (Tasks 1–2) must be importable by whatever Python runs `compute_outcomes`. The v2ecoli worktree venv runs a git-pinned `pbg-superpowers`; after Task 2, `uv pip install -e /Users/eranagmon/code/pbg-superpowers --no-deps` into the worktree venv (and the mini's, if running there) so the seam is live. This is called out again in Task 5.

---

## File Structure

**pbg-superpowers (framework seam — generic, report-card-agnostic):**
- Modify `pbg_superpowers/study_evaluator.py` — add `CUSTOM_EVALUATORS`, `load_workspace_evaluators`, `_workspace_package_slug`; thread `ws_root` through `evaluate_study`/`evaluate_test`; dispatch.
- Create `tests/test_workspace_evaluators.py` — loader + dispatch unit tests with a fixture workspace.

**v2ecoli worktree:**
- Modify `v2ecoli/library/report_card.py` — add `verdict_json(report)` (flat-axes → grouped `v1` schema) next to `grade_card`.
- Modify `reports/population_phenotype_basal_report.py` — emit `report_card_verdict.json` in `main()`.
- Create `pbg_v2ecoli/evaluators.py` — `register_evaluators` + `evaluate_report_card_group`.
- Create `pbg_v2ecoli/tests/test_report_card_evaluator.py` — evaluator unit tests with fixture verdict JSON.
- Modify `workspace/studies/showcase-6-equivalence-large/study.yaml` — migrate the 5 tests to `measure.kind: report_card_axis`.

---

## Task 1: Generic evaluator seam in pbg-superpowers

**Files:**
- Modify: `/Users/eranagmon/code/pbg-superpowers/pbg_superpowers/study_evaluator.py` (RUN_DATA_KINDS block ~35–47; `evaluate_study` 66–81; `evaluate_test` 84–146)
- Test: `/Users/eranagmon/code/pbg-superpowers/tests/test_workspace_evaluators.py`

- [ ] **Step 1: Branch the repo**

```bash
cd /Users/eranagmon/code/pbg-superpowers
git checkout -b feat/pluggable-workspace-evaluators
```

- [ ] **Step 2: Write the failing test for the loader + dispatch**

Create `tests/test_workspace_evaluators.py`:

```python
import sys
import textwrap
from pathlib import Path

import pytest

from pbg_superpowers import study_evaluator as se


def _make_fixture_ws(tmp_path: Path, kind: str = "toy_kind") -> Path:
    """A throwaway workspace whose pbg_<name>.evaluators registers one evaluator."""
    ws = tmp_path / "ws"
    pkg = ws / "pbg_toyws"
    pkg.mkdir(parents=True)
    (ws / "workspace.yaml").write_text("name: toyws\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "evaluators.py").write_text(textwrap.dedent(f"""
        def _toy(test, reader, ws_root):
            return {{"result": "PASS", "evaluated_by": "toy",
                    "detail": "from " + str(ws_root)}}
        def register_evaluators(registry):
            registry["{kind}"] = _toy
    """), encoding="utf-8")
    return ws


def test_loader_finds_and_calls_hook(tmp_path):
    ws = _make_fixture_ws(tmp_path)
    se.clear_workspace_evaluator_cache()
    reg = se.load_workspace_evaluators(ws)
    assert "toy_kind" in reg
    out = reg["toy_kind"]({}, None, ws)
    assert out["evaluated_by"] == "toy"


def test_loader_absent_hook_returns_empty(tmp_path):
    ws = tmp_path / "bare"
    ws.mkdir()
    (ws / "workspace.yaml").write_text("name: bare\n", encoding="utf-8")
    se.clear_workspace_evaluator_cache()
    assert se.load_workspace_evaluators(ws) == {}


def test_loader_broken_hook_degrades(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    pkg = ws / "pbg_brokenws"
    pkg.mkdir(parents=True)
    (ws / "workspace.yaml").write_text("name: brokenws\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "evaluators.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    se.clear_workspace_evaluator_cache()
    assert se.load_workspace_evaluators(ws) == {}  # never raises


def test_evaluate_test_dispatches_to_registered_evaluator(tmp_path):
    ws = _make_fixture_ws(tmp_path, kind="toy_kind")
    se.clear_workspace_evaluator_cache()
    test = {"name": "t", "measure": {"kind": "toy_kind"}, "pass_if": {"op": "x"}}
    out = se.evaluate_test(test, reader=None, ws_root=ws)
    assert out["evaluated_by"] == "toy"


def test_evaluate_test_unknown_kind_no_ws_still_agent():
    test = {"name": "t", "measure": {"kind": "nope"}, "pass_if": {"op": "x"}}
    out = se.evaluate_test(test, reader=None, ws_root=None)
    assert out["evaluated_by"] == "agent"
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_workspace_evaluators.py -v`
Expected: FAIL — `AttributeError: module 'pbg_superpowers.study_evaluator' has no attribute 'load_workspace_evaluators'` (and `evaluate_test()` got an unexpected keyword `ws_root`).

- [ ] **Step 4: Add the registry + loader to `study_evaluator.py`**

Insert after the `RUN_DATA_KINDS` block (after line 47):

```python
# ---------------------------------------------------------------------------
# Pluggable workspace evaluators (generic seam — framework ships none).
#
# A workspace augments evaluation by shipping a `register_evaluators(registry)`
# hook in its `pbg_<name>` package (the same package that hosts build_core()).
# When evaluate_test hits a measure.kind that is NOT a native RUN_DATA_KIND, it
# consults the workspace registry before falling back to the agent bucket.
# ---------------------------------------------------------------------------
from typing import Callable  # noqa: E402  (kept local to this block)

_WS_EVALUATOR_CACHE: dict[str, dict[str, Callable]] = {}


def clear_workspace_evaluator_cache() -> None:
    """Drop the per-workspace evaluator cache (used by tests / after reinstall)."""
    _WS_EVALUATOR_CACHE.clear()


def _workspace_package_slug(ws_root) -> str:
    """pbg_<name> for the workspace, mirroring build_core()'s home.

    NOTE: deliberately uses the pbg_<name> convention (where build_core lives),
    NOT workspace.yaml `package_path` (which points at the simulation package,
    e.g. v2ecoli). dashes -> underscores.
    """
    import yaml
    from pathlib import Path
    name = "workspace"
    wy = Path(ws_root) / "workspace.yaml"
    if wy.is_file():
        data = yaml.safe_load(wy.read_text(encoding="utf-8")) or {}
        name = data.get("name") or name
    return "pbg_" + str(name).replace("-", "_")


def load_workspace_evaluators(ws_root) -> dict[str, Callable]:
    """Import the workspace's pbg_<name>.evaluators and collect its registrations.

    Returns a {measure_kind: callable} dict. Empty if ws_root is None, the
    package/hook is absent, or the hook raises (a broken workspace hook must
    never crash evaluation — degrade to the agent bucket). Cached per ws_root.
    """
    import sys
    from pathlib import Path
    if ws_root is None:
        return {}
    key = str(Path(ws_root).resolve())
    if key in _WS_EVALUATOR_CACHE:
        return _WS_EVALUATOR_CACHE[key]
    registry: dict[str, Callable] = {}
    try:
        if key not in sys.path:
            sys.path.insert(0, key)
        pkg = _workspace_package_slug(ws_root)
        mod = __import__(f"{pkg}.evaluators", fromlist=["register_evaluators"])
        hook = getattr(mod, "register_evaluators", None)
        if callable(hook):
            hook(registry)
    except Exception:  # noqa: BLE001 — never let a workspace hook break evaluation
        registry = {}
    _WS_EVALUATOR_CACHE[key] = registry
    return registry
```

- [ ] **Step 5: Thread `ws_root` through `evaluate_study` and `evaluate_test` + dispatch**

Replace `evaluate_study` (lines 66–81) and the head of `evaluate_test` (lines 84–100) with:

```python
def evaluate_study(spec: dict, reader: "RunReader", ws_root=None) -> dict[str, dict]:
    """Evaluate all behavior tests in a study spec.

    ws_root enables workspace-pluggable evaluators (load_workspace_evaluators);
    omit it to evaluate with native kinds only.
    """
    tests = spec.get("tests") or spec.get("behavior_tests") or []
    results: dict[str, dict] = {}
    for i, test in enumerate(tests):
        name = test.get("name", f"test_{i}")
        results[name] = evaluate_test(test, reader, ws_root=ws_root)
    return results


def evaluate_test(test: dict, reader: "RunReader", ws_root=None) -> dict:
    """Evaluate a single behavior test against a run.

    Resolution order: native RUN_DATA_KIND (code) → workspace-registered
    evaluator for the kind → agent bucket.
    """
    # 1. Require measure block
    measure = test.get("measure")
    if not measure:
        return _agent("missing measure block")

    # 2. Kind: native run-data, else a workspace-registered evaluator, else agent
    kind = measure.get("kind", "")
    if kind not in RUN_DATA_KINDS:
        registry = load_workspace_evaluators(ws_root)
        evaluator = registry.get(kind)
        if evaluator is not None:
            try:
                return evaluator(test, reader, ws_root)
            except Exception as exc:  # noqa: BLE001
                return _agent(f"workspace evaluator {kind!r} error: {exc}")
        return _agent(f"non-run-data kind: {kind!r}")
```

(Leave the rest of `evaluate_test`, lines 102 onward, unchanged.)

- [ ] **Step 6: Run the test to confirm it passes**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_workspace_evaluators.py -v`
Expected: PASS (all 5).

- [ ] **Step 7: Run the existing evaluator suite to confirm no regression**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/ -k "evaluator or outcome or study" -q`
Expected: PASS (the new optional `ws_root` arg defaults to None — existing callers unaffected).

- [ ] **Step 8: Commit**

```bash
cd /Users/eranagmon/code/pbg-superpowers
git add pbg_superpowers/study_evaluator.py tests/test_workspace_evaluators.py
git commit -m "feat(evaluator): generic workspace-pluggable evaluator seam

evaluate_test dispatches a non-native measure.kind to an evaluator registered
by the workspace's pbg_<name>.register_evaluators(registry) hook (discovered
like build_core), before the agent fallback. Framework ships none; broken/absent
hooks degrade to agent. ws_root threaded through evaluate_study/evaluate_test."
```

---

## Task 2: Pass `ws_root` from `compute_outcomes` into `evaluate_study`

**Files:**
- Modify: `/Users/eranagmon/code/pbg-superpowers/pbg_superpowers/study_evaluator.py:966` (the `evaluate_study(spec, reader)` call inside `compute_outcomes`)
- Test: `/Users/eranagmon/code/pbg-superpowers/tests/test_workspace_evaluators.py` (add one)

- [ ] **Step 1: Write the failing test**

`compute_outcomes` resolves a real run store and lazily imports `RunReader`, so a
behavioral test would need a full fixture run. The robust, fast check is a
source-level assertion that the call site threads `ws_root`. Append to
`tests/test_workspace_evaluators.py`:

```python
import inspect


def test_compute_outcomes_threads_ws_root_into_evaluate_study():
    src = inspect.getsource(se.compute_outcomes)
    assert "evaluate_study(spec, reader, ws_root=ws_root)" in src, (
        "compute_outcomes must pass ws_root to evaluate_study so workspace "
        "evaluators are reachable"
    )
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_workspace_evaluators.py::test_compute_outcomes_threads_ws_root -v`
Expected: FAIL — current call is `evaluate_study(spec, reader)` (no `ws_root`).

- [ ] **Step 3: Make the one-line change**

In `compute_outcomes`, change line 966 from:

```python
            outcomes = evaluate_study(spec, reader)
```
to:
```python
            outcomes = evaluate_study(spec, reader, ws_root=ws_root)
```

- [ ] **Step 4: Run to confirm pass**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_workspace_evaluators.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/pbg-superpowers
git add pbg_superpowers/study_evaluator.py tests/test_workspace_evaluators.py
git commit -m "feat(evaluator): compute_outcomes threads ws_root into evaluate_study"
```

---

## Task 3: Emit `report_card_verdict.json` from the renderer

**Files:**
- Modify: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/v2ecoli/library/report_card.py` (add `verdict_json`, near `grade_card` ~line 121)
- Modify: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/reports/population_phenotype_basal_report.py` (imports ~22; `main()` ~73–81)
- Test: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/tests/test_report_card_verdict_json.py`

> All commands in Tasks 3–6 run from the worktree with its venv:
> `cd /Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large` and use `.venv/bin/python`.

- [ ] **Step 1: Write the failing test for `verdict_json`**

Create `tests/test_report_card_verdict_json.py`:

```python
from v2ecoli.library.report_card import verdict_json


def _fake_report():
    # grade_card output shape: {"overall": str, "axes": {path: {group, verdict, ...}}}
    return {
        "overall": "mismatch",
        "axes": {
            "physiology.doubling_time": {"group": "Physiology", "label": "Doubling time",
                "verdict": "within_tol", "value": 0.84, "meter": "Δ = -2.2%",
                "detail": {"p": 0.014, "cohens_d": -0.26, "delta_rel": -0.022}},
            "fluxes.o2": {"group": "Exchange fluxes", "label": "O2 exchange",
                "verdict": "mismatch", "value": -0.45, "meter": "Δ = -40.4%",
                "detail": {"p": 0.0, "cohens_d": 0.89, "delta_rel": -0.404}},
        },
    }


def test_verdict_json_groups_axes_and_slugs_group_names():
    vj = verdict_json(_fake_report(), model_ref="abc1234",
                      reference_model="vEcoli (v1)", generated="2026-06-13 00:00")
    assert vj["schema"] == "report_card_verdict/v1"
    assert vj["overall"] == "mismatch"
    assert set(vj["groups"]) == {"physiology", "exchange_fluxes"}
    phys = vj["groups"]["physiology"]
    assert phys["axes"][0]["id"] == "physiology.doubling_time"
    assert phys["axes"][0]["verdict"] == "within_tol"
    # group verdict = worst axis in group
    assert vj["groups"]["exchange_fluxes"]["verdict"] == "mismatch"
    assert vj["groups"]["physiology"]["verdict"] == "within_tol"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_report_card_verdict_json.py -v`
Expected: FAIL — `ImportError: cannot import name 'verdict_json'`.

- [ ] **Step 3: Implement `verdict_json` in `report_card.py`**

Add near `grade_card` (after it):

```python
# Worst-first severity used to roll a group's axis verdicts into one verdict.
_VERDICT_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def _slug_group(label: str) -> str:
    """'Exchange fluxes' -> 'exchange_fluxes'; 'Gene expression' -> 'gene_expression'."""
    return (label or "ungrouped").strip().lower().replace("&", "and").replace(" ", "_")


def verdict_json(report: dict, *, model_ref: str = "", reference_model: str = "",
                 generated: str = "") -> dict:
    """Serialize a grade_card() report into the machine-readable v1 verdict schema.

    grade_card returns FLAT axes keyed by path, each carrying a `group` label and
    a `verdict`. This regroups them by slugged group name and computes each
    group's verdict as the worst (most severe) axis verdict in that group.
    """
    groups: dict[str, dict] = {}
    for path, ax in (report.get("axes") or {}).items():
        gslug = _slug_group(ax.get("group", ""))
        g = groups.setdefault(gslug, {"verdict": "ungraded", "axes": []})
        v = ax.get("verdict", "ungraded")
        g["axes"].append({
            "id": path,
            "label": ax.get("label", path),
            "verdict": v,
            "value": ax.get("value"),
            "meter": ax.get("meter"),
            "detail": ax.get("detail") or {},
        })
        if _VERDICT_SEVERITY.get(v, 0) > _VERDICT_SEVERITY.get(g["verdict"], 0):
            g["verdict"] = v
    return {
        "schema": "report_card_verdict/v1",
        "model_ref": model_ref,
        "reference_model": reference_model,
        "generated": generated,
        "overall": report.get("overall", "ungraded"),
        "groups": groups,
    }
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_report_card_verdict_json.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the emit into the renderer**

In `reports/population_phenotype_basal_report.py`, update the import block (~line 22) to add `grade_card` and `verdict_json`:

```python
from v2ecoli.library.report_card import (
    card_from_analysis, grade_card, verdict_json, load_json, merge_vectors,
    render_html, render_markdown,
)
```

Then in `main()`, after `out_dir`/`model_ref`/`generated` are set and **before** `render_markdown` (i.e. between current lines 78 and 80), add:

```python
    import json
    report = grade_card(card, reference)
    with open(os.path.join(out_dir, "report_card_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict_json(report, model_ref=model_ref,
                               reference_model=reference.get("stimulus", {}).get(
                                   "reference_model", reference.get("reference_model", "")),
                               generated=generated), f, indent=2)
```

- [ ] **Step 6: Regenerate the showcase-6 verdict JSON from the real card (verify emit end-to-end)**

The card was already rendered on the mini. Re-run the renderer locally against the committed analysis is not possible without the sweep; instead emit the verdict JSON on the mini where the sweep lives:

Run:
```bash
ssh mini 'cd ~/code/v2ecoli-showcase6 && git pull --ff-only && \
  .venv/bin/python reports/population_phenotype_basal_report.py \
    --analysis out/ppb16_parallel/parquet/analysis.json \
    --reference docs/report_cards/population_phenotype_basal/vs_vecoli/vecoli_reference.json \
    --sweep-dir out/ppb16_parallel/parquet --gen-lb 3 --model-ref bd2123d2 \
    --out-dir docs/report_cards/population_phenotype_basal/vs_vecoli/ && \
  ls -la docs/report_cards/population_phenotype_basal/vs_vecoli/report_card_verdict.json'
```
Expected: `report_card_verdict.json` written. Then `scp` it into the worktree:
```bash
scp mini:/Users/eranagmon/code/v2ecoli-showcase6/docs/report_cards/population_phenotype_basal/vs_vecoli/report_card_verdict.json \
    docs/report_cards/population_phenotype_basal/vs_vecoli/report_card_verdict.json
```
Expected: a `groups` object with `physiology / composition / ribosomes / exchange_fluxes / gene_expression` (5 groups), `exchange_fluxes.verdict == "mismatch"`.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/library/report_card.py reports/population_phenotype_basal_report.py \
        tests/test_report_card_verdict_json.py \
        docs/report_cards/population_phenotype_basal/vs_vecoli/report_card_verdict.json
git commit -m "feat(report-card): emit machine-readable report_card_verdict.json (v1)

grade_card output (flat axes) regrouped into the v1 verdict schema (per-group
verdict = worst axis). The renderer writes it next to report_card.html."
```

---

## Task 4: The `report_card_axis` evaluator in pbg_v2ecoli

**Files:**
- Create: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/pbg_v2ecoli/evaluators.py`
- Create: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/pbg_v2ecoli/tests/__init__.py` (empty) and `.../tests/test_report_card_evaluator.py`

- [ ] **Step 1: Write the failing evaluator test**

Create `pbg_v2ecoli/tests/test_report_card_evaluator.py`:

```python
import json
from pathlib import Path

from pbg_v2ecoli.evaluators import evaluate_report_card_group, register_evaluators


def _ws_with_verdict(tmp_path: Path, groups: dict) -> Path:
    card = tmp_path / "docs" / "card"
    card.mkdir(parents=True)
    (card / "report_card_verdict.json").write_text(json.dumps({
        "schema": "report_card_verdict/v1", "overall": "mismatch", "groups": groups,
    }), encoding="utf-8")
    return tmp_path


def _test(group):
    return {"name": f"{group}-test",
            "measure": {"kind": "report_card_axis", "card": "docs/card", "group": group}}


def test_register_exposes_report_card_axis():
    reg = {}
    register_evaluators(reg)
    assert "report_card_axis" in reg


def test_mismatch_group_fails(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"flux": {"verdict": "mismatch",
        "axes": [{"id": "o2", "verdict": "mismatch"}]}})
    out = evaluate_report_card_group(_test("flux"), None, ws)
    assert out["result"] == "FAIL"
    assert out["evaluated_by"] == "report_card"


def test_drift_group_passes_with_caveat(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"ribo": {"verdict": "drift",
        "axes": [{"id": "total", "verdict": "drift"}, {"id": "ef", "verdict": "within_tol"}]}})
    out = evaluate_report_card_group(_test("ribo"), None, ws)
    assert out["result"] == "PASS"
    assert out["caveat"] == "drift"


def test_within_tol_group_passes(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"comp": {"verdict": "within_tol",
        "axes": [{"id": "protein", "verdict": "within_tol"}]}})
    out = evaluate_report_card_group(_test("comp"), None, ws)
    assert out["result"] == "PASS"
    assert "caveat" not in out


def test_all_ungraded_group_skips(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"x": {"verdict": "ungraded",
        "axes": [{"id": "a", "verdict": "ungraded"}]}})
    out = evaluate_report_card_group(_test("x"), None, ws)
    assert out["result"] == "ungraded"


def test_missing_file_or_group_skips(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"present": {"verdict": "within_tol", "axes": []}})
    out = evaluate_report_card_group(_test("absent"), None, ws)
    assert out["result"] == "ungraded"
    assert "absent" in out.get("detail", "")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest pbg_v2ecoli/tests/test_report_card_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pbg_v2ecoli.evaluators'`.

- [ ] **Step 3: Implement the evaluator**

Create `pbg_v2ecoli/evaluators.py`:

```python
"""Workspace-local study evaluators registered into the pbg-superpowers seam.

register_evaluators(registry) is discovered + called by
pbg_superpowers.study_evaluator.load_workspace_evaluators (mirrors build_core()).
The framework stays report-card-agnostic; all report-card logic lives here.
"""
import json
from pathlib import Path

# A group's outcome = the worst (most severe) axis verdict in that group.
_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def register_evaluators(registry: dict) -> None:
    registry["report_card_axis"] = evaluate_report_card_group


def evaluate_report_card_group(test: dict, reader, ws_root) -> dict:
    """Grade one study test against one group of a report card's verdict JSON.

    measure: {kind: report_card_axis, card: <dir relative to ws_root>, group: <name>}
    Aggregation: any mismatch -> FAIL; any drift (no mismatch) -> PASS + caveat;
    only within_tol -> PASS; all ungraded / missing -> result 'ungraded' (skip).
    """
    measure = test.get("measure") or {}
    card_dir = measure.get("card", "")
    group = measure.get("group", "")
    vpath = Path(ws_root) / card_dir / "report_card_verdict.json"

    if not vpath.is_file():
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"verdict json not found: {vpath}"}
    try:
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"unreadable verdict json: {exc}"}

    g = (verdict.get("groups") or {}).get(group)
    if g is None:
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"group {group!r} absent in card {card_dir!r}"}

    axes = g.get("axes") or []
    verdicts = [a.get("verdict", "ungraded") for a in axes]
    worst = max(verdicts, key=lambda v: _SEVERITY.get(v, 0)) if verdicts else "ungraded"

    provenance = {"card": card_dir, "group": group,
                  "overall": verdict.get("overall"),
                  "axis_verdicts": [{"id": a.get("id"), "verdict": a.get("verdict")}
                                    for a in axes]}

    if worst == "mismatch":
        return {"result": "FAIL", "evaluated_by": "report_card",
                "detail": f"group {group}: mismatch axis present", "provenance": provenance}
    if worst == "drift":
        return {"result": "PASS", "caveat": "drift", "evaluated_by": "report_card",
                "detail": f"group {group}: within tolerance with drift", "provenance": provenance}
    if worst == "within_tol":
        return {"result": "PASS", "evaluated_by": "report_card",
                "detail": f"group {group}: all axes within tolerance", "provenance": provenance}
    return {"result": "ungraded", "evaluated_by": "report_card",
            "detail": f"group {group}: all axes ungraded", "provenance": provenance}
```

Create empty `pbg_v2ecoli/tests/__init__.py`:
```python
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/python -m pytest pbg_v2ecoli/tests/test_report_card_evaluator.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
git add pbg_v2ecoli/evaluators.py pbg_v2ecoli/tests/__init__.py \
        pbg_v2ecoli/tests/test_report_card_evaluator.py
git commit -m "feat(pbg_v2ecoli): report_card_axis evaluator (per-group verdict)

register_evaluators registers a report_card_axis evaluator that reads a card's
report_card_verdict.json and rolls one group's axis verdicts into a study
outcome (mismatch->FAIL, drift->PASS+caveat, within_tol->PASS, ungraded->skip)."
```

---

## Task 5: Wire-up — make the seam see pbg_v2ecoli + reinstall editable

**Files:** none new — this verifies discovery end-to-end.

- [ ] **Step 1: Reinstall the framework editable into the worktree venv**

The worktree venv runs a git-pinned pbg-superpowers; make the Task 1–2 seam live:
```bash
cd /Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large
.venv/bin/python -m pip install -e /Users/eranagmon/code/pbg-superpowers --no-deps
```
Expected: `Successfully installed pbg-superpowers` (editable).

- [ ] **Step 2: Write + run the discovery integration test**

Create `tests/test_workspace_evaluator_discovery.py`:

```python
from pathlib import Path

from pbg_superpowers import study_evaluator as se


def test_load_finds_pbg_v2ecoli_report_card_axis():
    ws_root = Path(__file__).resolve().parents[1]  # the worktree root
    se.clear_workspace_evaluator_cache()
    reg = se.load_workspace_evaluators(ws_root)
    assert "report_card_axis" in reg
```

Run: `.venv/bin/python -m pytest tests/test_workspace_evaluator_discovery.py -v`
Expected: PASS — confirms `load_workspace_evaluators` resolves `pbg_v2ecoli.evaluators` and gets `report_card_axis`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_workspace_evaluator_discovery.py
git commit -m "test: framework seam discovers pbg_v2ecoli.report_card_axis evaluator"
```

---

## Task 6: Migrate showcase-6's 5 tests + sync + verify roll-up

**Files:**
- Modify: `/Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large/workspace/studies/showcase-6-equivalence-large/study.yaml` (the 5 `tests[]`)

- [ ] **Step 1: Rewrite each test's `measure`/`pass_if` to the card kind**

For each of the 5 tests, set `measure` + `pass_if` (keep `name`, `classification`, `question`). Group slugs must match the verdict JSON: `physiology`, `composition`, `ribosomes`, `exchange_fluxes`, `gene_expression`. Example for the first:

```yaml
- name: physiology-equivalent-to-vecoli
  classification: primary
  question: Is v2's physiology group equivalent to vEcoli within tolerance at 16×16?
  measure:
    kind: report_card_axis
    card: docs/report_cards/population_phenotype_basal/vs_vecoli
    group: physiology
  pass_if:
    op: report_card_group_within_tol
```

Repeat with `group: composition`, `group: ribosomes`, `group: exchange_fluxes`, `group: gene_expression`. Leave the authored `status:` lines as-is (they become the reconcile baseline).

- [ ] **Step 2: Run the evaluator over showcase-6 via the public sync path**

The card's `report_card_verdict.json` is already committed (Task 3 Step 6). Run the evaluator directly (no sim needed):

```bash
.venv/bin/python -c "
from pathlib import Path
import yaml
from pbg_superpowers import study_evaluator as se
ws = Path('.').resolve()
se.clear_workspace_evaluator_cache()
spec = yaml.safe_load(open('workspace/studies/showcase-6-equivalence-large/study.yaml'))
out = se.evaluate_study(spec, reader=None, ws_root=ws)
for k, v in out.items():
    print(k, '->', v.get('result') or v.get('evaluated_by'), v.get('evaluated_by'), v.get('caveat',''))
"
```
Expected (matching the rendered card):
```
physiology-equivalent-to-vecoli -> PASS report_card
composition-equivalent-to-vecoli -> PASS report_card
ribosomes-equivalent-to-vecoli -> PASS report_card (caveat: drift)
exchange-fluxes-equivalent-to-vecoli -> FAIL report_card
gene-expression-correlates-vecoli -> FAIL report_card     # transcriptome mismatch -> FAIL
```

> The gene-expression result flips the hand-authored "partial" to FAIL (transcriptome `mismatch` ⇒ FAIL under the aggregation rule). Update the authored `status:`/`outcomes` detail for that test to match, and note the correction in the study `result:` prose.

- [ ] **Step 3: Update showcase-6 authored outcomes to reconcile cleanly**

Edit the test `status:` lines + the run `outcomes` detail so the authored values agree with the card verdicts (so `reconcile: agree` rather than `divergent`). Set `gene-expression-correlates-vecoli` test `status: failed` and adjust its outcome detail to state the transcriptome R² mismatch drives a FAIL under the card's strict-band aggregation.

- [ ] **Step 4: Validate the study still passes the v4 validator**

```bash
.venv/bin/python -c "
import yaml
from vivarium_dashboard.lib.investigations import _validate_study_v4_redesign as v
v(yaml.safe_load(open('workspace/studies/showcase-6-equivalence-large/study.yaml'))); print('VALID')
"
```
Expected: `VALID`.

- [ ] **Step 5: Commit**

```bash
git add workspace/studies/showcase-6-equivalence-large/study.yaml
git commit -m "showcase-6: grade the 5 tests via the report_card_axis evaluator

The five equivalence tests now carry measure.kind: report_card_axis (one per
card group). study_evaluator dispatches them to pbg_v2ecoli's evaluator, which
reads report_card_verdict.json -> computed_outcomes (evaluated_by: report_card)
-> gate -> acceptance criteria. Tests are no longer evaluated_by: agent."
```

---

## Task 7: End-to-end roll-up check

**Files:** none new.

- [ ] **Step 1: Confirm gate + acceptance roll-up consume the card outcomes**

```bash
.venv/bin/python -c "
from pathlib import Path
import yaml
from pbg_superpowers import study_evaluator as se
from pbg_superpowers.study_verdict import roll_up_verdict
ws = Path('.').resolve()
se.clear_workspace_evaluator_cache()
spec = yaml.safe_load(open('workspace/studies/showcase-6-equivalence-large/study.yaml'))
# inject computed_outcomes onto the canonical run for the roll-up
outcomes = se.evaluate_study(spec, reader=None, ws_root=ws)
for r in spec.get('runs', []):
    if r.get('canonical'):
        r['computed_outcomes'] = outcomes
print('verdict:', roll_up_verdict(spec).get('result'))
"
```
Expected: `verdict: failed` (a FAIL test present — exchange-fluxes + gene-expression — gates the study `failed`, which is the honest at-scale equivalence verdict, now machine-derived from the card rather than hand-asserted).

- [ ] **Step 2: Run the full new test set across both repos**

```bash
cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_workspace_evaluators.py -q
cd /Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large && \
  .venv/bin/python -m pytest tests/test_report_card_verdict_json.py \
    tests/test_workspace_evaluator_discovery.py \
    pbg_v2ecoli/tests/test_report_card_evaluator.py -q
```
Expected: all PASS.

- [ ] **Step 3: Final commit (if any doc/status touch-ups remain)**

```bash
cd /Users/eranagmon/code/v2ecoli/.claude/worktrees/showcase-4-equivalence-large
git add -A && git commit -m "chore: finalize report-card evaluator integration" || echo "nothing to commit"
```

---

## Notes for the implementer

- **Two PRs**: pbg-superpowers (Tasks 1–2) is its own PR on `feat/pluggable-workspace-evaluators`. The v2ecoli changes (Tasks 3–7) extend the existing showcase-6 branch (PR #204) or a fresh branch — confirm with the user before opening.
- **Roll-up verdict semantics**: `roll_up_verdict` maps any FAIL → study `failed`; the card-driven `failed` is correct and intended (the at-scale equivalence has a real, localized divergence). The `caveat: drift` on PASS tests is informational; it does not change the gate.
- **`pass_if.op` is advisory** for `report_card_axis` (the evaluator returns a fully-formed outcome); we keep `report_card_group_within_tol` as a readable label. If a future `_op_supported` check rejects unknown ops for card kinds, that check is bypassed because dispatch happens before the op gate.
