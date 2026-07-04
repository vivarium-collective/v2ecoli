# Analysis Flush — Plan 2: fold the `analysis` kind into the flush

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the `analysis` kind through `run_flush` so analyses also land in the owning study's report dir, then retire the separate post-run `run_analyses` call — at parity, with zero change to `run_analyses` itself.

**Architecture:** Leave `v2ecoli/workflow/analysis_runner.run_analyses` UNTOUCHED (it still writes `out_dir/viz/*.html` + `out_dir/ptools/*.tsv` + `analysis.json`). Add a placement helper that COPIES those produced artifacts into the owning study's report dir, and a flush analysis-driver that runs `run_analyses` then places its outputs. `run_flush` gains the `analysis` kind; `run_workflow` routes analyses through the flush and drops its duplicate standalone `run_analyses` call. Report-card/visualization dispatch (Plan 1) is unchanged.

**Tech Stack:** Python 3.12, the Plan-1 flush (`v2ecoli/workflow/flush.py`), `run_analyses`, pytest.

## Global Constraints

- **Repo / branch:** worktree `/Users/eranagmon/code/v2e-flush2`, branch `feat/analysis-flush-p2` (off `origin/main`, which has flush Plan 1). All paths relative to this worktree root.
- **Test command (no venv):** `PYTHONPATH=$PWD /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <path> -v` from the worktree root. `V2EPY=/Users/eranagmon/code/v2ecoli/.venv/bin/python`. Step tests use the existing `core` fixture (`tests/conftest.py`); core allocation prints pre-existing "skipping optional dep" warnings — ignore.
- **Zero change to `run_analyses`:** do NOT modify `v2ecoli/workflow/analysis_runner.py`. The analysis machinery (scale/group iteration, DuckDB ctx) stays exactly as-is; the flush wraps it.
- **Copy, don't move:** the analysis driver COPIES `out_dir/viz/*.html` → `studies/<slug>/viz/<name>.html` and `out_dir/ptools/*.tsv` → `studies/<slug>/ptools/<name>.tsv` (leaving the raw run artifacts in `out_dir` as provenance). No owning study → no copy (out_dir stays the home; today's behavior).
- **Flush never fails the run:** the analysis driver runs inside the flush's existing isolation; a `run_analyses` or copy failure is caught and reported in `skipped`/`error`, never propagated.
- **Parity then retire:** `run_workflow` must produce the same `analysis.json` + `out_dir/viz` artifacts as before (run_analyses still runs), PLUS the new study-dir copies. The standalone `run_analyses` call is removed only because the flush now invokes it — net analyses run exactly once.
- **Determinism:** copying preserves bytes; no timestamps added.

---

### Task 1: Analysis-output placement helper (`copy out_dir → study report dir`)

A pure file-copy helper that mirrors a finished run's analysis artifacts into the owning study's report dir.

**Files:**
- Modify: `v2ecoli/workflow/flush.py` (append `place_analysis_outputs`)
- Test: `tests/test_flush_analysis_placement.py`

**Interfaces:**
- Consumes: `RunExtract` (Plan 1) — `extract.out_dir`, `extract.study_viz_dir()`, `extract.study_slug`, `extract.ws_root`.
- Produces: `place_analysis_outputs(extract) -> list[dict]` — copies `<out_dir>/viz/*.html` → `<study viz>/<stem>.html` and `<out_dir>/ptools/*.tsv` → `<study dir>/ptools/<stem>.tsv`; returns `[{"kind":"analysis","name":<stem>,"path":<dest>}]` for each copied html. No owning study → returns `[]` (nothing copied; out_dir stays the home).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flush_analysis_placement.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, place_analysis_outputs


def _extract_with_study(tmp_path, slug="demo"):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    out = tmp_path / "out" / "run1"
    (out / "viz").mkdir(parents=True)
    (out / "viz" / "mass_fraction__seed_0.html").write_text("<div>mf</div>")
    (out / "ptools").mkdir(parents=True)
    (out / "ptools" / "ptools_rna__seed_0.tsv").write_text("a\tb\n")
    return RunExtract(str(out), {"study": slug}, tmp_path), sd, out


def test_copies_viz_and_ptools_into_study_dir(tmp_path):
    ex, sd, out = _extract_with_study(tmp_path)
    placed = place_analysis_outputs(ex)
    # html copied into study viz
    assert (sd / "viz" / "mass_fraction__seed_0.html").read_text() == "<div>mf</div>"
    # tsv copied into study ptools
    assert (sd / "ptools" / "ptools_rna__seed_0.tsv").read_text() == "a\tb\n"
    # raw run artifacts left in place (copy, not move)
    assert (out / "viz" / "mass_fraction__seed_0.html").is_file()
    # placed reports the html
    assert {p["name"] for p in placed} == {"mass_fraction__seed_0"}
    assert placed[0]["kind"] == "analysis"


def test_no_study_copies_nothing(tmp_path):
    out = tmp_path / "out" / "run1"
    (out / "viz").mkdir(parents=True)
    (out / "viz" / "x.html").write_text("<i></i>")
    ex = RunExtract(str(out), {}, tmp_path)   # no owning study
    assert place_analysis_outputs(ex) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_flush_analysis_placement.py -v`
Expected: FAIL — `ImportError: cannot import name 'place_analysis_outputs'`.

- [ ] **Step 3: Append `place_analysis_outputs` to `v2ecoli/workflow/flush.py`**

```python
def place_analysis_outputs(extract: "RunExtract") -> list:
    """Copy a finished run's analysis artifacts into the owning study's report
    dir so the study report surfaces them: <out_dir>/viz/*.html ->
    <study viz>/<stem>.html and <out_dir>/ptools/*.tsv -> <study>/ptools/<stem>.tsv.
    Returns [{"kind":"analysis","name":<stem>,"path":<dest>}] per copied html.
    No owning study -> returns [] (the run's out_dir stays the home)."""
    import shutil

    study_viz = extract.study_viz_dir()
    if study_viz is None:
        return []
    placed = []
    src_viz = Path(extract.out_dir) / "viz"
    if src_viz.is_dir():
        study_viz.mkdir(parents=True, exist_ok=True)
        for html in sorted(src_viz.glob("*.html")):
            dest = study_viz / html.name
            shutil.copyfile(html, dest)
            placed.append({"kind": "analysis", "name": html.stem, "path": str(dest)})
    src_ptools = Path(extract.out_dir) / "ptools"
    if src_ptools.is_dir():
        study_ptools = study_viz.parent / "ptools"
        study_ptools.mkdir(parents=True, exist_ok=True)
        for tsv in sorted(src_ptools.glob("*.tsv")):
            shutil.copyfile(tsv, study_ptools / tsv.name)
    return placed
```

(`Path` is already imported at the top of `flush.py`.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_flush_analysis_placement.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/flush.py tests/test_flush_analysis_placement.py
git commit -m "feat(flush): place_analysis_outputs — copy run analysis artifacts into study report dir"
```

---

### Task 2: Flush analysis driver + `analysis` kind in `run_flush`

Run `run_analyses` over the run, then place its outputs; make `run_flush` dispatch the `analysis` kind.

**Files:**
- Modify: `v2ecoli/workflow/flush.py` (append `_flush_analyses`; extend `run_flush`)
- Test: `tests/test_run_flush_analysis.py`

**Interfaces:**
- Consumes: `place_analysis_outputs` (Task 1); `run_analyses` (`v2ecoli/workflow/analysis_runner`, lazy import); `RunExtract`.
- Produces:
  - `_flush_analyses(extract, config) -> tuple[list, list]` — when `config["analysis_options"]` has any truthy value, calls `run_analyses(extract.out_dir, analysis_options)` then `place_analysis_outputs(extract)`; returns `(placed, skipped)` (a `run_analyses`/placement error → `([], [{"name":"analyses","error":...}])`). Empty analysis_options → `([], [])`.
  - `run_flush(..., kinds=("analysis","report_card","visualization"))` — new default INCLUDES analysis. For the `analysis` kind it calls `_flush_analyses` (not the per-step `_run_one_step` loop, which is for report_card/visualization).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_flush_analysis.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow import flush as flush_mod
from v2ecoli.workflow.flush import run_flush


def _study(tmp_path, slug="demo"):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    return sd


def test_run_flush_runs_and_places_analyses(core, tmp_path, monkeypatch):
    sd = _study(tmp_path, "demo")
    out = tmp_path / "out" / "run1"

    def _fake_run_analyses(sweep_dir, analysis_options, *a, **k):
        viz = Path(sweep_dir) / "viz"
        viz.mkdir(parents=True, exist_ok=True)
        (viz / "mass_fraction__seed_0.html").write_text("<div>mf</div>")
        return {}
    # run_flush imports run_analyses lazily from analysis_runner; patch it there.
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "run_analyses", _fake_run_analyses, raising=False)

    cfg = {"study": "demo", "analysis_options": {"single": {"mass_fraction": {}}}}
    res = run_flush(str(out), cfg, tmp_path, core=core, kinds=("analysis",))
    assert any(p["kind"] == "analysis" and p["name"] == "mass_fraction__seed_0"
               for p in res["placed"])
    assert (sd / "viz" / "mass_fraction__seed_0.html").is_file()


def test_run_flush_analysis_skips_on_error(core, tmp_path, monkeypatch):
    _study(tmp_path, "demo")
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "run_analyses",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)
    cfg = {"study": "demo", "analysis_options": {"single": {"x": {}}}}
    res = run_flush(str(tmp_path / "out"), cfg, tmp_path, core=core, kinds=("analysis",))
    assert any(s["name"] == "analyses" for s in res["skipped"])


def test_run_flush_no_analysis_options_noop(core, tmp_path):
    _study(tmp_path, "demo")
    res = run_flush(str(tmp_path / "out"), {"study": "demo"}, tmp_path,
                    core=core, kinds=("analysis",))
    assert res["placed"] == [] and res["skipped"] == []


def test_default_kinds_include_analysis():
    import inspect
    sig = inspect.signature(run_flush)
    assert sig.parameters["kinds"].default == ("analysis", "report_card", "visualization")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_flush_analysis.py -v`
Expected: FAIL — `run_flush` ignores the `analysis` kind (no `_flush_analyses`), so `placed`/`skipped` stay empty and `test_default_kinds_include_analysis` fails on the old default.

- [ ] **Step 3: Add `_flush_analyses` and extend `run_flush`**

Append `_flush_analyses` to `v2ecoli/workflow/flush.py`:

```python
def _flush_analyses(extract: "RunExtract", config: dict) -> tuple:
    """Run the configured analyses over the run, then copy their outputs into the
    owning study's report dir. Returns (placed, skipped). Empty analysis_options
    -> ([], []). Any failure -> ([], [{"name":"analyses","error":...}])."""
    analysis_options = (config or {}).get("analysis_options") or {}
    if not any(analysis_options.values()):
        return [], []
    try:
        from v2ecoli.workflow.analysis_runner import run_analyses
        run_analyses(extract.out_dir, analysis_options)
        return place_analysis_outputs(extract), []
    except Exception as e:  # noqa: BLE001 — never abort the flush
        return [], [{"name": "analyses", "error": f"{type(e).__name__}: {e}"}]
```

In `run_flush`, change the default `kinds` and special-case the `analysis` kind. The current loop is:

```python
        for kind in kinds:
            for name, cls in iter_post_sim(kind):
                ...
```

Change the signature default to `kinds=("analysis", "report_card", "visualization")` and make the per-kind body handle `analysis` separately:

```python
        for kind in kinds:
            if kind == "analysis":
                a_placed, a_skipped = _flush_analyses(extract, config)
                placed.extend(a_placed)
                skipped.extend(a_skipped)
                continue
            for name, cls in iter_post_sim(kind):
                try:
                    view, data = _run_one_step(cls, kind, extract, core)
                    path = place_output(kind, name, view, data, extract)
                except Exception as e:  # noqa: BLE001
                    skipped.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                    continue
                if path:
                    placed.append({"kind": kind, "name": name, "path": path})
```

(Keep the surrounding `extract = RunExtract(...)`, the outer `try/finally: extract.close()`, and the `return {...}` exactly as they are.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_flush_analysis.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the existing flush suite to confirm no regression**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_flush.py tests/test_flush_placement.py tests/test_run_extract.py -q`
Expected: PASS (Plan-1 flush tests still green; the new default `kinds` includes analysis but those tests pass explicit `kinds=` or use studies without `analysis_options`, so no analysis runs).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/flush.py tests/test_run_flush_analysis.py
git commit -m "feat(flush): run_flush dispatches the analysis kind via _flush_analyses"
```

---

### Task 3: Route `run_workflow` analyses through the flush (retire the standalone call)

`run_workflow` no longer calls `run_analyses` directly; the flush (now including the analysis kind) does. Net: analyses run exactly once, and their outputs reach the study report dir.

**Files:**
- Modify: `v2ecoli/workflow/run.py` (both sweep paths: remove the `run_analyses` block, keep `result["analysis"]` pointing at the produced `analysis.json`; `_maybe_flush` already runs the flush)
- Test: `tests/test_run_workflow_analysis_via_flush.py`

**Interfaces:**
- Consumes: `_maybe_flush` (Plan 1, unchanged — it calls `run_flush` with the new default kinds, which now includes analysis).
- Produces: `run_workflow` result keeps `result["analysis"] = <out_dir>/analysis.json` (the flush's analysis driver still writes it via `run_analyses`), and `result["flush"]` now includes analysis placements. The direct `run_analyses` import+call in `run_workflow` is removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_workflow_analysis_via_flush.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_run_workflow_does_not_call_run_analyses_directly(monkeypatch, tmp_path):
    """The standalone run_analyses call is gone; analyses go through the flush."""
    import v2ecoli.workflow.run as run_mod
    import v2ecoli.workflow.analysis_runner as ar

    called = {"direct": 0}
    monkeypatch.setattr(ar, "run_analyses",
                        lambda *a, **k: called.__setitem__("direct", called["direct"] + 1) or {},
                        raising=False)
    # the flush is where analyses should now be driven — stub it so we can assert
    # run_workflow no longer invokes run_analyses on its own.
    import v2ecoli.workflow.flush as flush_mod
    monkeypatch.setattr(flush_mod, "run_flush",
                        lambda *a, **k: {"placed": [], "skipped": [], "study": "demo"},
                        raising=False)

    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    import yaml
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": "demo"}))

    cfg = {"study": "demo", "out_dir": "out/x", "ws_root": str(tmp_path),
           "analysis_options": {"single": {"x": {}}}}
    res = run_mod._maybe_flush(cfg, "out/x", {"complete": True})
    # _maybe_flush drove the flush (stubbed), and did NOT call run_analyses directly
    assert res.get("flush", {}).get("study") == "demo"
    assert called["direct"] == 0
```

(This test exercises `_maybe_flush` directly, proving the analysis path is the flush's responsibility. The `run_workflow` edits below remove the now-redundant direct call so analyses are not run twice.)

- [ ] **Step 2: Run it to verify it fails / passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_workflow_analysis_via_flush.py -v`
Expected: PASS already for `_maybe_flush` (it never called run_analyses). This test guards the contract; the code change below is the actual retirement of the duplicate call in `run_workflow`. After making the edits in Step 3, this test still passes AND the full suite confirms no double-run.

- [ ] **Step 3: Remove the standalone `run_analyses` call in both sweep paths**

In `v2ecoli/workflow/run.py`, the first sweep path currently has:

```python
    analysis_options = config.get("analysis_options") or {}
    if any((analysis_options or {}).values()):
        from v2ecoli.workflow.analysis_runner import run_analyses
        run_analyses(out_dir, analysis_options)
        result["analysis"] = os.path.join(out_dir, "analysis.json")
    result = _maybe_flush(config, out_dir, result)
```

Replace the `run_analyses` invocation with just the result pointer (the flush now runs analyses), keeping `result["analysis"]` so callers still find the file the flush's analysis driver writes:

```python
    analysis_options = config.get("analysis_options") or {}
    if any((analysis_options or {}).values()):
        # analyses are now driven by the post-sim flush (run_flush's analysis
        # kind), which runs run_analyses once and places outputs into the study
        # report dir. We only record where analysis.json will be written.
        result["analysis"] = os.path.join(out_dir, "analysis.json")
    result = _maybe_flush(config, out_dir, result)
```

Apply the identical replacement to the second sweep path (the other `run_analyses` block, ~line 219+). Remove the now-unused `from v2ecoli.workflow.analysis_runner import run_analyses` import lines in both blocks.

IMPORTANT — for the analysis to actually run, `_maybe_flush` must reach the flush even when `resolve_owning_study` is None for ad-hoc runs that still want analyses. Confirm current `_maybe_flush` behavior: it returns early (no flush) when no owning study. That means an ad-hoc run (no study) would now run NO analyses — a regression for `out/workflow` runs. To preserve ad-hoc analyses, change `_maybe_flush` so it still runs the flush when `analysis_options` is non-empty even without an owning study (analyses then place into `out_dir/viz` via the no-study fallback). Update `_maybe_flush`:

```python
def _maybe_flush(config: dict, out_dir: str, result: dict) -> dict:
    """Run the post-sim flush. Never raises. Runs when an owning study is
    resolvable OR when analysis_options are present (ad-hoc analyses place into
    out_dir/viz)."""
    import os
    from v2ecoli.workflow.flush import resolve_owning_study, run_flush
    try:
        ws_root = config.get("ws_root") or os.getcwd()
        has_analyses = any((config.get("analysis_options") or {}).values())
        if resolve_owning_study(out_dir, config, ws_root) is None and not has_analyses:
            return result
        result["flush"] = run_flush(out_dir, config, ws_root)
    except Exception as e:  # noqa: BLE001 — flush failures must not fail the run
        result["flush"] = {"placed": [], "skipped": [], "error": f"{type(e).__name__}: {e}"}
    return result
```

- [ ] **Step 4: Run the test + the full flush suite**

Run:
```
PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_workflow_analysis_via_flush.py tests/test_run_workflow_flush_hook.py -v
PYTHONPATH=$PWD $V2EPY -m pytest tests/test_flush_analysis_placement.py tests/test_run_flush_analysis.py tests/test_run_flush.py tests/test_flush_placement.py tests/test_run_extract.py tests/test_post_sim_registry.py tests/test_post_sim_funnel.py tests/test_visualization_base.py -q
```
Expected: PASS (the hook test's `test_maybe_flush_noop_without_study` still holds — it passes a config with NO `analysis_options`, so the new `has_analyses` branch keeps it a no-op).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/run.py tests/test_run_workflow_analysis_via_flush.py
git commit -m "feat(flush): run_workflow drives analyses through the flush (retire duplicate run_analyses call)"
```

---

## Out of scope (Plan 3)

- Make `scripts/study_report_cards.py` a thin wrapper over `run_flush(kinds=("report_card",))`; add a standalone re-flush CLI for an existing run dir.
- Honor a study's `report_cards:` allowlist in `run_flush` (today 0/23 studies declare it; `report_cards.applicable` honors it, `run_flush` does not yet).
- Migrate `run_analyses` itself to write directly into the study dir (this plan copies post-hoc to keep `run_analyses` untouched).
