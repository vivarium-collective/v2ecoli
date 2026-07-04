# Analysis Flush — Plan 1: registry + extraction + flush (report_card & visualization kinds)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the unified post-sim foundation — one kind-tagged `POST_SIM_REGISTRY`, a reusable `RunExtract`, and a `run_flush` that dispatches **report_card** and **visualization** steps into the owning study's report dir — wired additively into `run_workflow` with zero change to existing analyses.

**Architecture:** New `v2ecoli/workflow/post_sim.py` (unified registry + `Visualization` base) and `v2ecoli/workflow/flush.py` (`RunExtract` extraction/owning-study resolution, per-kind placement sinks, `run_flush` orchestrator). Existing `Analysis`/`AnalysisStep`/`ReportCardStep` also register into the unified registry (back-compat: their own registries are untouched). `run_workflow` additively calls `run_flush` after its existing `run_analyses` call. Analyses are NOT yet routed through the flush (Plan 2).

**Tech Stack:** Python 3.12, process-bigraph Steps (`V2Step`, `bigraph_schema.allocate_core`), the shipped `ReportCardStep` + report-card library, pytest.

## Global Constraints

- **Repo / branch:** worktree `/Users/eranagmon/code/v2e-flush`, branch `feat/analysis-flush` (off `origin/main`). All paths relative to this worktree root.
- **Test command (no venv in the worktree — shadow the main install):** run every test as `PYTHONPATH=$PWD /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <path> -v` from the worktree root. `V2EPY=/Users/eranagmon/code/v2ecoli/.venv/bin/python`. Tests that instantiate Steps use the existing `core` pytest fixture (`tests/conftest.py`); core allocation prints pre-existing "skipping optional dep" warnings — ignore them.
- **Additive / zero-regression:** existing `ANALYSIS_REGISTRY` (`v2ecoli/workflow/analysis.py`) and `REPORT_CARD_REGISTRY` (`v2ecoli/workflow/report_cards/__init__.py`) keep working exactly as now. The unified registry is a parallel, additive index. `run_analyses` is unchanged in this plan.
- **Output port contract:** every post-sim step exposes `outputs() == {"view": "string", "data": "map"}` (HTML + map). The flush reads `view`/`data` from each step's `update(state)`.
- **Placement target:** the owning study's report dir, `<ws_root>/workspace/studies/<slug>/viz/` — report-card outputs to `viz/report_card/<name>.{html,verdict.json}`, visualization views to `viz/<name>.html`. When no owning study is resolvable, placement falls back to `<out_dir>/viz/` (today's location) so nothing regresses.
- **Owning-study resolution:** `config["study"]` (a study slug) if present, else detect `out_dir` (or its parent) lying under a `studies/<slug>/` path, else `None`.
- **Determinism:** committed/written artifacts carry no wall-clock timestamp; JSON written with `allow_nan=False` after sanitizing non-finite floats to `null`.
- **Graceful skip:** one step raising (instantiate/update/placement) logs a skip and the flush continues — never aborts the whole flush.

---

### Task 1: Unified `POST_SIM_REGISTRY` (`v2ecoli/workflow/post_sim.py`)

The kind-tagged registry every post-sim step funnels into, plus the iterate/lookup API the flush uses.

**Files:**
- Create: `v2ecoli/workflow/post_sim.py`
- Test: `tests/test_post_sim_registry.py`

**Interfaces:**
- Produces:
  - `KINDS = ("analysis", "visualization", "report_card")`
  - `POST_SIM_REGISTRY: dict[str, dict]` — `name -> {"cls": type, "kind": str}`.
  - `register_post_sim(cls, kind, name=None) -> None` — register `cls` under `name` (default `cls.name`) with `kind`; no-op if `name` is falsy; raises `ValueError` on an unknown `kind`.
  - `iter_post_sim(kind=None) -> list[tuple[str, type]]` — `[(name, cls), ...]`, optionally filtered to one `kind`, sorted by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_post_sim_registry.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.post_sim import (
    KINDS, POST_SIM_REGISTRY, iter_post_sim, register_post_sim)


class _A: name = "a_demo"
class _V: name = "v_demo"


def test_register_and_iter_by_kind():
    register_post_sim(_A, "analysis")
    register_post_sim(_V, "visualization")
    assert POST_SIM_REGISTRY["a_demo"] == {"cls": _A, "kind": "analysis"}
    names = dict(iter_post_sim())
    assert "a_demo" in names and "v_demo" in names
    assert [n for n, _ in iter_post_sim("visualization")] == ["v_demo"]
    assert [n for n, _ in iter_post_sim("analysis")] == ["a_demo"]


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        register_post_sim(_A, "bogus")


def test_blank_name_is_noop():
    class _N: name = ""
    before = len(POST_SIM_REGISTRY)
    register_post_sim(_N, "analysis")
    assert len(POST_SIM_REGISTRY) == before


def test_kinds_constant():
    assert KINDS == ("analysis", "visualization", "report_card")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_post_sim_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2ecoli.workflow.post_sim'`.

- [ ] **Step 3: Create `v2ecoli/workflow/post_sim.py`**

```python
# v2ecoli/workflow/post_sim.py
"""Unified registry for post-simulation Steps (analyses, visualizations, report
cards). Each registered step is kind-tagged so the analysis flush can discover
and route every post-sim output from one place. Existing per-kind registries
(ANALYSIS_REGISTRY, REPORT_CARD_REGISTRY) remain the canonical homes; this is an
additive parallel index they funnel into."""
from __future__ import annotations

KINDS = ("analysis", "visualization", "report_card")

# name -> {"cls": <Step subclass>, "kind": <one of KINDS>}
POST_SIM_REGISTRY: dict[str, dict] = {}


def register_post_sim(cls, kind: str, name: "str | None" = None) -> None:
    """Register a post-sim Step subclass under ``name`` (default ``cls.name``)
    with its ``kind``. No-op when the resolved name is falsy (abstract bases).
    Raises ValueError for an unknown kind."""
    if kind not in KINDS:
        raise ValueError(f"unknown post-sim kind {kind!r}; expected one of {KINDS}")
    nm = name if name is not None else getattr(cls, "name", "")
    if not nm:
        return
    POST_SIM_REGISTRY[nm] = {"cls": cls, "kind": kind}


def iter_post_sim(kind: "str | None" = None) -> list:
    """[(name, cls), ...] sorted by name, optionally filtered to one kind."""
    out = [(nm, e["cls"]) for nm, e in POST_SIM_REGISTRY.items()
           if kind is None or e["kind"] == kind]
    return sorted(out, key=lambda t: t[0])
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_post_sim_registry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/post_sim.py tests/test_post_sim_registry.py
git commit -m "feat(flush): unified kind-tagged POST_SIM_REGISTRY"
```

---

### Task 2: Funnel `Analysis`/`AnalysisStep` + `ReportCardStep` into the unified registry

Existing steps register into `POST_SIM_REGISTRY` in addition to their own registries — so the flush sees them — without changing current behavior.

**Files:**
- Modify: `v2ecoli/workflow/analysis.py` (`Analysis.__init_subclass__` ~line 56 and `AnalysisStep.__init_subclass__` ~line 105 — add a `register_post_sim` call)
- Modify: `v2ecoli/workflow/report_cards/__init__.py` (`ReportCardStep.__init_subclass__` — add a `register_post_sim` call)
- Test: `tests/test_post_sim_funnel.py`

**Interfaces:**
- Consumes: `register_post_sim` (Task 1); `Analysis`, `AnalysisStep` (`v2ecoli/workflow/analysis.py`); `ReportCardStep` (`v2ecoli/workflow/report_cards`).
- Produces: any concrete `Analysis`/`AnalysisStep` subclass also appears in `POST_SIM_REGISTRY` with `kind="analysis"`; any `ReportCardStep` subclass with `kind="report_card"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_post_sim_funnel.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_report_cards_funnel_into_post_sim():
    import v2ecoli.workflow.report_cards  # noqa: F401 — triggers card registration
    from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY
    assert POST_SIM_REGISTRY.get("tests") == {
        "cls": __import__("v2ecoli.workflow.report_cards.tests_card",
                          fromlist=["TestsCard"]).TestsCard,
        "kind": "report_card"}
    assert POST_SIM_REGISTRY["vs_vecoli"]["kind"] == "report_card"


def test_analysis_funnels_into_post_sim():
    from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
    from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY

    class _ProbeViz(Analysis):
        name = "probe_viz_demo"
        scale = "single"
        def analyze(self, **kw):
            return {"view": "<i></i>", "data": {}}

    assert POST_SIM_REGISTRY["probe_viz_demo"]["kind"] == "analysis"
    # back-compat: the legacy registry still has it too
    assert ANALYSIS_REGISTRY["probe_viz_demo"] is _ProbeViz
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_post_sim_funnel.py -v`
Expected: FAIL — `tests`/`vs_vecoli`/`probe_viz_demo` absent from `POST_SIM_REGISTRY` (funnel not wired yet).

- [ ] **Step 3: Wire the funnels**

In `v2ecoli/workflow/analysis.py`, add an import near the top (after the existing imports):

```python
from v2ecoli.workflow.post_sim import register_post_sim
```

In `Analysis.__init_subclass__`, after the existing `if "name" in cls.__dict__: ANALYSIS_REGISTRY[cls.name] = cls` line, add:

```python
        if "name" in cls.__dict__:
            register_post_sim(cls, "analysis")
```

In `AnalysisStep.__init_subclass__`, after its existing `ANALYSIS_REGISTRY[cls.name] = cls` line, add the same two lines:

```python
        if "name" in cls.__dict__:
            register_post_sim(cls, "analysis")
```

In `v2ecoli/workflow/report_cards/__init__.py`, add to the top imports:

```python
from v2ecoli.workflow.post_sim import register_post_sim
```

and in `ReportCardStep.__init_subclass__`, after the existing `REPORT_CARD_REGISTRY[cls.name] = cls`, add:

```python
        if cls.__dict__.get("name"):
            register_post_sim(cls, "report_card")
```

(Both `analysis.py` and `report_cards/__init__.py` import `post_sim` — `post_sim` imports nothing from either, so there is no import cycle.)

- [ ] **Step 4: Run it to verify it passes**

First fix the test per the Step-1 note (remove the `ANALYSIS_HINT` import). Then:
Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_post_sim_funnel.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the report-card suite to confirm no regression**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_report_card_step.py tests/test_tests_card.py tests/test_vs_vecoli_card.py -q`
Expected: PASS (existing card tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/analysis.py v2ecoli/workflow/report_cards/__init__.py tests/test_post_sim_funnel.py
git commit -m "feat(flush): funnel Analysis + ReportCardStep into POST_SIM_REGISTRY"
```

---

### Task 3: `Visualization` base (the third kind)

A thin `Visualization(V2Step)` mirroring `Analysis`'s `{view,data}` ports, registering with `kind="visualization"`.

**Files:**
- Modify: `v2ecoli/workflow/post_sim.py` (append the `Visualization` base)
- Test: `tests/test_visualization_base.py`

**Interfaces:**
- Consumes: `V2Step` (`v2ecoli.steps.base`); `register_post_sim` (Task 1).
- Produces: `Visualization(V2Step)` — attr `name: str`; `inputs()` overridable (default `{"study": "any"}`); `outputs() -> {"view": "string", "data": "map"}`; `render(study) -> tuple[str, dict] | None` to override; `update(state) -> {"view","data"}`. Subclasses with a `name` register `kind="visualization"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visualization_base.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.steps.base import V2Step
from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY, Visualization


class _DemoViz(Visualization):
    name = "demo_viz"
    def render(self, study):
        return "<div>viz</div>", {"k": 1}


def test_visualization_ports_and_registration(core):
    v = _DemoViz({}, core=core)
    assert isinstance(v, V2Step)
    assert v.outputs() == {"view": "string", "data": "map"}
    assert POST_SIM_REGISTRY["demo_viz"]["kind"] == "visualization"


def test_visualization_update_returns_view_and_data(core):
    out = _DemoViz({}, core=core).update({"study": object()})
    assert out["view"] == "<div>viz</div>"
    assert out["data"] == {"k": 1}


def test_visualization_render_none_yields_empty(core):
    class _Empty(Visualization):
        name = "empty_viz"
        def render(self, study):
            return None
    out = _Empty({}, core=core).update({"study": None})
    assert out == {"view": "", "data": {}}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_visualization_base.py -v`
Expected: FAIL — `ImportError: cannot import name 'Visualization'`.

- [ ] **Step 3: Append `Visualization` to `v2ecoli/workflow/post_sim.py`**

```python
from v2ecoli.steps.base import V2Step  # add to the imports at the top of post_sim.py


class Visualization(V2Step):
    """A post-sim visualization Step: emits a rendered ``view`` (HTML) + ``data``
    (map), like ``Analysis`` but tagged ``kind="visualization"`` so the flush can
    route it distinctly. Subclasses set ``name`` and implement
    ``render(study) -> (html, data) | None``. Inputs default to a StudyContext;
    override ``inputs()`` to consume the run extraction instead."""

    name: str = ""
    config_schema: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            register_post_sim(cls, "visualization")

    def inputs(self):
        return {"study": "any"}

    def outputs(self):
        return {"view": "string", "data": "map"}

    def render(self, study) -> "tuple[str, dict] | None":
        raise NotImplementedError

    def update(self, state, interval=None):
        study = state.get("study")
        res = self.render(study)
        if not res:
            return {"view": "", "data": {}}
        view, data = res
        return {"view": view, "data": data}
```

(Place the `from v2ecoli.steps.base import V2Step` import at the top of the file with the other imports; `register_post_sim` is already defined above in the same module.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_visualization_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/post_sim.py tests/test_visualization_base.py
git commit -m "feat(flush): Visualization base (kind=visualization)"
```

---

### Task 4: `RunExtract` — extraction + owning-study resolution (`v2ecoli/workflow/flush.py`)

The shared context source: lazy run extraction (records + DuckDB conn + sim_data) plus owning-study resolution and the study viz dir. Lazy bits are NOT built unless accessed.

**Files:**
- Create: `v2ecoli/workflow/flush.py`
- Test: `tests/test_run_extract.py`

**Interfaces:**
- Consumes: `build_cell_records`, `_history_from_clause`, `resolve_sim_data`, `resolve_validation_data` (all in `v2ecoli/workflow/analysis_runner.py`); `StudyContext` (`v2ecoli/workflow/report_cards`).
- Produces:
  - `resolve_owning_study(out_dir, config, ws_root) -> str | None` — slug from `config.get("study")` else from an `out_dir` path containing `studies/<slug>/`, else `None`.
  - `RunExtract(out_dir, config, ws_root)` with: `study_slug: str | None`; `study_ctx() -> StudyContext | None`; `study_viz_dir() -> Path | None` (= `ws_root/workspace/studies/<slug>/viz`, or `None`); `records()`, `conn_ctx()` (returns `(conn, from_clause, sim_data, validation_data)`, built once), `context_bag()` (the dict the flush filters by `inputs()`); `close()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_extract.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, resolve_owning_study


def _ws_with_study(tmp_path, slug="demo"):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug, "tests": []}))
    return sd


def test_resolve_owning_study_from_config(tmp_path):
    _ws_with_study(tmp_path, "demo")
    assert resolve_owning_study("out/workflow", {"study": "demo"}, tmp_path) == "demo"


def test_resolve_owning_study_from_out_dir_path(tmp_path):
    sd = _ws_with_study(tmp_path, "demo")
    out = sd / "runs" / "r1"
    out.mkdir(parents=True)
    assert resolve_owning_study(str(out), {}, tmp_path) == "demo"


def test_resolve_owning_study_none(tmp_path):
    assert resolve_owning_study("out/workflow", {}, tmp_path) is None


def test_study_viz_dir_and_ctx(tmp_path):
    _ws_with_study(tmp_path, "demo")
    ex = RunExtract("out/workflow", {"study": "demo"}, tmp_path)
    assert ex.study_slug == "demo"
    assert ex.study_viz_dir() == tmp_path / "workspace" / "studies" / "demo" / "viz"
    assert ex.study_ctx().study_name == "demo"


def test_extraction_is_lazy(tmp_path):
    # No run data on disk; constructing RunExtract + reading study info must NOT
    # touch the (absent) parquet/sim_data. Only conn_ctx()/records() would.
    _ws_with_study(tmp_path, "demo")
    ex = RunExtract("out/workflow", {"study": "demo"}, tmp_path)
    bag = ex.context_bag()              # must not raise though no parquet exists
    assert bag["study"].study_name == "demo"
    assert "conn" in bag                 # present as a lazy handle/None, not built
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2ecoli.workflow.flush'`.

- [ ] **Step 3: Create `v2ecoli/workflow/flush.py` (RunExtract + resolver)**

```python
# v2ecoli/workflow/flush.py
"""The post-simulation analysis flush: extract a finished run once, then dispatch
to the unified POST_SIM_REGISTRY and place each output where the report renders
it. Plan 1 wires the report_card + visualization kinds; analyses keep their
existing run_analyses path (folded in by Plan 2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from v2ecoli.workflow.report_cards import StudyContext

_STUDIES_RE = re.compile(r"(?:^|/)studies/([A-Za-z0-9_.\-]+)(?:/|$)")


def resolve_owning_study(out_dir: str, config: dict, ws_root) -> "str | None":
    """The study slug this run belongs to: config['study'] if set, else a
    studies/<slug>/ segment in out_dir, else None. Only returns a slug whose
    studies/<slug>/study.yaml exists under ws_root."""
    ws_root = Path(ws_root)
    slug = (config or {}).get("study")
    if not slug:
        m = _STUDIES_RE.search(str(out_dir).replace("\\", "/"))
        slug = m.group(1) if m else None
    if not slug:
        return None
    if (ws_root / "workspace" / "studies" / slug / "study.yaml").is_file():
        return slug
    return None


class RunExtract:
    """Lazy extraction context for a finished run. Heavy bits (DuckDB conn +
    sim_data) are provisioned only when conn_ctx()/records() are called."""

    def __init__(self, out_dir: str, config: dict, ws_root):
        self.out_dir = str(out_dir)
        self.config = config or {}
        self.ws_root = Path(ws_root)
        self.study_slug = resolve_owning_study(out_dir, config, ws_root)
        self._ctx: dict[str, Any] = {}
        self._records = None

    def study_ctx(self) -> "StudyContext | None":
        if not self.study_slug:
            return None
        return StudyContext.load(self.ws_root, self.study_slug)

    def study_viz_dir(self) -> "Path | None":
        if not self.study_slug:
            return None
        return self.ws_root / "workspace" / "studies" / self.study_slug / "viz"

    def records(self) -> list:
        if self._records is None:
            from v2ecoli.workflow.analysis_runner import build_cell_records
            self._records = list(build_cell_records(self.out_dir).values())
        return self._records

    def conn_ctx(self) -> tuple:
        if not self._ctx:
            import duckdb
            from v2ecoli.workflow.analysis_runner import (
                _history_from_clause, resolve_sim_data, resolve_validation_data)
            self._ctx["conn"] = duckdb.connect()
            self._ctx["from_clause"] = _history_from_clause(self.out_dir)
            self._ctx["sim_data"] = resolve_sim_data(self.out_dir)
            self._ctx["validation_data"] = resolve_validation_data(self._ctx["sim_data"])
        return (self._ctx["conn"], self._ctx["from_clause"],
                self._ctx["sim_data"], self._ctx["validation_data"])

    def context_bag(self) -> dict:
        """The full provisioning bag the flush filters by each step's inputs().
        `study` is eager (cheap); `conn`/`sim_data`/`history_sql`/`records` are
        lazy CALLABLES so a step that doesn't declare them never triggers the
        heavy extraction."""
        return {
            "study": self.study_ctx(),
            "out_dir": self.out_dir,
            "conn": None,
            "history_sql": "",
            "config_sql": "",
            "success_sql": "",
            "sim_data": None,
            "validation_data": None,
            "_conn_ctx": self.conn_ctx,   # callable: () -> (conn, from_clause, sim_data, validation_data)
            "_records": self.records,     # callable: () -> records
        }

    def close(self) -> None:
        conn = self._ctx.get("conn")
        if conn is not None:
            conn.close()
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_extract.py -v`
Expected: PASS (5 passed). (No parquet/sim_data is touched — the test only reads study info + the bag.)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/flush.py tests/test_run_extract.py
git commit -m "feat(flush): RunExtract — lazy extraction + owning-study resolution"
```

---

### Task 5: Placement sinks (per-kind → study report dir)

Route a step's `(view, data)` output to the owning study's report location by kind, with a no-study fallback to `out_dir/viz/`.

**Files:**
- Modify: `v2ecoli/workflow/flush.py` (append placement helpers)
- Test: `tests/test_flush_placement.py`

**Interfaces:**
- Consumes: `RunExtract` (Task 4); `write_card` (`v2ecoli/workflow/report_cards`, signature `write_card(ctx, name, verdict, html) -> Path`).
- Produces: `place_output(kind, name, view, data, extract) -> str | None` — writes the output to its canonical location and returns the written html path (str) or `None` if nothing was written. `report_card` → `studies/<slug>/viz/report_card/<name>.{html,verdict.json}` via `write_card`; `visualization`/`analysis` → `<study viz>/<name>.html` (or `<out_dir>/viz/<name>.html` when no study).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flush_placement.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import RunExtract, place_output


def _extract(tmp_path, slug="demo", with_study=True, out=None):
    if with_study:
        sd = tmp_path / "workspace" / "studies" / slug
        sd.mkdir(parents=True)
        (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
        return RunExtract(out or "out/x", {"study": slug}, tmp_path)
    return RunExtract(out or str(tmp_path / "out"), {}, tmp_path)


def test_report_card_placed_into_study_viz(tmp_path):
    ex = _extract(tmp_path, "demo")
    p = place_output("report_card", "tests", "<i>card</i>",
                     {"overall": "drift"}, ex)
    base = tmp_path / "workspace" / "studies" / "demo" / "viz" / "report_card"
    assert Path(p) == base / "tests.html"
    assert (base / "tests.html").read_text() == "<i>card</i>"
    assert json.loads((base / "tests.verdict.json").read_text())["overall"] == "drift"


def test_visualization_placed_into_study_viz(tmp_path):
    ex = _extract(tmp_path, "demo")
    p = place_output("visualization", "massfrac", "<div>v</div>", {}, ex)
    assert Path(p) == tmp_path / "workspace" / "studies" / "demo" / "viz" / "massfrac.html"
    assert Path(p).read_text() == "<div>v</div>"


def test_no_study_falls_back_to_out_dir(tmp_path):
    out = tmp_path / "out"
    ex = _extract(tmp_path, with_study=False, out=str(out))
    p = place_output("visualization", "massfrac", "<div>v</div>", {}, ex)
    assert Path(p) == out / "viz" / "massfrac.html"


def test_empty_view_writes_nothing(tmp_path):
    ex = _extract(tmp_path, "demo")
    assert place_output("visualization", "x", "", {}, ex) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_flush_placement.py -v`
Expected: FAIL — `ImportError: cannot import name 'place_output'`.

- [ ] **Step 3: Append placement to `v2ecoli/workflow/flush.py`**

```python
def _write_html(path, html: str):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def place_output(kind: str, name: str, view: str, data: dict,
                 extract: "RunExtract") -> "str | None":
    """Route one step's output to the owning study's report location by kind.
    Returns the written html path (str) or None if nothing was written."""
    if not view:
        return None
    if kind == "report_card":
        from v2ecoli.workflow.report_cards import write_card
        ctx = extract.study_ctx()
        if ctx is not None:
            return str(write_card(ctx, name, data or {}, view))
        # no study: drop the card next to the run so it is not lost
        return str(_write_html(Path(extract.out_dir) / "viz" / "report_card" / f"{name}.html", view))
    # visualization / analysis view
    viz = extract.study_viz_dir() or (Path(extract.out_dir) / "viz")
    return str(_write_html(viz / f"{name}.html", view))
```

(Add `from pathlib import Path` is already imported at the top of `flush.py` from Task 4.)

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_flush_placement.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/flush.py tests/test_flush_placement.py
git commit -m "feat(flush): per-kind placement sinks into the study report dir"
```

---

### Task 6: `run_flush` orchestrator (report_card + visualization kinds)

Dispatch the registered report_card + visualization steps over the extraction, place outputs, write a manifest. Analyses are NOT run here (Plan 2).

**Files:**
- Modify: `v2ecoli/workflow/flush.py` (append `run_flush`)
- Test: `tests/test_run_flush.py`

**Interfaces:**
- Consumes: `iter_post_sim` (Task 1), `POST_SIM_REGISTRY` (Task 1), `RunExtract` (Task 4), `place_output` (Task 5), `bigraph_schema.allocate_core`.
- Produces: `run_flush(out_dir, config, ws_root, *, core=None, kinds=("report_card","visualization")) -> dict` — returns `{"placed": [{"kind","name","path"}], "skipped": [{"name","error"}], "study": slug|None}`. Builds a core once. Filters each step's input bag by its `inputs()`. Graceful per-step skip.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_flush.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import run_flush


def _study(tmp_path, slug="demo", tests=None):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(
        {"name": slug, "tests": tests if tests is not None else [
            {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]}))
    return sd


def test_run_flush_places_report_card(core, tmp_path):
    sd = _study(tmp_path, "demo")
    res = run_flush("out/x", {"study": "demo"}, tmp_path, core=core)
    assert res["study"] == "demo"
    placed = {(p["kind"], p["name"]) for p in res["placed"]}
    assert ("report_card", "tests") in placed
    assert (sd / "viz" / "report_card" / "tests.html").is_file()


def test_run_flush_skips_card_that_raises(core, tmp_path, monkeypatch):
    _study(tmp_path, "demo")
    # Force one card's build to raise; the flush must skip it and still return.
    import v2ecoli.workflow.report_cards.tests_card as tc
    monkeypatch.setattr(tc.TestsCard, "build",
                        lambda self, study: (_ for _ in ()).throw(RuntimeError("boom")))
    res = run_flush("out/x", {"study": "demo"}, tmp_path, core=core)
    assert any(s["name"] == "tests" for s in res["skipped"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_flush.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_flush'`.

- [ ] **Step 3: Append `run_flush` to `v2ecoli/workflow/flush.py`**

```python
def _run_one_step(cls, kind, extract, core):
    """Instantiate + run one post-sim step; return (view, data). For report
    cards we call build()/applies() directly (their native API); visualizations
    and analyses go through update() with an inputs()-filtered bag."""
    step = cls({}, core=core)
    bag = extract.context_bag()
    inputs = {}
    try:
        inputs = step.inputs() or {}
    except Exception:  # noqa: BLE001
        inputs = {}
    # report cards: skip when applies() is False; build() returns (verdict, html)
    if kind == "report_card":
        ctx = bag.get("study")
        if ctx is None or not step.applies(ctx):
            return "", {}
        res = step.build(ctx)
        if not res:
            return "", {}
        verdict, html = res
        return html, verdict
    # analyses/visualizations declaring DuckDB inputs get the lazy conn ctx
    state = {}
    for key in inputs:
        if key in ("conn", "history_sql", "sim_data", "validation_data") and bag.get("conn") is None:
            conn, from_clause, sim_data, validation_data = bag["_conn_ctx"]()
            state.update({"conn": conn, "history_sql": from_clause,
                          "sim_data": sim_data, "validation_data": validation_data})
        elif key in bag:
            state[key] = bag[key]
    out = step.update(state) or {}
    return out.get("view", ""), out.get("data", {}) or {}


def run_flush(out_dir, config, ws_root, *, core=None,
              kinds=("report_card", "visualization")) -> dict:
    """Dispatch the registered post-sim steps of the given kinds over a finished
    run and place each output where the study report renders it. Plan 1 omits
    the 'analysis' kind (Plan 2 folds it in). Graceful per-step skip."""
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.post_sim import iter_post_sim
    if core is None:
        core = allocate_core()
    extract = RunExtract(out_dir, config, ws_root)
    placed, skipped = [], []
    try:
        for kind in kinds:
            for name, cls in iter_post_sim(kind):
                try:
                    view, data = _run_one_step(cls, kind, extract, core)
                except Exception as e:  # noqa: BLE001 — one step never aborts the flush
                    skipped.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                    continue
                path = place_output(kind, name, view, data, extract)
                if path:
                    placed.append({"kind": kind, "name": name, "path": path})
    finally:
        extract.close()
    return {"placed": placed, "skipped": skipped, "study": extract.study_slug}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_flush.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/flush.py tests/test_run_flush.py
git commit -m "feat(flush): run_flush orchestrator (report_card + visualization kinds)"
```

---

### Task 7: Wire `run_flush` into `run_workflow` (additive)

After the existing post-run `run_analyses` call, additively run the flush so report-card + visualization outputs land in the study report — without disturbing analyses.

**Files:**
- Modify: `v2ecoli/workflow/run.py` (both post-run sites: ~line 131-135 and ~line 204-206)
- Test: `tests/test_run_workflow_flush_hook.py`

**Interfaces:**
- Consumes: `run_flush` (Task 6).
- Produces: `run_workflow` result dict gains `result["flush"] = {placed, skipped, study}` when a study is resolvable; absent/`None` otherwise. The existing `result["analysis"]` behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_workflow_flush_hook.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _study(tmp_path, slug="demo"):
    import yaml
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    return sd


def test_run_workflow_calls_flush_when_study_resolvable(monkeypatch, tmp_path):
    import v2ecoli.workflow.run as run_mod
    _study(tmp_path, "demo")  # resolve_owning_study runs for real — needs the study

    calls = {}
    def _fake_flush(out_dir, config, ws_root, **kw):
        calls["study"] = config.get("study")
        return {"placed": [{"kind": "report_card", "name": "tests", "path": "p"}],
                "skipped": [], "study": config.get("study")}
    # _maybe_flush does `from v2ecoli.workflow.flush import run_flush` internally,
    # so patch run_flush on the flush module (where the local import resolves it).
    import v2ecoli.workflow.flush as flush_mod
    monkeypatch.setattr(flush_mod, "run_flush", _fake_flush, raising=False)

    cfg = {"study": "demo", "out_dir": "out/x", "ws_root": str(tmp_path)}
    res = run_mod._maybe_flush(cfg, "out/x", {"complete": True})
    assert res["flush"]["study"] == "demo"
    assert calls["study"] == "demo"


def test_maybe_flush_noop_without_study(tmp_path):
    import v2ecoli.workflow.run as run_mod
    res = run_mod._maybe_flush({"out_dir": "out/workflow", "ws_root": str(tmp_path)},
                               "out/workflow", {"complete": True})
    assert "flush" not in res
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_workflow_flush_hook.py -v`
Expected: FAIL — `AttributeError: module 'v2ecoli.workflow.run' has no attribute '_maybe_flush'`.

- [ ] **Step 3: Add `_maybe_flush` and call it in `run_workflow`**

Add this helper to `v2ecoli/workflow/run.py` (near the other module-level helpers):

```python
def _maybe_flush(config: dict, out_dir: str, result: dict) -> dict:
    """Additively run the post-sim flush (report_card + visualization kinds) when
    an owning study is resolvable, attaching result['flush']. Never raises: a
    flush failure must not fail the run."""
    import os
    from v2ecoli.workflow.flush import resolve_owning_study, run_flush
    ws_root = config.get("ws_root") or os.getcwd()
    if resolve_owning_study(out_dir, config, ws_root) is None:
        return result
    try:
        result["flush"] = run_flush(out_dir, config, ws_root)
    except Exception as e:  # noqa: BLE001
        result["flush"] = {"placed": [], "skipped": [], "error": f"{type(e).__name__}: {e}"}
    return result
```

Then, in `run_workflow`, immediately BEFORE `return result` (after the `analysis_options`/`run_analyses` block, ~line 135), insert:

```python
    result = _maybe_flush(config, out_dir, result)
```

Apply the same one-line insertion before the `return` of the other sweep path (~line 206, after its `run_analyses` block).

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_run_workflow_flush_hook.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full new flush suite together**

Run: `PYTHONPATH=$PWD $V2EPY -m pytest tests/test_post_sim_registry.py tests/test_post_sim_funnel.py tests/test_visualization_base.py tests/test_run_extract.py tests/test_flush_placement.py tests/test_run_flush.py tests/test_run_workflow_flush_hook.py -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/run.py tests/test_run_workflow_flush_hook.py
git commit -m "feat(flush): run_workflow additively runs the post-sim flush"
```

---

## Out of scope (later plans)

- **Plan 2 — fold analyses into the flush:** route the `analysis` kind through `run_flush` (delegating the scale/group machinery via a placement-aware `run_analyses`), so analyses also land in `studies/<slug>/viz/`; retire the separate post-run `run_analyses` call once at parity.
- **Plan 3 — absorb the runner + CLI + parity:** make `scripts/study_report_cards.py` a thin wrapper over `run_flush(kinds=("report_card",))`; add a standalone re-flush CLI; parity-test the flushed analysis outputs against the legacy `run_analyses` outputs.
- Dashboard-side rendering: unchanged — it already reads `studies/<slug>/viz/*.html` + `viz/report_card/*`.
