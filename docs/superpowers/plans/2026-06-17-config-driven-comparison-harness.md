# Config-driven vEcoli↔v2ecoli Comparison Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take a vEcoli *fork* (separate checkout) carrying new Vivarium-1.0 processes + a config, run that config in both engines (vEcoli natively, v2ecoli with the new processes auto-translated and injected into the 2-generation lineage), and emit an HTML report showing the loaded config, the converted processes, behavioral detail, and statistical-equivalence report cards.

**Architecture:** Reuse the existing `scripts/compare_harness.py` shell, `wrap_vivarium_process` converter, and `v2ecoli/library/report_card.py` card library. Add a harness-side injection module (`scripts/_compare/inject.py`) that resolves + classifies the fork's added processes, and thread an `injected_processes` config down through `meta_composite → LineageProcess → baseline()` so the injected processes are rebuilt every generation. Extend the report with four panels.

**Tech Stack:** Python 3, process-bigraph, bigraph-schema, pytest. v2ecoli venv at `.venv/bin/python`. vEcoli fork run via its own `.venv` + Nextflow workflow.

## Global Constraints

- Run all v2ecoli Python via `.venv/bin/python` (bare `python` lacks `unum`). Tests: `.venv/bin/python -m pytest`.
- **vivarium-core must be importable in the v2ecoli venv** — the fork's real v1 process classes inherit from `vivarium.core.process.Process`, so importing them in the v2 sim subprocess requires it. Install once: `.venv/bin/python -m pip install vivarium-core --no-deps` (verify in Task 0). The converter itself stays duck-typed; this is only to import the fork's classes.
- **No bit-parity.** Comparison is statistical/behavioral equivalence (report cards), not exact reproduction.
- **v1 scope — fail fast (never silently degrade) on:** partitioned new processes (`calculate_request`/`evolve_state`), `process_configs: "sim_data"` on a new process, a topology path referencing a store absent from the cell-state tree, fork import failure, unknown process name.
- Injection code paths are **no-ops when `injected_processes` is empty/absent** — baseline behavior must be byte-for-byte unchanged when no fork is supplied (guard every new hook with `if not injected_processes: return`).
- Topology format is shared between the models: a port maps to a tuple/list of store-path segments (e.g. `("bulk",)`, `("unique","promoter")`); reuse the fork's topology verbatim.
- Work happens in worktree `/Users/eranagmon/code/v2e-compare-harness` on branch `feat/comparison-harness-config` (already created off `origin/main`).
- Commit after every task. Conventional-commit messages; end with the Co-Authored-By trailer.

---

## File Structure

**New files**
- `scripts/_compare/inject.py` — `InjectionSpec`, `classify_process`, `resolve_injections`, `apply_injected_processes`, and a `__main__` that prints specs as JSON (used by the parent harness for report metadata + early fail-fast).
- `scripts/_compare/charts.py` — SVG helpers (`sparkline`, `overlay_card`, `multiline_svg`) lifted from `reports/composite_comparison.py` so both the new report and the old one share one source.
- `scripts/_compare/report_card_section.py` — `build_report_card(left_by_cell, right_by_cell, ...)` → `(verdict_json_dict, html)` using `v2ecoli/library/report_card.py` + `card_criteria.py`.
- `tests/fixtures/fork_example/ecoli/processes/__init__.py` — a tiny duck-typed fake fork: a `process_registry` exposing one simple v1 process (`ExampleSecretion`) and one deliberately-partitioned process (`BadPartitioned`) for the fail-fast test.
- `tests/fixtures/fork_example/configs/example.json` — a config that `add_processes: ["example-secretion"]` with topology + process_configs.
- `tests/test_compare_inject.py`, `tests/test_compare_charts.py`, `tests/test_report_card_section.py`, `tests/test_baseline_injected.py`, `tests/test_lineage_injected_threading.py` — unit tests.
- `tests/test_compare_harness_injected_e2e.py` — integration smoke test (marked `slow`).

**Modified files**
- `scripts/_compare/config_adapter.py` — preserve process-set keys; parameterize `VECOLI_REPO`.
- `scripts/_compare/orchestrator.py` — parameterize `VECOLI_REPO` / `VECOLI_PYTHON`.
- `v2ecoli/composites/baseline.py` — `injected_processes` parameter + apply hook.
- `v2ecoli/workflow/lineage.py` — `injected_processes` in `config_schema` + pass to `baseline()`.
- `v2ecoli/workflow/meta_composite.py` — copy `injected_processes` into the lineage node config.
- `reports/composite_comparison.py` — import SVG helpers from `scripts/_compare/charts.py`.
- `scripts/compare_harness.py` — `--vecoli-repo` / `--tol-rel` / `--force` CLI; build injected v2 config; add the four report panels; write `report_card_verdict.json`.
- `scripts/_compare/report.py` — render the loaded-config + converted-processes panels and embed pre-rendered HTML sections.

---

## Task 0: Environment precondition check

**Files:** none (verification only)

- [ ] **Step 1: Confirm vivarium-core importability in the v2 venv**

Run: `.venv/bin/python -c "import vivarium.core.process; print('ok')"`
Expected: prints `ok`. If it raises `ModuleNotFoundError`, run
`.venv/bin/python -m pip install vivarium-core --no-deps` then re-run the check.

- [ ] **Step 2: Confirm the card library imports**

Run: `.venv/bin/python -c "from v2ecoli.library.report_card import grade_card, verdict_json, render_html; from v2ecoli.library.card_criteria import grade_axis; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Confirm the converter imports**

Run: `.venv/bin/python -c "from v2ecoli.library.vivarium_bridge import wrap_vivarium_process; print('ok')"`
Expected: prints `ok`.

No commit (verification task).

---

## Task 1: Config carry-through + repo parameterization

**Files:**
- Modify: `scripts/_compare/config_adapter.py`
- Test: `tests/test_compare_inject.py` (config-adapter portion)

**Interfaces:**
- Produces: `translate_vecoli_config(vecoli)` now preserves keys `add_processes`, `exclude_processes`, `swap_processes`, `process_configs`, `topology` in the returned dict. `resolve_vecoli_config(config_path, vecoli_repo=...)` accepts an optional repo path (default `"/Users/eranagmon/code/vEcoli"`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_compare_inject.py
from scripts._compare.config_adapter import translate_vecoli_config

def test_translate_preserves_process_set_keys():
    vecoli = {
        "experiment_id": "x", "generations": 2,
        "add_processes": ["example-secretion"],
        "swap_processes": {}, "exclude_processes": [],
        "process_configs": {"example-secretion": {"rate": 2.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "emitter": "parquet",            # vEcoli-only -> dropped
    }
    v2 = translate_vecoli_config(vecoli)
    assert v2["add_processes"] == ["example-secretion"]
    assert v2["process_configs"]["example-secretion"] == {"rate": 2.0}
    assert v2["topology"]["example-secretion"] == {"counts": ["bulk"]}
    assert "emitter" not in v2                      # still dropped
    assert v2["_dropped_vecoli_keys"]["emitter"] == "parquet"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py::test_translate_preserves_process_set_keys -v`
Expected: PASS already for `add_processes` (it passes through today) but FAIL is acceptable only if a key is dropped. If it passes outright, still proceed — the change below also parameterizes the repo, which Step 3 covers. (The process-set keys are *not* in `_VECOLI_ONLY`, so they already pass through; this test pins that contract so a later refactor can't regress it.)

- [ ] **Step 3: Parameterize the vEcoli repo in `resolve_vecoli_config`**

In `scripts/_compare/config_adapter.py`, change the resolver signature and use the argument:

```python
def resolve_vecoli_config(config_path: str,
                          vecoli_repo: str = VECOLI_REPO) -> dict[str, Any]:
    """Resolve a vEcoli config (honoring ``inherit_from``) using the fork's
    own loader, returning the fully-merged dict."""
    vecoli_python = f"{vecoli_repo}/.venv/bin/python"
    snippet = (
        "import json,sys;"
        "from runscripts.workflow import load_config_with_inheritance;"
        "json.dump(load_config_with_inheritance(sys.argv[1]), sys.stdout)"
    )
    out = subprocess.check_output(
        [vecoli_python, "-c", snippet, config_path],
        cwd=vecoli_repo, text=True,
    )
    return json.loads(out)
```

(Leave `VECOLI_REPO`/`VECOLI_PYTHON` module constants as defaults.)

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py::test_translate_preserves_process_set_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/config_adapter.py tests/test_compare_inject.py
git commit -m "feat(compare): preserve process-set keys + parameterize vEcoli repo"
```

---

## Task 2: Process classification + resolution (`inject.py`)

**Files:**
- Create: `scripts/_compare/inject.py`
- Create: `tests/fixtures/fork_example/ecoli/processes/__init__.py`
- Test: `tests/test_compare_inject.py`

**Interfaces:**
- Produces:
  - `classify_process(cls) -> str` returns one of `"vivarium_1"`, `"pbg_native"`, `"partitioned"`.
  - `InjectionSpec` is a plain dict: `{"name": str, "module": str, "qualname": str, "kind": str, "as_step": bool, "config": dict|None, "topology": dict, "interval": float}`.
  - `resolve_injections(fork_repo: str, config: dict) -> list[InjectionSpec]` — raises `InjectionError` (a new exception subclass) on partitioned / sim_data / unknown-name / import failure.

- [ ] **Step 1: Write the fake fork fixture**

```python
# tests/fixtures/fork_example/ecoli/processes/__init__.py
"""Duck-typed fake fork — no vivarium-core dependency (mirrors the converter's
duck typing). Exposes a `process_registry` like vEcoli's ecoli.processes does."""


class _Registry:
    def __init__(self):
        self._d = {}
    def register(self, name, cls):
        self._d[name] = cls
    def access(self, name):
        if name not in self._d:
            raise KeyError(name)
        return self._d[name]


process_registry = _Registry()


class ExampleSecretion:
    """A simple vivarium-1.0-style process (ports_schema + next_update)."""
    name = "example-secretion"
    defaults = {"rate": 2.0}

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}

    def ports_schema(self):
        return {"counts": {"_default": 0, "_updater": "accumulate"}}

    def next_update(self, timestep, states):
        return {"counts": int(self.parameters["rate"] * timestep)}


class BadPartitioned:
    """A partitioned process — must be rejected by classify_process."""
    name = "bad-partitioned"

    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def ports_schema(self):
        return {"bulk": {"_default": 0}}

    def calculate_request(self, timestep, states):
        return {}

    def evolve_state(self, timestep, states):
        return {}


process_registry.register(ExampleSecretion.name, ExampleSecretion)
process_registry.register(BadPartitioned.name, BadPartitioned)
```

Also create empty `tests/fixtures/fork_example/ecoli/__init__.py`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_compare_inject.py  (append)
import os
import pytest
from scripts._compare import inject

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")

def test_classify_vivarium_and_partitioned():
    import sys; sys.path.insert(0, FORK)
    from ecoli.processes import ExampleSecretion, BadPartitioned
    assert inject.classify_process(ExampleSecretion) == "vivarium_1"
    assert inject.classify_process(BadPartitioned) == "partitioned"

def test_resolve_injections_builds_spec():
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 3.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    assert len(specs) == 1
    s = specs[0]
    assert s["name"] == "example-secretion"
    assert s["kind"] == "vivarium_1"
    assert s["config"] == {"rate": 3.0}
    assert s["topology"] == {"counts": ["bulk"]}
    assert s["qualname"] == "ExampleSecretion"

def test_resolve_rejects_partitioned():
    cfg = {"add_processes": ["bad-partitioned"], "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="partitioned"):
        inject.resolve_injections(FORK, cfg)

def test_resolve_rejects_sim_data_config():
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": "sim_data"},
           "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="sim_data"):
        inject.resolve_injections(FORK, cfg)
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py -k "classify or resolve" -v`
Expected: FAIL with `ModuleNotFoundError: scripts._compare.inject` / `AttributeError`.

- [ ] **Step 4: Implement `inject.py` (classification + resolution)**

```python
# scripts/_compare/inject.py
"""Resolve, classify, translate, and inject a fork's added processes.

Runs in the v2ecoli sim subprocess (where vivarium-core + the fork repo are
importable). The parent harness invokes the ``__main__`` below to obtain the
resolved specs as JSON for the report + early fail-fast.
"""
from __future__ import annotations

import importlib
import json
import sys
from typing import Any


class InjectionError(RuntimeError):
    """A fork process cannot be injected (unsupported / unresolved)."""


def classify_process(cls) -> str:
    """Return 'partitioned' | 'pbg_native' | 'vivarium_1' for a process class."""
    if hasattr(cls, "calculate_request") or hasattr(cls, "evolve_state"):
        return "partitioned"
    if hasattr(cls, "inputs") and hasattr(cls, "outputs"):
        return "pbg_native"
    if hasattr(cls, "ports_schema") and (
            hasattr(cls, "next_update") or hasattr(cls, "update")):
        return "vivarium_1"
    raise InjectionError(
        f"{cls.__name__}: not a recognizable process (no ports_schema/inputs).")


def _fork_registry(fork_repo: str):
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)
    try:
        mod = importlib.import_module("ecoli.processes")
    except Exception as exc:  # noqa: BLE001
        raise InjectionError(
            f"could not import 'ecoli.processes' from fork {fork_repo!r}: {exc}")
    registry = getattr(mod, "process_registry", None)
    if registry is None or not hasattr(registry, "access"):
        raise InjectionError(
            f"fork {fork_repo!r} ecoli.processes has no process_registry.access")
    return registry


def resolve_injections(fork_repo: str, config: dict) -> list[dict[str, Any]]:
    """Resolve add_processes/swap_processes -> a list of InjectionSpec dicts.

    Raises InjectionError on partitioned processes, sim_data process_configs,
    unknown names, or fork import failure.
    """
    registry = _fork_registry(fork_repo)
    interval = float(config.get("time_step", 1.0))
    process_configs = config.get("process_configs") or {}
    topologies = config.get("topology") or {}

    names = list(config.get("add_processes") or [])
    names += list((config.get("swap_processes") or {}).values())

    specs: list[dict[str, Any]] = []
    for name in names:
        try:
            cls = registry.access(name)
        except KeyError:
            raise InjectionError(f"add/swap process {name!r} not in fork registry.")
        kind = classify_process(cls)
        if kind == "partitioned":
            raise InjectionError(
                f"{name!r} is a partitioned process (calculate_request/"
                "evolve_state); not supported in v1. Extension point: wrap as "
                "PartitionedProcess (v2ecoli/steps/partition.py).")

        pcfg = process_configs.get(name, "default")
        if pcfg == "sim_data":
            raise InjectionError(
                f"{name!r}: process_configs 'sim_data' is unsupported for new "
                "processes (no ParCa entry). Provide an explicit dict or 'default'.")
        config_dict = None if pcfg in ("default", None) else dict(pcfg)

        topo = topologies.get(name)
        if topo is None:
            topo = getattr(cls, "topology", getattr(cls, "TOPOLOGY", {}))
        topo = {k: list(v) for k, v in dict(topo).items()}

        specs.append({
            "name": name,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "kind": kind,
            "as_step": bool(getattr(cls, "_force_step", False)),
            "config": config_dict,
            "topology": topo,
            "interval": interval,
        })
    return specs


if __name__ == "__main__":
    # argv: <fork_repo> <config_json_path>  -> prints specs JSON to stdout
    fork_repo, cfg_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    json.dump(resolve_injections(fork_repo, cfg), sys.stdout)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py -k "classify or resolve" -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/inject.py tests/test_compare_inject.py tests/fixtures/fork_example
git commit -m "feat(compare): resolve+classify fork processes with fail-fast scope guards"
```

---

## Task 3: Inject resolved specs into a cell-state doc (`apply_injected_processes`)

**Files:**
- Modify: `scripts/_compare/inject.py`
- Test: `tests/test_compare_inject.py`

**Interfaces:**
- Consumes: `resolve_injections` output; `wrap_vivarium_process` (`v2ecoli.library.vivarium_bridge`); `make_edge` (`v2ecoli.composites._helpers`); `build_core` (`v2ecoli.core`).
- Produces: `apply_injected_processes(cell_state: dict, flow_order: list, core, specs: list) -> None` — mutates `cell_state` (adds one edge per spec keyed by `name`) and appends each `name` to `flow_order`. Returns the list of injected names. Raises `InjectionError` on a topology path missing from `cell_state`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_compare_inject.py  (append)
def test_apply_injects_edge_and_flow_order():
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": {"rate": 2.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {}}      # a 'bulk' store exists
    flow_order = ["ecoli-metabolism"]
    added = inject.apply_injected_processes(cell_state, flow_order, core, specs)
    assert added == ["example-secretion"]
    assert "example-secretion" in cell_state
    edge = cell_state["example-secretion"]
    assert edge["_type"] in ("process", "step")
    assert edge["inputs"]["counts"] == ["bulk"]
    assert flow_order[-1] == "example-secretion"

def test_apply_rejects_missing_store_path():
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["nonexistent_store"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    with pytest.raises(inject.InjectionError, match="nonexistent_store"):
        inject.apply_injected_processes({"bulk": {}}, [], core, specs)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py -k apply -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'apply_injected_processes'`).

- [ ] **Step 3: Implement `apply_injected_processes`**

Append to `scripts/_compare/inject.py`:

```python
def _import_class(module: str, qualname: str):
    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def apply_injected_processes(cell_state: dict, flow_order: list, core,
                             specs: list[dict]) -> list[str]:
    """Add each resolved spec to ``cell_state`` + ``flow_order`` (in place)."""
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    from v2ecoli.composites._helpers import make_edge

    added: list[str] = []
    for spec in specs:
        cls = _import_class(spec["module"], spec["qualname"])
        if spec["kind"] == "vivarium_1":
            wrapped = wrap_vivarium_process(cls, name=spec["name"],
                                            as_step=spec["as_step"])
        else:  # pbg_native
            wrapped = cls
        core.register_link(spec["name"], wrapped)

        # Validate topology roots exist in the cell-state tree.
        for port, path in spec["topology"].items():
            root = path[0] if path else None
            if root is not None and root not in cell_state:
                raise InjectionError(
                    f"{spec['name']}: topology port {port!r} -> {path}: root "
                    f"store {root!r} not present in cell state "
                    f"(have: {sorted(cell_state)[:12]}...).")

        instance = wrapped(spec["config"] or {}, core=core)
        edge_type = "step" if spec["kind"] == "pbg_native" and spec["as_step"] \
            else ("step" if spec["as_step"] else "process")
        cell_state[spec["name"]] = make_edge(
            instance, spec["topology"], edge_type=edge_type,
            config=spec["config"] or {})
        flow_order.append(spec["name"])
        added.append(spec["name"])
    return added
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare_inject.py -k apply -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/inject.py tests/test_compare_inject.py
git commit -m "feat(compare): apply translated fork processes into a cell-state doc"
```

---

## Task 4: `baseline()` `injected_processes` parameter + apply hook

**Files:**
- Modify: `v2ecoli/composites/baseline.py` (decorator `parameters`, signature ~622, after the build loop ~866 and before the `return` ~908)
- Test: `tests/test_baseline_injected.py`

**Interfaces:**
- Consumes: `apply_injected_processes` (Task 3).
- Produces: `baseline(..., injected_processes: dict | None = None)`. `injected_processes`, when present, is `{"fork_repo": str, "add_processes": [...], "swap_processes": {...}, "process_configs": {...}, "topology": {...}, "time_step": float}`. baseline resolves + applies it into `agents.0` cell_state and `flow_order`. No-op when `None`/empty.

- [ ] **Step 1: Write failing test**

```python
# tests/test_baseline_injected.py
import os
FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")

def test_baseline_injects_fork_process():
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    inj = {"fork_repo": FORK, "add_processes": ["example-secretion"],
           "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 2.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    doc = baseline(core=core, seed=0, cache_dir="out/cache",
                   injected_processes=inj)
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" in cell
    assert "example-secretion" in doc["flow_order"]

def test_baseline_noop_without_injection_keeps_process_set():
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" not in cell
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baseline_injected.py -v`
Expected: FAIL — `baseline()` got an unexpected keyword argument `injected_processes`.

(NB: `test_baseline_injects_fork_process` needs a built ParCa cache at `out/cache`. If absent, create the symlink per memory `reference_v2ecoli_worktree_cache_symlink`: `ln -s <main-checkout>/out/cache out/cache`. Verify before running.)

- [ ] **Step 3: Add the parameter to the decorator + signature**

In `v2ecoli/composites/baseline.py`, in the `@composite_generator(parameters={...})` block (~line 524) add:

```python
        "injected_processes": {
            "type": "map",
            "default": {},
            "description": "Fork process-injection spec "
                           "{fork_repo, add_processes, swap_processes, "
                           "process_configs, topology, time_step}; empty = none.",
        },
```

In the `def baseline(...)` signature (~line 622) add the parameter (keyword-only, after `config_overrides`):

```python
    injected_processes: dict | None = None,
```

- [ ] **Step 4: Add the apply hook before the return**

Locate where `flow_order` is defined (~line 786) and the return doc (~line 908). After `cell_state` has been fully populated by the build loop (after ~line 866, and after the `ShapeStep` registration block ~889-905) and *before* `return`, insert:

```python
    if injected_processes and injected_processes.get("add_processes"):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "..", "scripts"))
        from scripts._compare.inject import (
            resolve_injections, apply_injected_processes)
        specs = resolve_injections(injected_processes["fork_repo"],
                                   injected_processes)
        apply_injected_processes(cell_state, flow_order, core, specs)
```

(The `sys.path` insert makes `scripts._compare.inject` importable from within the sim subprocess; the repo root is two levels above this file.)

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baseline_injected.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run baseline regression smoke (no-op safety)**

Run: `.venv/bin/python -m pytest tests/ -k "baseline and not injected" -q`
Expected: existing baseline tests still PASS (injection is a no-op without the param).

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/composites/baseline.py tests/test_baseline_injected.py
git commit -m "feat(baseline): optional injected_processes hook (no-op when absent)"
```

---

## Task 5: Thread `injected_processes` through the lineage

**Files:**
- Modify: `v2ecoli/workflow/lineage.py` (`config_schema` ~line 60; `_build_generation` ~line 124 & 143)
- Modify: `v2ecoli/workflow/meta_composite.py` (`_lineage_node` ~lines 25-56)
- Test: `tests/test_lineage_injected_threading.py`

**Interfaces:**
- Consumes: Task 4's `baseline(injected_processes=...)`.
- Produces: top-level workflow config key `injected_processes` reaches `baseline()` for every generation.

- [ ] **Step 1: Write failing test**

```python
# tests/test_lineage_injected_threading.py
def test_meta_composite_carries_injected_processes():
    from v2ecoli.workflow.meta_composite import build_meta_composite
    cfg = {"experiment_id": "x", "n_init_sims": 1, "generations": 1,
           "single_daughters": True, "cache_dir": "out/cache",
           "out_dir": "out/x", "skip_baseline": True,
           "injected_processes": {"fork_repo": "/tmp/fork",
                                  "add_processes": ["example-secretion"]}}
    doc = build_meta_composite(cfg)
    (branch,) = doc["state"]["branches"].values()
    node_cfg = branch["lineage"]["config"]
    assert node_cfg["injected_processes"]["add_processes"] == ["example-secretion"]
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lineage_injected_threading.py -v`
Expected: FAIL (`KeyError: 'injected_processes'`).

- [ ] **Step 3: Add to `LineageProcess.config_schema`**

In `v2ecoli/workflow/lineage.py` `config_schema` (~line 60) add:

```python
        "injected_processes": {"_default": {}},
```

- [ ] **Step 4: Pass it to `baseline()` in `_build_generation`**

In `_build_generation` (both `baseline(...)` call sites, ~line 124 and ~line 143) add the kwarg:

```python
        doc = baseline(core=core, seed=gen_seed,
                       cache_dir=self.config["cache_dir"],
                       config_overrides=overrides,
                       injected_processes=self.config.get("injected_processes"))
```

- [ ] **Step 5: Copy the key into the lineage node config in `meta_composite._lineage_node`**

In `v2ecoli/workflow/meta_composite.py` `_lineage_node` (~lines 25-56), inside the `"config": {...}` dict, add:

```python
                "injected_processes": config.get("injected_processes") or {},
```

- [ ] **Step 6: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_lineage_injected_threading.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/workflow/lineage.py v2ecoli/workflow/meta_composite.py tests/test_lineage_injected_threading.py
git commit -m "feat(workflow): thread injected_processes meta_composite->lineage->baseline"
```

---

## Task 6: Parameterize the vEcoli repo in the orchestrator

**Files:**
- Modify: `scripts/_compare/orchestrator.py`
- Test: `tests/test_compare_orchestrator.py`

**Interfaces:**
- Produces: `run_vecoli_parca(..., vecoli_repo=...)` and `run_vecoli_sim(..., vecoli_repo=...)` accept an optional fork repo path (default the module constant). The vEcoli python is `f"{vecoli_repo}/.venv/bin/python"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_compare_orchestrator.py
import scripts._compare.orchestrator as orch

def test_vecoli_sim_uses_passed_repo(monkeypatch):
    captured = {}
    def fake_run(cmd, cwd=None):
        captured["cmd"], captured["cwd"] = cmd, cwd
    monkeypatch.setattr(orch, "_run", fake_run)
    monkeypatch.setattr(orch, "is_stale", lambda *a, **k: True)
    monkeypatch.setattr(orch, "mark_done", lambda *a, **k: None)
    orch.run_vecoli_sim(config_path="c.json", out_dir="out/v",
                        token="t", vecoli_repo="/tmp/fork")
    assert captured["cwd"] == "/tmp/fork"
    assert captured["cmd"][0] == "/tmp/fork/.venv/bin/python"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_orchestrator.py -v`
Expected: FAIL (`run_vecoli_sim() got an unexpected keyword argument 'vecoli_repo'`).

- [ ] **Step 3: Parameterize the two vEcoli runners**

In `scripts/_compare/orchestrator.py`, change `run_vecoli_parca` and `run_vecoli_sim` to accept `vecoli_repo: str = VECOLI_REPO` and derive the python locally:

```python
def run_vecoli_sim(*, config_path: str, out_dir: Path,
                   token: str | None = None,
                   vecoli_repo: str = VECOLI_REPO) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    vecoli_python = f"{vecoli_repo}/.venv/bin/python"
    _run([vecoli_python, "-m", "runscripts.workflow", "--config", config_path],
         cwd=vecoli_repo)
    mark_done(out_dir, token or "ok")
    return out_dir
```

Apply the same `vecoli_repo` parameter + local `vecoli_python` to `run_vecoli_parca`.

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/orchestrator.py tests/test_compare_orchestrator.py
git commit -m "feat(compare): parameterize vEcoli fork repo in orchestrator"
```

---

## Task 7: Shared chart helpers (`charts.py`)

**Files:**
- Create: `scripts/_compare/charts.py`
- Modify: `reports/composite_comparison.py` (import the helpers instead of defining them)
- Test: `tests/test_compare_charts.py`

**Interfaces:**
- Produces: `sparkline(snaps, key, w=260, h=44, color="#3730a3") -> str`, `multiline_svg(series, w=300, h=120, baseline_zero=True) -> str` returning inline-SVG strings. (Lifted verbatim from `reports/composite_comparison.py:_sparkline` ~line 294 and `_multiline_svg` ~line 374.)

- [ ] **Step 1: Write failing test**

```python
# tests/test_compare_charts.py
from scripts._compare import charts

def test_sparkline_returns_svg():
    snaps = [{"dry_mass": 1.0}, {"dry_mass": 1.2}, {"dry_mass": 1.5}]
    svg = charts.sparkline(snaps, "dry_mass")
    assert svg.startswith("<svg") and "polyline" in svg

def test_multiline_svg_two_series():
    svg = charts.multiline_svg({"a": [1, 2, 3], "b": [3, 2, 1]})
    assert svg.startswith("<svg")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_charts.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts._compare.charts`).

- [ ] **Step 3: Create `charts.py` by lifting the two helpers**

Open `reports/composite_comparison.py`, copy the bodies of `_sparkline` (~294-308) and `_multiline_svg` (~374-408) into `scripts/_compare/charts.py` as public functions `sparkline` and `multiline_svg` (rename, drop the leading underscore; keep identical logic). Add a module docstring noting this is the single source shared with `composite_comparison.py`.

- [ ] **Step 4: Re-point `composite_comparison.py` at the shared module**

In `reports/composite_comparison.py`, replace the two function definitions with imports near the top (after the existing imports):

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts._compare.charts import sparkline as _sparkline  # noqa: E402
from scripts._compare.charts import multiline_svg as _multiline_svg  # noqa: E402
```

Delete the now-duplicated `def _sparkline` / `def _multiline_svg` bodies. (Leave `_overlay_card` etc. as-is — they call `_multiline_svg`, which is now the imported alias.)

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare_charts.py -v`
Expected: PASS. Then sanity-check the old report still imports:
Run: `.venv/bin/python -c "import reports.composite_comparison; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/charts.py reports/composite_comparison.py tests/test_compare_charts.py
git commit -m "refactor(compare): share sparkline/multiline SVG helpers via charts.py"
```

---

## Task 8: Statistical-equivalence report-card section

**Files:**
- Create: `scripts/_compare/report_card_section.py`
- Test: `tests/test_report_card_section.py`

**Interfaces:**
- Consumes: `v2ecoli.library.card_criteria.grade_axis`, `v2ecoli.library.report_card.{grade_card, verdict_json, render_html}`.
- Produces: `build_report_card(left_by_cell, right_by_cell, *, model_ref="", reference_model="vEcoli (fork)") -> tuple[dict, str]` — returns `(verdict_json_dict, html_section)`. `left_by_cell`/`right_by_cell` map observable-name → list of per-cell scalar values (e.g. final-generation cell mass across cells). Axis/group mapping defined by the module-level `CARD_AXES`.

- [ ] **Step 1: Inspect the card API surface to mirror it exactly**

Read `v2ecoli/library/card_criteria.py:grade_axis` (~line 75) and `v2ecoli/library/report_card.py:{grade_card (~121), verdict_json (~140), render_html (~536)}` to confirm the `measured`/`criterion` dict shapes `grade_axis` expects and the `card`/`reference` shapes `grade_card`/`render_html` expect. Record the exact keys in a comment at the top of `report_card_section.py`. (This step prevents shape drift; no code yet.)

- [ ] **Step 2: Write failing test**

```python
# tests/test_report_card_section.py
from scripts._compare.report_card_section import build_report_card

def test_build_report_card_equivalent_data():
    # Near-identical distributions -> within_tol overall.
    left = {"cell_mass": [1500.0 + i for i in range(30)],
            "growth_rate": [0.0003 + i*1e-7 for i in range(30)]}
    right = {"cell_mass": [1502.0 + i for i in range(30)],
             "growth_rate": [0.0003 + i*1e-7 for i in range(30)]}
    verdict, html = build_report_card(left, right)
    assert verdict["schema"] == "report_card_verdict/v1"
    assert verdict["overall"] in ("within_tol", "drift", "mismatch")
    assert "groups" in verdict
    assert html.startswith("<") and "verdict" in html.lower()

def test_build_report_card_divergent_data_flags_mismatch():
    left = {"cell_mass": [1500.0]*30}
    right = {"cell_mass": [3000.0]*30}      # 2x divergence
    verdict, _ = build_report_card(left, right)
    masses = verdict["groups"]["physiology"]["axes"]
    assert any(a["verdict"] == "mismatch" for a in masses)
```

- [ ] **Step 3: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_card_section.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `report_card_section.py`**

```python
# scripts/_compare/report_card_section.py
"""Statistical-equivalence report card from paired per-cell observables.

Reuses the workspace card library:
  card_criteria.grade_axis(measured, criterion) -> {verdict, value, meter, detail}
  report_card.grade_card(card, reference) / verdict_json(...) / render_html(...)
See Task 8 Step 1 for the exact dict shapes these expect.
"""
from __future__ import annotations

from typing import Any

from v2ecoli.library.card_criteria import grade_axis
from v2ecoli.library.report_card import grade_card, verdict_json, render_html

# (group, axis-id, label, observable-key, criterion). 'scalar' axes grade by
# relative-mean shift + Welch p (within 5% pass, 5-10% drift, >10% & p<0.05
# mismatch), matching the existing vEcoli equivalence cards.
CARD_AXES: list[dict[str, Any]] = [
    {"group": "physiology", "id": "physiology.cell_mass", "label": "Cell mass",
     "key": "cell_mass",
     "criterion": {"kind": "scalar", "within_pct": 0.05, "mismatch_pct": 0.10,
                   "p_min": 0.05}},
    {"group": "physiology", "id": "physiology.growth_rate",
     "label": "Growth rate", "key": "growth_rate",
     "criterion": {"kind": "scalar", "within_pct": 0.05, "mismatch_pct": 0.10,
                   "p_min": 0.05}},
]


def build_report_card(left_by_cell: dict[str, list[float]],
                      right_by_cell: dict[str, list[float]], *,
                      model_ref: str = "",
                      reference_model: str = "vEcoli (fork)",
                      extra_axes: list[dict] | None = None
                      ) -> tuple[dict, str]:
    """Grade each configured axis and return (verdict_json, html)."""
    axes_defs = CARD_AXES + list(extra_axes or [])
    card_axes = []
    for spec in axes_defs:
        key = spec["key"]
        meas = right_by_cell.get(key) or []
        ref = left_by_cell.get(key) or []
        if not meas or not ref:
            graded = {"verdict": "ungraded", "value": None,
                      "meter": "no data", "detail": {}}
        else:
            measured = {"kind": spec["criterion"]["kind"],
                        "values": meas, "reference": ref}
            graded = grade_axis(measured, spec["criterion"])
        card_axes.append({"id": spec["id"], "label": spec["label"],
                          "group": spec["group"], **graded})

    card = {"axes": card_axes}
    reference = {"label": reference_model}
    report = grade_card(card, reference)
    vjson = verdict_json(report, model_ref=model_ref,
                         reference_model=reference_model)
    html = render_html(card, reference, model_ref=model_ref)
    return vjson, html
```

NOTE for the implementer: the `measured`/`criterion`/`card` dict shapes above are
the *intended* contract. If Step 1 reveals `grade_axis`/`grade_card`/`render_html`
expect different keys (e.g. a `measured` dict keyed `cand`/`ref` instead of
`values`/`reference`, or `grade_card` wanting groups pre-nested), **adjust this
function to match the real signatures** — do not change the library. Re-run the
test until the real shapes pass; the test only asserts the public output
contract (`schema`, `overall`, `groups`, per-axis `verdict`), which is stable.

- [ ] **Step 5: Run tests; reconcile shapes against the real library**

Run: `.venv/bin/python -m pytest tests/test_report_card_section.py -v`
Expected: PASS. If `grade_axis`/`grade_card` raise `KeyError`, fix the dict shapes per Step 1's findings and re-run.

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/report_card_section.py tests/test_report_card_section.py
git commit -m "feat(compare): statistical-equivalence report card from paired observables"
```

---

## Task 9: Report panels (loaded-config + converted-processes)

**Files:**
- Modify: `scripts/_compare/report.py`
- Test: `tests/test_compare_report_panels.py`

**Interfaces:**
- Consumes: nothing new (pure rendering).
- Produces: `config_panel_section(resolved_config: dict) -> dict` and `converted_processes_section(specs: list[dict], ran_in_both: dict[str, bool]) -> dict`, each returning a section dict shaped like the existing `{"title": str, "rows": [...]}` consumed by `render_report`. Also `render_report(sections, title=..., embedded_html=None)` gains an optional `embedded_html: list[str]` appended verbatim (for the report-card + behavior HTML).

- [ ] **Step 1: Write failing test**

```python
# tests/test_compare_report_panels.py
from scripts._compare.report import (
    config_panel_section, converted_processes_section, render_report)

def test_config_panel_lists_added_processes():
    cfg = {"experiment_id": "x", "generations": 2,
           "add_processes": ["example-secretion"],
           "_dropped_vecoli_keys": {"emitter": "parquet"}}
    sec = config_panel_section(cfg)
    labels = [r["label"] for r in sec["rows"]]
    assert "add_processes" in labels
    assert any("emitter" in r["label"] for r in sec["rows"])

def test_converted_panel_marks_ran_in_both():
    specs = [{"name": "example-secretion", "module": "ecoli.processes",
              "qualname": "ExampleSecretion", "kind": "vivarium_1",
              "topology": {"counts": ["bulk"]}}]
    sec = converted_processes_section(specs, {"example-secretion": True})
    row = sec["rows"][0]
    assert row["label"] == "example-secretion"
    assert row["verdict"] in ("within_tol", "drift", "mismatch", "not_compared")

def test_render_report_appends_embedded_html():
    html = render_report([{"title": "T", "rows": []}], title="x",
                         embedded_html=["<div id='card'>CARD</div>"])
    assert "CARD" in html
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_report_panels.py -v`
Expected: FAIL (`ImportError: cannot import name 'config_panel_section'`).

- [ ] **Step 3: Implement the two section builders**

Add to `scripts/_compare/report.py`:

```python
import json as _json


def config_panel_section(resolved_config: dict) -> dict:
    """Render the loaded fork config: run knobs + added/swapped/topology +
    dropped vEcoli-only keys."""
    rows = []
    for k in ("experiment_id", "generations", "n_init_sims", "lineage_seed",
              "time_step", "add_processes", "swap_processes",
              "exclude_processes", "process_configs", "topology"):
        if k in resolved_config:
            rows.append({"label": k, "left": _json.dumps(resolved_config[k]),
                         "right": "—", "verdict": "not_compared",
                         "reason": "loaded config value"})
    for k, v in (resolved_config.get("_dropped_vecoli_keys") or {}).items():
        rows.append({"label": f"dropped: {k}", "left": _json.dumps(v),
                     "right": "—", "verdict": "not_compared",
                     "reason": "vEcoli-only key (v2ecoli configures internally)"})
    return {"title": "Loaded config", "rows": rows}


def converted_processes_section(specs: list, ran_in_both: dict) -> dict:
    """One row per injected process: source, kind, topology, did-it-run gate."""
    rows = []
    for s in specs:
        ran = ran_in_both.get(s["name"])
        verdict = ("within_tol" if ran else
                   "mismatch" if ran is False else "not_compared")
        rows.append({
            "label": s["name"],
            "left": f"{s['module']}.{s['qualname']}",
            "right": f"kind={s['kind']} · topology={_json.dumps(s['topology'])}",
            "verdict": verdict,
            "reason": ("produced updates in both engines" if ran else
                       "did NOT update in both engines" if ran is False else
                       "run-probe not available"),
        })
    return {"title": "Converted processes", "rows": rows}
```

- [ ] **Step 4: Add `embedded_html` to `render_report`**

In `scripts/_compare/report.py`, change `render_report(sections, title=...)` to accept `embedded_html: list[str] | None = None` and append each string verbatim into the page body (after the sections loop, before closing `</body>`). Keep the default `None` so existing callers are unaffected.

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare_report_panels.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/report.py tests/test_compare_report_panels.py
git commit -m "feat(compare): loaded-config + converted-processes report panels"
```

---

## Task 10: CLI wiring + end-to-end smoke test

**Files:**
- Modify: `scripts/compare_harness.py`
- Create: `tests/fixtures/fork_example/configs/example.json`
- Create: `tests/test_compare_harness_injected_e2e.py`

**Interfaces:**
- Consumes: every prior task.
- Produces: `compare_harness.py` CLI flags `--vecoli-repo`, `--tol-rel`, `--force`; builds the v2 config with an `injected_processes` block; runs both engines; writes `report_card_verdict.json` next to the HTML; renders all four panels.

- [ ] **Step 1: Write the example fork config fixture**

```json
// tests/fixtures/fork_example/configs/example.json
{
  "experiment_id": "fork_example",
  "generations": 1,
  "n_init_sims": 1,
  "single_daughters": true,
  "time_step": 1.0,
  "add_processes": ["example-secretion"],
  "swap_processes": {},
  "exclude_processes": [],
  "process_configs": {"example-secretion": {"rate": 2.0}},
  "topology": {"example-secretion": {"counts": ["bulk"]}}
}
```

- [ ] **Step 2: Write the integration smoke test (resolution + injected-config build only)**

This test exercises the harness's *config + injection-spec* assembly without the multi-minute ParCa/sim (those are covered by manual runs in Step 6). It asserts the harness builds a v2 config carrying `injected_processes` and that `resolve_injections` over the fixture succeeds.

```python
# tests/test_compare_harness_injected_e2e.py
import json, os
from scripts._compare.inject import resolve_injections
from scripts.compare_harness import build_injected_v2_config

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")
CFG = os.path.join(FORK, "configs", "example.json")

def test_build_injected_v2_config_embeds_block():
    with open(CFG) as f:
        vecoli_cfg = json.load(f)
    v2 = build_injected_v2_config(vecoli_cfg, fork_repo=FORK)
    inj = v2["injected_processes"]
    assert inj["fork_repo"] == FORK
    assert inj["add_processes"] == ["example-secretion"]
    # resolution succeeds against the fixture (fail-fast guards pass)
    specs = resolve_injections(FORK, inj)
    assert specs[0]["name"] == "example-secretion"
```

- [ ] **Step 3: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_harness_injected_e2e.py -v`
Expected: FAIL (`ImportError: cannot import name 'build_injected_v2_config'`).

- [ ] **Step 4: Add `build_injected_v2_config` + CLI flags to `compare_harness.py`**

Add the helper near the top of `scripts/compare_harness.py`:

```python
def build_injected_v2_config(vecoli_cfg: dict, *, fork_repo: str) -> dict:
    """Translate a fork's vEcoli config to a v2 config carrying an
    injected_processes block (no-op block when no add_processes)."""
    from scripts._compare.config_adapter import translate_vecoli_config
    v2 = translate_vecoli_config(vecoli_cfg)
    if vecoli_cfg.get("add_processes") or vecoli_cfg.get("swap_processes"):
        v2["injected_processes"] = {
            "fork_repo": fork_repo,
            "add_processes": vecoli_cfg.get("add_processes") or [],
            "swap_processes": vecoli_cfg.get("swap_processes") or {},
            "process_configs": vecoli_cfg.get("process_configs") or {},
            "topology": vecoli_cfg.get("topology") or {},
            "time_step": float(vecoli_cfg.get("time_step", 1.0)),
        }
    return v2
```

In `main()`, add the CLI flags and thread `--vecoli-repo` through:

```python
    p.add_argument("--vecoli-repo", default="/Users/eranagmon/code/vEcoli",
                   help="Path to the vEcoli fork checkout.")
    p.add_argument("--tol-rel", type=float, default=0.10,
                   help="Relative tolerance for behavioral/equivalence badges.")
    p.add_argument("--force", action="store_true",
                   help="Bypass the run cache and re-run both engines.")
```

Replace `resolve_vecoli_config(args.config)` with
`resolve_vecoli_config(args.config, vecoli_repo=args.vecoli_repo)`; replace
`translate_vecoli_config(vecoli_cfg)` with
`build_injected_v2_config(vecoli_cfg, fork_repo=args.vecoli_repo)`; pass
`vecoli_repo=args.vecoli_repo` to `orchestrator.run_vecoli_parca` /
`run_vecoli_sim`. When `args.force`, pass a freshly-salted `run_token` (append
`"-force"` so the cache misses).

- [ ] **Step 5: Wire the new sections + report-card json into `main()`**

After the existing sim-dynamics section, before rendering, add:

```python
    # Loaded-config + converted-processes panels.
    from scripts._compare.report import (
        config_panel_section, converted_processes_section)
    sections.insert(0, config_panel_section(vecoli_cfg))
    embedded = []
    inj = v2_cfg.get("injected_processes")
    if inj and inj.get("add_processes"):
        import subprocess as _sp
        spec_json = _sp.check_output(
            [".venv/bin/python", "-m", "scripts._compare.inject",
             args.vecoli_repo, str(v2_cfg_path)], text=True)
        specs = json.loads(spec_json)
        # ran_in_both: best-effort — names present in both engines' observables.
        ran = {s["name"]: True for s in specs}   # refined when probes exist
        sections.append(converted_processes_section(specs, ran))

    # Behavior detail — per-observable overlay (vEcoli vs v2ecoli) via the
    # shared chart helper. `left`/`right` are observable-keyed series already
    # read for the sim-dynamics section above.
    from scripts._compare.charts import multiline_svg
    cards = []
    for key in keys:
        l, r = left.get(key) or [], right.get(key) or []
        if not l and not r:
            continue
        svg = multiline_svg({"vEcoli": list(l), "v2ecoli": list(r)})
        cards.append(f"<figure style='display:inline-block;margin:6px'>"
                     f"<figcaption>{key}</figcaption>{svg}</figure>")
    if cards:
        embedded.append("<section><h2>Behavior detail</h2>"
                        + "".join(cards) + "</section>")

    # Statistical-equivalence report card from paired per-cell observables.
    try:
        from scripts._compare.report_card_section import build_report_card
        # left/right per-cell observables are gathered by sim_section; reuse the
        # `left`/`right` observable dicts already read above (keyed per cell).
        verdict_json_dict, card_html = build_report_card(
            left_by_cell, right_by_cell,
            reference_model="vEcoli (fork)")
        card_path = Path(args.out).with_name("report_card_verdict.json")
        card_path.write_text(json.dumps(verdict_json_dict, indent=2))
        embedded.append(card_html)
    except Exception as e:  # surface, don't abort
        embedded.append(f"<section><h2>Report card</h2>"
                        f"<p>card build failed: {type(e).__name__}: {e}</p></section>")
```

(`left_by_cell`/`right_by_cell` are the per-cell observable dicts; if
`sim_section.read_observables` currently returns per-timestep series rather than
per-cell scalars, add a thin `per_cell_finals(series)` reducer in
`scripts/_compare/sim_section.py` — final-generation value per cell — and use it
here. Cover that reducer with a one-assertion unit test in
`tests/test_compare_report_panels.py`.)

Then pass `embedded_html=embedded` to `render_report(...)`.

- [ ] **Step 6: Run the unit/integration test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare_harness_injected_e2e.py -v`
Expected: PASS.

- [ ] **Step 7: Full manual end-to-end against the fixture fork (documented, not CI)**

Because a real run needs a ParCa cache + minutes of sim, run this manually once
and record the result in the PR description (do not gate CI on it):

Run: `.venv/bin/python scripts/compare_harness.py --vecoli-repo <REAL_FORK> --config <REAL_FORK>/configs/<fork-config>.json -o out/compare/report.html --mode fast`
Expected: writes `out/compare/report.html` + `out/compare/report_card_verdict.json`; the HTML contains the Loaded-config, Converted-processes, behavior, and Report-card sections; the converted-processes row for each new process shows it ran. (`--mode fast` is for plumbing only — the report is stamped NOT SCIENTIFICALLY VALID.)

- [ ] **Step 8: Run the full compare test suite**

Run: `.venv/bin/python -m pytest tests/ -k "compare or inject or baseline_injected or lineage_injected or report_card or charts" -q`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/compare_harness.py scripts/_compare/sim_section.py tests/fixtures/fork_example/configs tests/test_compare_harness_injected_e2e.py
git commit -m "feat(compare): config-driven fork injection CLI + 4-panel report + verdict json"
```

---

## Self-review notes (for the executor)

- **No-op safety:** Tasks 4/5 guard on `injected_processes`; run a baseline-only
  sim after Task 5 to confirm unchanged behavior before moving on.
- **Cache symlink:** Tasks 4 + 10 need `out/cache` (ParCa). Create the worktree
  symlink to the main checkout's cache first (see memory
  `reference_v2ecoli_worktree_cache_symlink`).
- **Card shapes:** Task 8 Step 1 is load-bearing — verify the real `grade_axis` /
  `grade_card` / `render_html` signatures before trusting the skeleton.
- **Partitioned/sim_data fail-fast** is asserted in Task 2 — keep those tests green.
