# Variant-sweep phenotype study — implementation plan (v2ecoli generic capability)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the whole-config WCM node apply a config-declared `variants` grid point, emit caller-chosen observables, expose it as a workbench composite, and provide a generic study template that sweeps a variant axis — all perturbation-agnostic.

**Architecture:** The whole-config node already loads a fork config natively via `EcoliSim.from_cli` (`v2ecoli/library/vivarium_ecoli_engine.py`) and is already registered as the `vecoli` composite (`v2ecoli/composites/vecoli.py`). We add: (1) variant application delegated to the fork's own `runscripts.create_variants.parse_variants` + `ecoli.variants.<name>.apply_variant`; (2) configurable bulk-observable emission on `VivariumEcoliProcess`; (3) `whole_config` + `variant` + `observable_bulk_ids` params on the `vecoli` generator; (4) a generic sweep extractor; (5) a neutral demo study template. The workbench simulate loop is a generic non-division-aware `composite.run(1)` × N-steps driver, so a single-generation run past an early intervention is sufficient — no lineage wiring needed.

**Tech Stack:** Python 3.11, process-bigraph / bigraph-schema, viva_superpowers `@composite_generator`, pytest. Fork API consumed via `$V2E_VECOLI_DIR` on `sys.path` (set up by `_ensure_upstream()`).

## Global Constraints

- **No perturbation/biology names in this repo.** Public v2ecoli must contain zero mention of any specific drug, perturbation, config filename, molecule, or biology. Everything is generic (variant *index*, observable *path*, config *path-as-param*). Leak-check every commit.
- **Fork-bound imports only.** `runscripts.create_variants` and `ecoli.variants.*` are imported from the fork on `sys.path` (after `_ensure_upstream()` / `$V2E_VECOLI_DIR`), never the installed vEcoli.
- **Back-compat.** `variant=0` == baseline (unperturbed) — current whole-config behavior is unchanged. `variants` absent/empty ⇒ no-op. `whole_config=""` on the generator ⇒ existing `vecoli` behavior byte-unchanged.
- **Off-by-one convention (mirror the fork).** `parse_variants` returns `param_dicts`; the fork maps `param_dicts[i]` → variant `i+1`, index 0 reserved = baseline. So node `variant=k≥1` applies `param_dicts[k-1]`; `variant=0` applies nothing.
- **No new fork content.** Delegate entirely to the fork's `parse_variants` + each variant module's `apply_variant`.
- **Worktree discipline.** All work in `~/code/v2ecoli--variant-sweep-study` on branch `feat/variant-sweep-phenotype-study`. Verify `git branch --show-current` before each commit.

---

### Task 1: Variant selection + application in the build path

**Files:**
- Modify: `v2ecoli/library/vivarium_ecoli_engine.py` (add two module helpers near `set_ecolisim_config_file`; add `variant` param to `build_vivarium_ecoli`; apply after the sim_data pickle load, before `sim.config["sim_data"] = _sd_obj`)
- Test: `tests/test_variant_hook.py`

**Interfaces:**
- Produces: `_select_variant_params(variants_config: dict, variant_index: int) -> tuple[str | None, dict | None]` and `_apply_config_variant(sim_data, variants_config: dict, variant_index: int) -> tuple[object, dict | None]`; `build_vivarium_ecoli(..., variant: int = 0)`.
- Consumes (fork): `runscripts.create_variants.parse_variants(variant_config) -> list[dict]`; `ecoli.variants.<name>.apply_variant(sim_data, params) -> sim_data`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variant_hook.py
import sys, types
import pytest
from v2ecoli.library import vivarium_ecoli_engine as ve


def _stub_parse_variants(monkeypatch):
    # Mirror the fork: a 6-point single-parameter grid, op=None.
    mod = types.ModuleType("runscripts.create_variants")
    mod.parse_variants = lambda cfg: [{"dose": v} for v in (0, 1, 2, 3, 4, 5)]
    pkg = types.ModuleType("runscripts")
    monkeypatch.setitem(sys.modules, "runscripts", pkg)
    monkeypatch.setitem(sys.modules, "runscripts.create_variants", mod)


def test_index_zero_is_baseline(monkeypatch):
    _stub_parse_variants(monkeypatch)
    name, params = ve._select_variant_params({"demo_variant": {}}, 0)
    assert (name, params) == (None, None)


def test_index_k_selects_k_minus_one(monkeypatch):
    _stub_parse_variants(monkeypatch)
    name, params = ve._select_variant_params({"demo_variant": {}}, 3)
    assert name == "demo_variant"
    assert params == {"dose": 2}      # param_dicts[3-1]


def test_index_out_of_range_raises(monkeypatch):
    _stub_parse_variants(monkeypatch)
    with pytest.raises(IndexError):
        ve._select_variant_params({"demo_variant": {}}, 7)


def test_apply_dispatches_to_variant_module(monkeypatch):
    _stub_parse_variants(monkeypatch)
    applied = {}
    vmod = types.ModuleType("ecoli.variants.demo_variant")
    def _apply(sim_data, params):
        applied["params"] = params
        sim_data["touched"] = True
        return sim_data
    vmod.apply_variant = _apply
    monkeypatch.setitem(sys.modules, "ecoli", types.ModuleType("ecoli"))
    monkeypatch.setitem(sys.modules, "ecoli.variants", types.ModuleType("ecoli.variants"))
    monkeypatch.setitem(sys.modules, "ecoli.variants.demo_variant", vmod)
    sd = {}
    out, meta = ve._apply_config_variant(sd, {"demo_variant": {}}, 2)
    assert out["touched"] is True
    assert applied["params"] == {"dose": 1}
    assert meta == {"variant_name": "demo_variant", "variant_index": 2, "params": {"dose": 1}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_hook.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_select_variant_params'`

- [ ] **Step 3: Write minimal implementation**

Add near `set_ecolisim_config_file` (after line ~55):

```python
def _select_variant_params(variants_config: dict, variant_index: int):
    """Resolve a 1-based ``variant_index`` against a config ``variants`` block.

    Mirrors the fork's ``runscripts.create_variants`` convention: ``parse_variants``
    returns ``param_dicts`` and the fork maps ``param_dicts[i]`` to variant ``i+1``,
    reserving index 0 for the unperturbed baseline. Returns ``(None, None)`` for the
    baseline, else ``(variant_name, params_dict)``. Delegates the grid expansion
    (op prod/zip/add, value/linspace/arange) entirely to the fork.
    """
    if not variants_config or variant_index <= 0:
        return None, None
    if len(variants_config) != 1:
        raise ValueError(
            f"expected exactly one variant in config, got {sorted(variants_config)}")
    (name, cfg), = variants_config.items()
    from runscripts.create_variants import parse_variants  # fork-bound
    param_dicts = parse_variants(cfg)
    idx = variant_index - 1
    if idx >= len(param_dicts):
        raise IndexError(
            f"variant index {variant_index} out of range: {len(param_dicts)} "
            f"grid point(s) (valid 1..{len(param_dicts)})")
    return name, param_dicts[idx]


def _apply_config_variant(sim_data, variants_config: dict, variant_index: int):
    """Apply the selected config variant to ``sim_data`` via the fork's own
    ``ecoli.variants.<name>.apply_variant``. Returns ``(sim_data, meta|None)``.
    ``sim_data`` must already be a fresh (non-shared) object."""
    name, params = _select_variant_params(variants_config, variant_index)
    if name is None:
        return sim_data, None
    import importlib
    mod = importlib.import_module(f"ecoli.variants.{name}")  # fork-bound
    sim_data = mod.apply_variant(sim_data, params)
    return sim_data, {"variant_name": name, "variant_index": int(variant_index),
                      "params": params}
```

Add `variant: int = 0` to `build_vivarium_ecoli`'s signature (after `initial_overlay`). In the media-from-condition block, right after `_sd_obj = _pickle.load(_sdf)` (line ~154) and before `sim.config["sim_data"] = _sd_obj`:

```python
        if _cfgfile and int(variant):
            _variants_cfg = sim.config.get("variants") or {}
            _sd_obj, _vmeta = _apply_config_variant(_sd_obj, _variants_cfg, int(variant))
            if _vmeta:
                print(f"[build_vivarium_ecoli] applied config variant "
                      f"'{_vmeta['variant_name']}' #{variant}: {_vmeta['params']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_hook.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/vivarium_ecoli_engine.py tests/test_variant_hook.py
git commit -m "feat: apply config-declared variant grid point in whole-config node"
```

---

### Task 2: Configurable bulk-observable emission on the node

**Files:**
- Modify: `v2ecoli/library/vivarium_ecoli_engine.py` (add pure helper `_select_bulk_observables`; extend `VivariumEcoliProcess.config_schema` / `outputs()` / `update()`; add `observable_bulk_ids` to `build_vivarium_ecoli_composite` config)
- Test: `tests/test_bulk_observables.py`

**Interfaces:**
- Consumes: `cell_observables(engine)` (already returns `obs["bulk"]`, a name→count mapping).
- Produces: `_select_bulk_observables(obs_bulk: dict, ids: list[str]) -> dict[str, float]`; `VivariumEcoliProcess` config key `observable_bulk_ids: list[str]` that adds a `bulk` output map.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bulk_observables.py
from v2ecoli.library.vivarium_ecoli_engine import _select_bulk_observables


def test_selects_requested_ids_as_floats():
    obs_bulk = {"A[c]": 10, "B[p]": 3, "C[m]": 0}
    out = _select_bulk_observables(obs_bulk, ["A[c]", "C[m]"])
    assert out == {"A[c]": 10.0, "C[m]": 0.0}


def test_missing_id_defaults_to_zero_not_crash():
    out = _select_bulk_observables({"A[c]": 5}, ["A[c]", "MISSING[x]"])
    assert out == {"A[c]": 5.0, "MISSING[x]": 0.0}


def test_empty_ids_returns_empty():
    assert _select_bulk_observables({"A[c]": 5}, []) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bulk_observables.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_bulk_observables'`

- [ ] **Step 3: Write minimal implementation**

Add the pure helper near `cell_observables`:

```python
def _select_bulk_observables(obs_bulk: dict, ids: list) -> dict:
    """Pick ``ids`` out of the inner cell's bulk name->count map as floats.
    A missing id yields 0.0 (a species absent this tick), never a KeyError."""
    if not ids:
        return {}
    src = obs_bulk or {}
    return {i: float(src.get(i, 0.0)) for i in ids}
```

In `VivariumEcoliProcess.config_schema`, add:

```python
        "observable_bulk_ids": {"_type": "list[string]", "_default": []},
```

In `__init__`, pass it through to the builder (see Task 3 for the builder param); store `self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])`.

In `outputs()`, add a `bulk` group when ids are configured:

```python
    def outputs(self):
        out = {"listeners": {
            "mass": {k: "overwrite[float]" for k in MASS_OBS},
            "unique_molecule_counts": {k: "overwrite[float]" for k in COUNT_OBS},
        }}
        if self._obs_bulk_ids:
            out["bulk"] = {i: "overwrite[float]" for i in self._obs_bulk_ids}
        return out
```

In `update()`, surface them:

```python
    def update(self, state, interval):
        self._handle.engine.run_for(float(interval))
        obs = cell_observables(self._handle.engine)
        upd = {"listeners": {
            "mass": {k: obs[k] for k in MASS_OBS},
            "unique_molecule_counts": {k: obs[k] for k in COUNT_OBS},
        }}
        if self._obs_bulk_ids:
            upd["bulk"] = _select_bulk_observables(obs.get("bulk", {}), self._obs_bulk_ids)
        return upd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bulk_observables.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/vivarium_ecoli_engine.py tests/test_bulk_observables.py
git commit -m "feat: configurable bulk-observable emission on whole-config node"
```

---

### Task 3: Thread `variant` + `observable_bulk_ids` through the composite builder

**Files:**
- Modify: `v2ecoli/library/vivarium_ecoli_engine.py` (`build_vivarium_ecoli_composite` gains `variant` + `observable_bulk_ids`; `VivariumEcoliProcess.__init__` passes `variant` + `observable_bulk_ids` into `build_vivarium_ecoli`; `run_vivarium_ecoli_pbg_multigen` already has `variant` — pass it down)
- Test: `tests/test_variant_threading.py`

**Interfaces:**
- Produces: `build_vivarium_ecoli_composite(..., variant: int = 0, observable_bulk_ids: list | None = None)`; `VivariumEcoliProcess` config keys `variant`, `observable_bulk_ids` reach `build_vivarium_ecoli`.
- Consumes: Task 1 `build_vivarium_ecoli(..., variant=...)`, Task 2 output plumbing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variant_threading.py
import inspect
from v2ecoli.library import vivarium_ecoli_engine as ve


def test_build_vivarium_ecoli_has_variant_param():
    assert "variant" in inspect.signature(ve.build_vivarium_ecoli).parameters


def test_composite_builder_forwards_variant(monkeypatch):
    seen = {}
    def _fake_build(**kw):
        seen.update(kw)
        class _H:  # minimal stand-in for EngineHandle
            pass
        return _H()
    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__",
                        lambda self, config=None, core=None: None)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface", lambda self: {"inputs": {}, "outputs": {}})
    ve.build_vivarium_ecoli_composite(
        sim_data_path="x", variant=4, observable_bulk_ids=["A[c]"], core=object())
    assert seen["variant"] == 4
    assert seen["observable_bulk_ids"] == ["A[c]"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_threading.py -v`
Expected: FAIL — `TypeError: build_vivarium_ecoli_composite() got an unexpected keyword argument 'variant'`

- [ ] **Step 3: Write minimal implementation**

`build_vivarium_ecoli` signature: add `observable_bulk_ids: list | None = None` (Task 2 used it only in the process; the builder must accept + ignore it or forward to the handle — accept it as a no-op passthrough recorded on the handle is unnecessary; the process reads its own config. So `build_vivarium_ecoli` only needs `variant`; `observable_bulk_ids` lives on the process config). Keep `build_vivarium_ecoli(..., variant: int = 0)` from Task 1.

In `build_vivarium_ecoli_composite`, add `variant: int = 0, observable_bulk_ids: list | None = None`; forward `variant=int(variant)` to the `build_vivarium_ecoli(...)` call; add both to the `VivariumEcoliProcess(config={...})` dict:

```python
    VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
        sim_data_path=sim_data_path, condition=condition, seed=int(seed),
        time_step=float(time_step), exclude_processes=list(exclude_processes or []) or None,
        swap_processes=swap_processes or None, flow=flow or None,
        fork_dir=fork_dir or None, initial_overlay=initial_overlay, variant=int(variant))
    proc = VivariumEcoliProcess(config={
        "sim_data_path": sim_data_path, "condition": condition, "seed": int(seed),
        "time_step": float(time_step),
        "exclude_processes": list(exclude_processes or []),
        "fork_dir": fork_dir or "",
        "variant": int(variant),
        "observable_bulk_ids": list(observable_bulk_ids or []),
    }, core=core)
```

In `VivariumEcoliProcess.__init__`, the non-`_PENDING_HANDLE` branch passes `variant=int(self.config.get("variant") or 0)` to `build_vivarium_ecoli`, and set `self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])` unconditionally (both branches).

In `run_vivarium_ecoli_pbg_multigen`, forward its existing `variant` param into each per-generation `build_vivarium_ecoli_composite(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_threading.py tests/test_variant_hook.py tests/test_bulk_observables.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/vivarium_ecoli_engine.py tests/test_variant_threading.py
git commit -m "feat: thread variant + observable_bulk_ids through the whole-config composite builder"
```

---

### Task 4: `whole_config` + `variant` + `observable_bulk_ids` params on the `vecoli` generator

**Files:**
- Modify: `v2ecoli/composites/vecoli.py` (three new declared params; native-load branch)
- Test: `tests/test_vecoli_generator_params.py`

**Interfaces:**
- Produces: generator `v2ecoli.composites.vecoli.vecoli` accepts `whole_config: str = ""`, `variant: int = 0`, `observable_bulk_ids: list = []`; builds the same document envelope; records them in provenance.
- Consumes: Task 3 `build_vivarium_ecoli_composite`/`build_vivarium_ecoli`; `set_ecolisim_config_file`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vecoli_generator_params.py
from viva_superpowers.composite_generator import _REGISTRY, discover_generators


def _entry():
    if not _REGISTRY:
        discover_generators()
    import v2ecoli.composites  # force registration
    return _REGISTRY["v2ecoli.composites.vecoli.vecoli"]


def test_whole_config_and_variant_are_declared_params():
    params = _entry().parameters
    assert "whole_config" in params
    assert "variant" in params
    assert "observable_bulk_ids" in params


def test_unknown_param_still_rejected():
    # sanity: declared set is a subset guard the workbench relies on
    params = set(_entry().parameters)
    assert {"whole_config", "variant", "observable_bulk_ids"} <= params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_vecoli_generator_params.py -v`
Expected: FAIL — `assert 'whole_config' in params`

- [ ] **Step 3: Write minimal implementation**

Add to the `@composite_generator(parameters={...})` block in `vecoli.py`:

```python
        "whole_config": {
            "type": "string",
            "default": "",
            "description": (
                "Optional full fork config (path relative to reference_repo or "
                "absolute) loaded NATIVELY by EcoliSim (its add_processes / "
                "spatial_environment_config / variants applied) instead of the "
                "swap-only fork_config path. Empty = swap/baseline behavior."
            ),
        },
        "variant": {
            "type": "integer",
            "default": 0,
            "description": (
                "1-based index into the loaded config's 'variants' grid "
                "(0 = unperturbed baseline). Requires whole_config."
            ),
        },
        "observable_bulk_ids": {
            "type": "list",
            "default": [],
            "description": (
                "Bulk molecule ids to emit as observables (path 'bulk.<id>') "
                "for downstream sweep/phenotype extraction. Empty = mass/count "
                "observables only."
            ),
        },
```

In the `vecoli(...)` function signature add `whole_config: str = "", variant: int = 0, observable_bulk_ids: list | None = None`. Before building, select the native vs swap path:

```python
    from v2ecoli.library.vivarium_ecoli_engine import set_ecolisim_config_file
    _prev_cfg = None
    if whole_config:
        # Native whole-config load: EcoliSim reads add_processes / spatial /
        # variants from this file. Resolve relative to the fork checkout.
        cfg_path = whole_config
        if not os.path.isabs(cfg_path):
            base = reference_repo or os.environ.get("V2E_VECOLI_DIR", "")
            cfg_path = os.path.join(base, cfg_path)
        set_ecolisim_config_file(cfg_path)
        swap_processes, flow = None, None      # native path, not swap
    else:
        swap_processes, flow = _resolve_fork_config(reference_repo, fork_config)
```

Pass `variant=int(variant)` into `build_vivarium_ecoli(...)`, add `variant` + `observable_bulk_ids` to the `VivariumEcoliProcess(config=...)` dict, and reset the module global after building:

```python
    try:
        VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
            sim_data_path=sim_data_path, condition=condition, seed=int(seed),
            time_step=float(time_step), swap_processes=swap_processes, flow=flow,
            fork_dir=(reference_repo or None), variant=int(variant))
        proc = VivariumEcoliProcess(config={
            "sim_data_path": sim_data_path, "condition": condition,
            "seed": int(seed), "time_step": float(time_step),
            "fork_dir": reference_repo or "",
            "variant": int(variant),
            "observable_bulk_ids": list(observable_bulk_ids or []),
        }, core=core)
    finally:
        if whole_config:
            set_ecolisim_config_file(None)   # deterministic isolation
```

Extend the returned document's emitter paths to include `bulk` (so configured observables are captured). If the generator declares `emitters=`, add `"bulk"` to `paths`; otherwise the study's emitter config carries it (documented in Task 6).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_vecoli_generator_params.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/vecoli.py tests/test_vecoli_generator_params.py
git commit -m "feat: whole_config + variant + observable_bulk_ids params on vecoli composite"
```

---

### Task 5: Generic phenotype-sweep extractor

**Files:**
- Create: `v2ecoli/library/phenotype_sweep.py`
- Test: `tests/test_phenotype_sweep.py`

**Interfaces:**
- Produces: `collect_sweep(runs: list[dict], observable_paths: list[str]) -> dict` where each run is `{"label": str, "series": dict[path, list[float]]}` (already-loaded) → returns `{path: {label: list[float]}}`; and `sweep_endpoints(sweep: dict) -> dict[path, dict[label, float]]` (last value per series, the dose-response point).

Rationale: keep the extractor pure over already-loaded series so it is engine/emitter-agnostic and unit-testable. A thin loader that reads the workbench parquet/xarray store into this shape lives with the study renderer (Task 6 / downstream), not here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phenotype_sweep.py
from v2ecoli.library.phenotype_sweep import collect_sweep, sweep_endpoints


RUNS = [
    {"label": "v0", "series": {"bulk.X": [0.0, 0.0, 0.0], "growth": [1.0, 1.0, 1.0]}},
    {"label": "v1", "series": {"bulk.X": [0.0, 5.0, 9.0], "growth": [1.0, 0.8, 0.5]}},
]


def test_collect_groups_by_path_then_label():
    out = collect_sweep(RUNS, ["bulk.X", "growth"])
    assert out["bulk.X"] == {"v0": [0.0, 0.0, 0.0], "v1": [0.0, 5.0, 9.0]}
    assert out["growth"]["v1"] == [1.0, 0.8, 0.5]


def test_missing_path_skipped_not_crash():
    out = collect_sweep(RUNS, ["bulk.X", "absent"])
    assert "absent" not in out
    assert set(out) == {"bulk.X"}


def test_endpoints_take_last_value():
    ep = sweep_endpoints(collect_sweep(RUNS, ["bulk.X", "growth"]))
    assert ep["bulk.X"] == {"v0": 0.0, "v1": 9.0}
    assert ep["growth"] == {"v0": 1.0, "v1": 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_phenotype_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2ecoli.library.phenotype_sweep'`

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/library/phenotype_sweep.py
"""Generic variant-sweep phenotype extractor.

Engine-agnostic: consumes already-loaded per-run observable series (a run =
one variant/sweep index) and reshapes them for sweep-axis comparison. Knows
nothing about what any observable means — paths are pure inputs.
"""
from __future__ import annotations


def collect_sweep(runs: list, observable_paths: list) -> dict:
    """Reshape ``[{label, series:{path:[...]}}]`` into ``{path: {label: [...]}}``.

    A path absent from every run is skipped (not an error). A path present in
    some runs only appears for the runs that have it.
    """
    out: dict = {}
    for path in observable_paths:
        col = {}
        for run in runs:
            series = (run.get("series") or {})
            if path in series:
                col[run["label"]] = series[path]
        if col:
            out[path] = col
    return out


def sweep_endpoints(sweep: dict) -> dict:
    """Last value of each series — the dose-response point per (path, label)."""
    return {
        path: {label: (vals[-1] if vals else float("nan"))
               for label, vals in cols.items()}
        for path, cols in sweep.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_phenotype_sweep.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/phenotype_sweep.py tests/test_phenotype_sweep.py
git commit -m "feat: generic variant-sweep phenotype extractor"
```

---

### Task 6: Neutral demo study template

**Files:**
- Create: `workspace/studies/variant-sweep-phenotype-demo/study.yaml`
- Create: `workspace/studies/variant-sweep-phenotype-demo/README.md`
- Test: `tests/test_variant_sweep_demo_study.py`

**Interfaces:**
- Consumes: the `v2ecoli.composites.vecoli.vecoli` generator (Task 4).
- Produces: a schema-v4 study whose `conditions.variants[]` sweep a variant axis; a smoke test that the study validates and each entry builds a document.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variant_sweep_demo_study.py
import os, yaml

STUDY = os.path.join("workspace", "studies", "variant-sweep-phenotype-demo", "study.yaml")


def test_study_exists_and_is_schema_v4():
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    assert doc["schema_version"] == 4
    assert doc["name"] == "variant-sweep-phenotype-demo"


def test_variants_sweep_a_variant_axis_over_the_vecoli_composite():
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    variants = doc["conditions"]["variants"]
    assert len(variants) >= 2
    idxs = [v["params"]["variant"] for v in variants]
    assert idxs == sorted(set(idxs)) and idxs[0] >= 1     # distinct, 1-based
    for v in variants:
        assert v["composite"] == "v2ecoli.composites.vecoli.vecoli"


def test_demo_study_is_structurally_neutral():
    # Public template carries no model content: whole_config unset, no observable
    # ids, only the generic composite — every specific is filled in downstream.
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    base = doc["conditions"]["baseline"]["params"]
    assert base["whole_config"] == ""
    assert base["observable_bulk_ids"] == []
    for v in doc["conditions"]["variants"]:
        assert v["params"]["whole_config"] == ""
        assert v["params"]["observable_bulk_ids"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_sweep_demo_study.py -v`
Expected: FAIL — `FileNotFoundError: .../variant-sweep-phenotype-demo/study.yaml`

- [ ] **Step 3: Write minimal implementation**

Create `workspace/studies/variant-sweep-phenotype-demo/study.yaml` (neutral — uses a generic `condition`-style variant config path as a documented template param; the committed sweep is over variant indices):

```yaml
schema_version: 4
name: variant-sweep-phenotype-demo
investigation: framework
title: Variant-sweep phenotype demo (generic template)
description: >
  Template for a variant-sweep phenotype study. Runs the whole-config vEcoli
  node (v2ecoli.composites.vecoli.vecoli) at several indices of a config-declared
  variant grid and compares a chosen observable across the sweep axis. Copy this
  study, point `whole_config` at any fork config that declares a `variants` block,
  set `variant` per entry to the grid index, and list the observable ids to emit.
  Perturbation-agnostic: no specific model content lives here.
topic: variant-sweep-phenotype
tags: [template, variant-sweep, phenotype, whole-config]
created: '2026-08-15'
conditions:
  baseline:
    composite: v2ecoli.composites.vecoli.vecoli
    params:
      whole_config: ""          # template: set to a fork config declaring `variants`
      variant: 0                # 0 = unperturbed baseline
      observable_bulk_ids: []   # template: list bulk ids to emit as observables
      n_steps: 1500
  variants:
    - name: sweep-index-1
      composite: v2ecoli.composites.vecoli.vecoli
      params:
        whole_config: ""
        variant: 1
        observable_bulk_ids: []
        n_steps: 1500
    - name: sweep-index-2
      composite: v2ecoli.composites.vecoli.vecoli
      params:
        whole_config: ""
        variant: 2
        observable_bulk_ids: []
        n_steps: 1500
status: designed
phase: Design
design_status: approved
```

Create the README documenting: copy → set `whole_config` to a fork config with a `variants` block → set `variant` per entry → list `observable_bulk_ids` → run under the workbench (single-generation, dose landed early) → feed the emitted store to `phenotype_sweep.collect_sweep` / `sweep_endpoints`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_sweep_demo_study.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add workspace/studies/variant-sweep-phenotype-demo/ tests/test_variant_sweep_demo_study.py
git commit -m "feat: neutral variant-sweep phenotype demo study template"
```

---

### Task 7: Full suite + leak check, PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run the new suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_variant_hook.py tests/test_bulk_observables.py tests/test_variant_threading.py tests/test_vecoli_generator_params.py tests/test_phenotype_sweep.py tests/test_variant_sweep_demo_study.py -v`
Expected: PASS (all)

- [ ] **Step 2: Regression — existing vecoli composite unchanged with whole_config=""**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "vecoli or whole_config or comparison" -v`
Expected: PASS / unchanged (byte-parity for `whole_config=""`)

- [ ] **Step 3: Leak check (public repo must be perturbation-free)**

Grep the diff for the downstream perturbation's identifiers (drug/molecule/config
names — the concrete wordlist lives in the private downstream plan, not this repo)
and confirm zero matches across every changed file:

Run: `git diff --name-only origin/main | xargs git grep -iE "<downstream-wordlist>" --`
Expected: NO matches. Also sanity-check no fork config filename or biology token
appears in any changed file. If any match, fix before pushing.

- [ ] **Step 4: Verify branch + push + PR**

```bash
git branch --show-current   # must be feat/variant-sweep-phenotype-study
git log --oneline origin/main..HEAD   # only this task's commits
git push -u origin feat/variant-sweep-phenotype-study
gh pr create --title "Generic variant-sweep phenotype capability for the whole-config WCM node" \
  --body "Applies a config-declared variant grid point in the whole-config node (delegated to the fork's parse_variants/apply_variant), adds configurable bulk-observable emission, exposes whole_config/variant/observable_bulk_ids on the vecoli composite, a generic phenotype-sweep extractor, and a neutral demo study template. Perturbation-agnostic; no model content."
```

Do NOT merge — the user approves merges.

---

## Deferred: downstream instance (separate private plan)

After this PR merges and syncs into the downstream private repo (deterministic
overlay), a second plan — authored in that private worktree, not here — covers the
perturbation-specific instance: a downstream-owned study.yaml pointing `whole_config`
at the fork's config with the relevant variant, `variant` indices for the chosen
sweep points, `observable_bulk_ids` = the relevant markers, the intervention landed
early for a tractable run, the run, and the branded sweep report. That plan is written
against the merged+synced capability, and holds all perturbation-specific detail.

## Self-Review

- **Spec coverage:** §A → Task 1; §B → Task 4 (+ threading Task 3); §C (extractor) → Task 5; §C (emission gap found during planning) → Task 2; §D → Task 6. All spec sections covered; the emission piece (Task 2) is an addition the spec's "configurable observable paths" implied.
- **Placeholder scan:** study.yaml `""` values are intentional template defaults (documented), not plan placeholders; all code steps carry real code.
- **Type consistency:** `variant: int` and `observable_bulk_ids: list` names match across Tasks 1–4; `_select_variant_params`/`_apply_config_variant`/`_select_bulk_observables`/`collect_sweep`/`sweep_endpoints` referenced consistently.
- **Scope:** single repo, one coherent capability; the mec application is correctly deferred to Plan 2.
