# v2ecoli Workflow Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run vEcoli-style `variants × seeds × generations` sweeps in v2ecoli as a single inspectable process-bigraph meta-composite, driven by ported vEcoli JSON configs, with partitioned parquet emission.

**Architecture:** A `MetaComposite` document holds one `LineageProcess` node per `(variant, seed)` branch under `state.branches[...]`. Each `LineageProcess` wraps a `baseline` cell composite (the proven `EcoliWCM` embedding pattern), runs it generation-by-generation (carrying one daughter forward — vEcoli's `single_daughters=true` default), applies declarative variant config-overrides at build time, and emits partitioned parquet with per-generation metadata. A config loader ports vEcoli's `inherit_from` + variant grammar; a CLI driver loops `composite.run(dt)` until all branches report `complete`. Analyses are process-bigraph Steps with a five-scale registry (framework + one example here).

**Tech Stack:** Python 3.11+, `process-bigraph`, `bigraph-schema`, `pbg_emitters` (parquet), `numpy`, `pytest`. All new code lives under `v2ecoli/workflow/`, `v2ecoli/steps/`, and `v2ecoli/configs/`.

**Refinements vs. the design spec (intentional, see §Architecture):**
- The spec's standalone `GenerationController` Step is realized as the branch-wrapping `LineageProcess` (each branch must be a wrapped sub-composite to own its `flow_order`/division — the `EcoliWCM` pattern).
- MVP implements the **single-lineage** walk only (`single_daughters=true`). `single_daughters=false` (binary tree) raises `NotImplementedError` and is deferred.
- Variant grammar MVP supports `value` / `linspace` + `op: prod|zip|add` with a required `target` per parameter. `nested` is deferred (clear error).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `v2ecoli/workflow/__init__.py` | Package marker; re-export `load_config`, `expand_branches`, `build_meta_composite`, `run_workflow`. |
| `v2ecoli/workflow/config.py` | Port of vEcoli `load_config_with_inheritance` + `_merge_configs`; `LIST_KEYS_TO_MERGE`. |
| `v2ecoli/workflow/variants.py` | Variant grammar (`parse_variant_params`) + `expand_branches(config) -> list[BranchSpec]`. |
| `v2ecoli/workflow/lineage.py` | `LineageProcess` — wraps a baseline composite, runs one `(variant, seed)` lineage. |
| `v2ecoli/workflow/meta_composite.py` | `build_meta_composite(config) -> dict` document; `register_workflow_processes(core)`. |
| `v2ecoli/workflow/run.py` | `run_workflow(...)` driver loop + `main()` CLI entry. |
| `v2ecoli/workflow/analysis.py` | `AnalysisStep` base, five-scale `ANALYSIS_SCALES` registry, one example Step. |
| `v2ecoli/configs/default.json` | Ported baseline config (sweep defaults). |
| `v2ecoli/configs/two_generations.json` | Ported example: 2 seeds × 2 generations. |
| `tests/test_workflow_config.py` | Inheritance + merge semantics. |
| `tests/test_workflow_variants.py` | Variant grammar + branch-grid expansion. |
| `tests/test_baseline_overrides.py` | `baseline(config_overrides=...)` patches process config without corrupting cache. |
| `tests/test_meta_composite_build.py` | Grid → branch nodes; addresses resolve. |
| `tests/test_workflow_analysis.py` | `AnalysisStep` registry + example Step output. |
| `tests/test_workflow_smoke.py` | Tiny config (1 variant, 1 seed, 1 gen) builds + runs + emits (cache-gated). |

**Existing files modified:** `v2ecoli/composites/baseline.py` (add `config_overrides` param); `pyproject.toml` (add `v2ecoli-workflow` console script).

---

## Task 1: Config loader with inheritance

**Files:**
- Create: `v2ecoli/workflow/__init__.py`
- Create: `v2ecoli/workflow/config.py`
- Test: `tests/test_workflow_config.py`

- [ ] **Step 1: Create the empty package marker**

Create `v2ecoli/workflow/__init__.py`:

```python
"""v2ecoli workflow framework: meta-composite variants × seeds × generations sweeps."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_workflow_config.py`:

```python
import json
from v2ecoli.workflow.config import load_config_with_inheritance, _merge_configs


def test_merge_overlay_wins(tmp_path):
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    overlay = {"a": 2, "nested": {"y": 3}}
    _merge_configs(base, overlay)
    assert base["a"] == 2
    assert base["nested"] == {"x": 1, "y": 3}


def test_inheritance_priority(tmp_path):
    (tmp_path / "C.json").write_text(json.dumps({"v": "C", "only_c": 1}))
    (tmp_path / "B.json").write_text(json.dumps({"inherit_from": ["C.json"], "v": "B"}))
    (tmp_path / "D.json").write_text(json.dumps({"v": "D", "only_d": 1}))
    (tmp_path / "A.json").write_text(
        json.dumps({"inherit_from": ["B.json", "D.json"], "v": "A"}))
    cfg = load_config_with_inheritance(str(tmp_path / "A.json"), config_dir=str(tmp_path))
    # Priority A > B > C > D
    assert cfg["v"] == "A"
    assert cfg["only_c"] == 1
    assert cfg["only_d"] == 1


def test_list_keys_merge_and_dedup(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"add_processes": ["p1", "p2"]}))
    (tmp_path / "top.json").write_text(
        json.dumps({"inherit_from": ["base.json"], "add_processes": ["p2", "p3"]}))
    cfg = load_config_with_inheritance(str(tmp_path / "top.json"), config_dir=str(tmp_path))
    assert cfg["add_processes"] == ["p1", "p2", "p3"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.config'`

- [ ] **Step 4: Write the implementation**

Create `v2ecoli/workflow/config.py`:

```python
"""Config loading with vEcoli-style inheritance.

Ported from vEcoli runscripts/workflow.py:load_config_with_inheritance +
_merge_configs. ``inherit_from`` chains resolve with priority
current > first-inherited > ... > last-inherited.
"""

from __future__ import annotations

import json
import os

# Keys whose list values are concatenated + deduplicated across the chain
# rather than overwritten (subset of vEcoli's; extend as configs need it).
LIST_KEYS_TO_MERGE = {
    "save_times",
    "add_processes",
    "exclude_processes",
    "engine_process_reports",
}


def load_config_with_inheritance(config_path: str, config_dir: str | None = None) -> dict:
    """Load a config file, recursively resolving ``inherit_from`` chains.

    ``config_dir`` is the directory inherited paths are resolved against
    (defaults to the directory of ``config_path``).
    """
    with open(config_path) as f:
        config = json.load(f)

    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))

    if "inherit_from" not in config:
        return config

    inherit_chain = []
    for inherit_path in reversed(config["inherit_from"]):
        inherited = load_config_with_inheritance(
            os.path.join(config_dir, inherit_path), config_dir=config_dir)
        inherit_chain.append(inherited)

    result: dict = {}
    for inherited_config in inherit_chain:
        _merge_configs(result, inherited_config)
    _merge_configs(result, config)
    result.pop("inherit_from", None)
    return result


def _merge_configs(base_config: dict, overlay_config: dict) -> None:
    """Merge ``overlay_config`` into ``base_config`` in place (overlay wins)."""
    for key, value in overlay_config.items():
        if key in LIST_KEYS_TO_MERGE:
            base_config.setdefault(key, [])
            base_config[key].extend(value)
            base_config[key] = sorted(set(base_config[key]))
        elif (
            isinstance(value, dict)
            and key in base_config
            and isinstance(base_config[key], dict)
        ):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/__init__.py v2ecoli/workflow/config.py tests/test_workflow_config.py
git commit -m "feat(workflow): config loader with inherit_from chains"
```

---

## Task 2: Variant grammar & branch expansion

**Files:**
- Create: `v2ecoli/workflow/variants.py`
- Test: `tests/test_workflow_variants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_variants.py`:

```python
from v2ecoli.workflow.variants import parse_variant_params, expand_branches, BranchSpec


def test_single_param_value():
    params = parse_variant_params({"kcat": {"target": "ecoli-metabolism.kcat",
                                             "value": [1, 2, 3]}})
    assert params == [
        {"ecoli-metabolism.kcat": 1},
        {"ecoli-metabolism.kcat": 2},
        {"ecoli-metabolism.kcat": 3},
    ]


def test_linspace_param():
    params = parse_variant_params({"d": {"target": "p.q",
                                         "linspace": {"start": 0.0, "stop": 1.0, "num": 3}}})
    assert [round(list(p.values())[0], 3) for p in params] == [0.0, 0.5, 1.0]


def test_prod_of_two_params():
    params = parse_variant_params({
        "a": {"target": "p.a", "value": [1, 2]},
        "b": {"target": "p.b", "value": [10, 20]},
        "op": "prod",
    })
    assert {"p.a": 1, "p.b": 10} in params
    assert {"p.a": 2, "p.b": 20} in params
    assert len(params) == 4


def test_expand_branches_grid():
    config = {
        "n_init_sims": 2,
        "lineage_seed": 0,
        "variants": {"kcat": {"target": "ecoli-metabolism.kcat", "value": [1, 2]}},
        "skip_baseline": True,
    }
    branches = expand_branches(config)
    # 2 variants × 2 seeds = 4 branches
    assert len(branches) == 4
    assert all(isinstance(b, BranchSpec) for b in branches)
    seeds = sorted({b.seed for b in branches})
    assert seeds == [0, 1]
    overrides = {tuple(sorted(b.overrides.items())) for b in branches}
    assert (("ecoli-metabolism.kcat", 1),) in overrides
    assert (("ecoli-metabolism.kcat", 2),) in overrides


def test_expand_branches_baseline_included_by_default():
    config = {"n_init_sims": 1, "variants": {}}
    branches = expand_branches(config)
    assert len(branches) == 1
    assert branches[0].variant_name == "baseline"
    assert branches[0].overrides == {}


def test_different_seeds_per_variant_non_overlapping():
    config = {
        "n_init_sims": 2,
        "lineage_seed": 0,
        "different_seeds_per_variant": True,
        "variants": {"kcat": {"target": "m.k", "value": [1, 2]}},
        "skip_baseline": True,
    }
    branches = expand_branches(config)
    by_variant = {}
    for b in branches:
        by_variant.setdefault(b.variant_index, set()).add(b.seed)
    seed_sets = list(by_variant.values())
    # No seed shared between the two variants
    assert seed_sets[0].isdisjoint(seed_sets[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_variants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.variants'`

- [ ] **Step 3: Write the implementation**

Create `v2ecoli/workflow/variants.py`:

```python
"""Variant grammar → declarative config-override branch specs.

Adapts vEcoli's parse_variants (runscripts/create_variants.py). Each variant
parameter declares a ``target`` path (``"<process-name>.<config-key>"``) plus
exactly one value source: ``value`` (a list) or a numpy generator such as
``linspace`` ({start, stop, num}). Multiple parameters combine via top-level
``op``: ``prod`` (cartesian), ``zip`` (elementwise), ``add`` (concatenate).
``nested`` is not supported in MVP.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BranchSpec:
    variant_index: int
    variant_name: str
    overrides: dict[str, Any]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_variant_params(variant_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one variant block into a list of ``{target_path: value}`` dicts."""
    variant_config = dict(variant_config)  # don't mutate caller's dict
    operation = None
    if sum(1 for k in variant_config if k != "op") > 1:
        if "op" not in variant_config:
            raise ValueError("Variant has >1 parameter but no 'op' key.")
        operation = variant_config.pop("op")
    elif "op" in variant_config:
        raise ValueError("Single-parameter variant must not define 'op'.")

    parsed: dict[str, list[Any]] = {}
    targets: dict[str, str] = {}
    for param_name, param_conf in variant_config.items():
        param_conf = dict(param_conf)
        target = param_conf.pop("target", None)
        if target is None:
            raise ValueError(f"variant param {param_name!r} missing 'target'.")
        targets[param_name] = target
        if len(param_conf) != 1:
            raise ValueError(f"variant param {param_name!r} needs exactly one value source.")
        ptype, pvals = next(iter(param_conf.items()))
        if ptype == "value":
            if not isinstance(pvals, list):
                raise ValueError(f"{param_name!r} 'value' must be a list.")
            parsed[param_name] = pvals
        elif ptype == "nested":
            raise NotImplementedError("nested variants are deferred (MVP).")
        else:
            try:
                np_func = getattr(np, ptype)
            except AttributeError as e:
                raise ValueError(f"{param_name!r} unknown value source {ptype!r}.") from e
            parsed[param_name] = np_func(**pvals).tolist()

    names = list(parsed.keys())
    if operation == "prod":
        combos = itertools.product(*(parsed[k] for k in names))
        dicts = [dict(zip(names, combo)) for combo in combos]
    elif operation == "zip":
        n = len(parsed[names[0]])
        for k in names:
            if len(parsed[k]) != n:
                raise ValueError("zip requires equal-length parameters.")
        dicts = [{k: parsed[k][i] for k in names} for i in range(n)]
    elif operation == "add":
        dicts = []
        for k in names:
            dicts.extend({k: v} for v in parsed[k])
    elif operation is None:
        k = names[0]
        dicts = [{k: v} for v in parsed[k]]
    else:
        raise ValueError(f"Unknown op {operation!r}.")

    # Re-key by target path.
    return [{targets[name]: val for name, val in d.items()} for d in dicts]


def expand_branches(config: dict[str, Any]) -> list[BranchSpec]:
    """Cross the variant grid with the seed range into a flat branch list."""
    n_init_sims = int(config.get("n_init_sims", 1))
    lineage_seed = int(config.get("lineage_seed", 0))
    skip_baseline = bool(config.get("skip_baseline", False))
    different_seeds = bool(config.get("different_seeds_per_variant", False))

    variants_block = config.get("variants") or {}

    # Build the ordered list of (variant_name, overrides) entries.
    variant_entries: list[tuple[str, dict[str, Any]]] = []
    if not skip_baseline:
        variant_entries.append(("baseline", {}))
    for vname, vconf in variants_block.items():
        for overrides in parse_variant_params({vname: vconf} if "target" in vconf
                                              else dict(vconf)):
            variant_entries.append((vname, overrides))

    branches: list[BranchSpec] = []
    for v_idx, (vname, overrides) in enumerate(variant_entries):
        if different_seeds:
            base = lineage_seed + v_idx * n_init_sims
        else:
            base = lineage_seed
        for s in range(n_init_sims):
            branches.append(BranchSpec(
                variant_index=v_idx,
                variant_name=vname,
                overrides=dict(overrides),
                seed=base + s,
                metadata={"variant_name": vname, **{f"override:{k}": v
                                                    for k, v in overrides.items()}},
            ))
    return branches
```

Note: the `{vname: vconf} if "target" in vconf else dict(vconf)` branch handles both the single-param shorthand (`"kcat": {"target":..., "value":...}`) and the multi-param form (`"kcat": {"a": {...}, "b": {...}, "op": "prod"}`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_variants.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/variants.py tests/test_workflow_variants.py
git commit -m "feat(workflow): variant grammar + branch-grid expansion"
```

---

## Task 3: `config_overrides` in baseline()

**Files:**
- Modify: `v2ecoli/composites/baseline.py:405-412`
- Modify: `v2ecoli/composites/baseline.py:364-381` (generator parameters + signature)
- Test: `tests/test_baseline_overrides.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baseline_overrides.py`:

```python
import os
import copy
import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def test_override_patches_process_config():
    from v2ecoli.core import build_core, load_cache_bundle
    from v2ecoli.composites.baseline import baseline

    # Pick any process+key actually present in the cached configs.
    bundle = load_cache_bundle(CACHE)
    proc = "ecoli-metabolism"
    cfg = bundle["configs"][proc]
    key = next(k for k, v in cfg.items() if isinstance(v, (int, float)))
    original = copy.deepcopy(cfg[key])
    sentinel = (original if isinstance(original, int) else 0.0)
    sentinel = sentinel + 12345 if isinstance(sentinel, int) else 99999.0

    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir=CACHE,
                   config_overrides={f"{proc}.{key}": sentinel})
    edge = doc["state"]["agents"]["0"][proc]
    assert edge["config"][key] == sentinel

    # The cached bundle must NOT be mutated (lru_cache shares it).
    bundle2 = load_cache_bundle(CACHE)
    assert bundle2["configs"][proc][key] == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && V2ECOLI_CACHE=out/cache python -m pytest tests/test_baseline_overrides.py -v`
Expected: FAIL (`baseline() got an unexpected keyword argument 'config_overrides'`) — or SKIP if no cache; if skipped, set up the cache first (see Task 8 note) or run on a machine with `out/cache`.

- [ ] **Step 3: Add the generator parameter**

In `v2ecoli/composites/baseline.py`, in the `@composite_generator(... parameters={...})` block (around line 367), add a `config_overrides` parameter after `cache_dir`:

```python
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": "Declarative '<process>.<key>': value config overrides (variants)",
        },
```

- [ ] **Step 4: Update the function signature and apply overrides**

Change the signature (line 381) to:

```python
def baseline(core: Any = None, *, seed: int = 0, cache_dir: str = "out/cache",
             config_overrides: dict | None = None) -> dict:
```

Then replace the config-loading block (currently lines 405-407):

```python
    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
```

with:

```python
    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    if config_overrides:
        # Deep-copy before patching: load_cache_bundle returns the cache dict
        # by reference (lru_cache-shared); mutating it would corrupt other runs.
        configs = copy.deepcopy(configs)
        for path, value in config_overrides.items():
            proc, _, key = path.partition(".")
            if not key:
                raise ValueError(f"override path {path!r} must be '<process>.<key>'.")
            if proc not in configs:
                raise KeyError(f"override target process {proc!r} not in cache configs.")
            configs[proc][key] = value
```

(`copy` is already imported at the top of `baseline.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && V2ECOLI_CACHE=out/cache python -m pytest tests/test_baseline_overrides.py -v`
Expected: PASS (or SKIP if cache absent on this machine — note in commit if skipped).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/composites/baseline.py tests/test_baseline_overrides.py
git commit -m "feat(composites): baseline config_overrides for declarative variants"
```

---

## Task 4: `LineageProcess`

**Files:**
- Create: `v2ecoli/workflow/lineage.py`
- Test: covered indirectly by Task 5 (`test_meta_composite_build.py`) and Task 6 (`test_workflow_smoke.py`); this task adds a focused unit test for generation bookkeeping with a stub composite.
- Test: `tests/test_workflow_lineage.py`

- [ ] **Step 1: Write the failing test (generation bookkeeping, no cache)**

Create `tests/test_workflow_lineage.py`:

```python
import pytest
from v2ecoli.workflow.lineage import LineageProcess


def _make(monkeypatch, generations, divide_after=2):
    """Build a LineageProcess whose _build_generation/_run_until_division are
    stubbed so we can test generation counting without a real cell composite."""
    from process_bigraph import Process
    lp = LineageProcess.__new__(LineageProcess)
    # Minimal config + state normally set by Process.__init__/initialize.
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": generations,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0,
    }
    lp.initialize(lp.config)
    calls = {"built": 0}

    def fake_build():
        calls["built"] += 1
        lp._gen_elapsed = 0.0

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        return lp._gen_elapsed >= divide_after, {"bulk": {}, "unique": {}}, 100.0 + lp._generation

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)
    return lp, calls


def test_completes_after_generations(monkeypatch):
    lp, calls = _make(monkeypatch, generations=3, divide_after=2)
    out = {}
    for _ in range(20):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True
    assert len(lp._summaries) == 3            # 3 generations recorded
    assert [s["generation"] for s in lp._summaries] == [0, 1, 2]


def test_single_daughters_false_not_implemented(monkeypatch):
    lp, _ = _make(monkeypatch, generations=2)
    lp.config["single_daughters"] = False
    with pytest.raises(NotImplementedError):
        lp.update({}, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_lineage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.lineage'`

- [ ] **Step 3: Write the implementation**

Create `v2ecoli/workflow/lineage.py`:

```python
"""LineageProcess — one (variant, seed) lineage as an embeddable Process.

Wraps a baseline cell composite (the EcoliWCM embedding pattern) and runs it
generation-by-generation, carrying a single daughter forward (vEcoli's
``single_daughters=true`` default). Variant overrides are applied at build
time; each generation emits partitioned parquet with its own metadata.
The meta-composite ticks this process via update(); it reports ``complete``
when ``generations`` cells have been run.
"""

from __future__ import annotations

from typing import Any

from process_bigraph import Process


class LineageProcess(Process):
    config_schema = {
        "cache_dir": {"_type": "string", "_default": "out/cache"},
        "seed": {"_type": "integer", "_default": 0},
        "lineage_seed": {"_type": "integer", "_default": 0},
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "config_overrides": {"_default": {}},
        "generations": {"_type": "integer", "_default": 1},
        "single_daughters": {"_type": "boolean", "_default": True},
        "experiment_id": {"_type": "string", "_default": "default"},
        "out_dir": {"_type": "string", "_default": "out/workflow"},
        "max_duration_per_gen": {"_type": "float", "_default": 3600.0},
    }

    def initialize(self, config):
        self._composite = None
        self._core = None
        self._generation = 0          # 0-based current generation
        self._agent_id = "0"
        self._gen_elapsed = 0.0
        self._carry_state: dict | None = None
        self._complete = False
        self._summaries: list[dict] = []

    def inputs(self):
        return {}

    def outputs(self):
        return {"summary": "map", "complete": "boolean"}

    # --- build / run helpers (stubbed in unit tests) ---------------------

    def _build_generation(self):
        from process_bigraph import Composite
        from v2ecoli.core import build_core
        from v2ecoli.composites.baseline import baseline, seed_mass_listener
        from v2ecoli.composites._helpers import set_parquet_emitter_override
        from v2ecoli.library.emitter_presets import parquet_vecoli

        core = build_core()
        gen_seed = (int(self.config["seed"]) + self._generation) % (2 ** 31)
        emitter_cfg = parquet_vecoli(
            out_dir=self.config["out_dir"],
            experiment_id=self.config["experiment_id"],
            variant=int(self.config["variant_index"]),
            lineage_seed=int(self.config["lineage_seed"]),
            agent_id=self._agent_id,
            generation=self._generation,
        )
        set_parquet_emitter_override(emitter_cfg)
        try:
            doc = baseline(
                core=core, seed=gen_seed, cache_dir=self.config["cache_dir"],
                config_overrides=dict(self.config.get("config_overrides") or {}))
        finally:
            set_parquet_emitter_override(None)

        if self._carry_state is not None:
            agent = doc["state"]["agents"]["0"]
            for key in ("bulk", "unique", "environment", "boundary"):
                if key in self._carry_state:
                    agent[key] = self._carry_state[key]
            agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
            seed_mass_listener(agent, core)

        self._core = core
        self._composite = Composite(doc, core=core)
        self._gen_elapsed = 0.0

    def _run_until_division(self, interval):
        """Run the internal composite for ``interval`` seconds. Returns
        ``(divided, daughter_cell_data_or_None, final_dry_mass)``."""
        agents_before = set((self._composite.state.get("agents") or {}).keys())
        divided = False
        try:
            self._composite.run(interval)
        except Exception as e:  # process-bigraph surfaces division as _add/_remove
            msg = str(e).lower()
            if "divi" in msg or "_add" in str(e) or "_remove" in str(e):
                divided = True
            else:
                raise
        self._gen_elapsed += interval
        agents_after = set((self._composite.state.get("agents") or {}).keys())
        if agents_before and agents_after != agents_before:
            divided = True

        cell = (self._composite.state.get("agents", {}).get(self._agent_id)
                or next(iter(self._composite.state.get("agents", {}).values()), {}))
        dry_mass = float(cell.get("listeners", {}).get("mass", {}).get("dry_mass", 0.0))

        daughter = None
        if divided:
            from v2ecoli.library.division import divide_cell
            cell_data = {
                "bulk": cell.get("bulk"),
                "unique": cell.get("unique", {}),
                "environment": cell.get("environment", {}),
                "boundary": cell.get("boundary", {}),
            }
            if cell_data["bulk"] is not None:
                d1, _d2 = divide_cell(cell_data)
                daughter = d1
        return divided, daughter, dry_mass

    # --- main tick -------------------------------------------------------

    def update(self, state, interval):
        if not self.config.get("single_daughters", True):
            raise NotImplementedError(
                "single_daughters=False (binary-tree lineage) is deferred; "
                "MVP supports the single-lineage walk only.")
        if self._complete:
            return {"complete": True}
        if self._composite is None:
            self._build_generation()

        divided, daughter, dry_mass = self._run_until_division(interval)
        timed_out = self._gen_elapsed >= float(self.config["max_duration_per_gen"])
        if not (divided or timed_out):
            return {}

        # End of this generation: flush emitter, record summary.
        from v2ecoli.composites._helpers import flush_parquet
        try:
            flush_parquet(self._composite, success=True)
        except Exception:
            pass
        self._summaries.append({
            "generation": self._generation,
            "agent_id": self._agent_id,
            "duration": self._gen_elapsed,
            "dry_mass": dry_mass,
            "divided": bool(divided),
        })

        self._generation += 1
        if self._generation >= int(self.config["generations"]):
            self._complete = True
            self._composite = None
            return {"complete": True, "summary": {"generations": self._summaries}}

        # Carry daughter 0 forward; rebuild a fresh composite next tick.
        self._carry_state = daughter
        self._agent_id = self._agent_id + "0"
        self._composite = None
        return {"summary": {"generations": self._summaries}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_lineage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/lineage.py tests/test_workflow_lineage.py
git commit -m "feat(workflow): LineageProcess — single-lineage generation walk"
```

---

## Task 5: `build_meta_composite` + process registration

**Files:**
- Create: `v2ecoli/workflow/meta_composite.py`
- Test: `tests/test_meta_composite_build.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_meta_composite_build.py`:

```python
from v2ecoli.core import build_core
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)


def test_build_meta_composite_branches():
    config = {
        "experiment_id": "twovar",
        "n_init_sims": 2,
        "generations": 1,
        "single_daughters": True,
        "cache_dir": "out/cache",
        "out_dir": "out/twovar",
        "variants": {"kcat": {"target": "ecoli-metabolism.kcat", "value": [1, 2]}},
        "skip_baseline": True,
    }
    doc = build_meta_composite(config)
    branches = doc["state"]["branches"]
    assert len(branches) == 4  # 2 variants × 2 seeds

    # Each branch holds a LineageProcess node with the right config wiring.
    sample_key = next(iter(branches))
    node = branches[sample_key]["lineage"]
    assert node["_type"] == "process"
    assert node["address"] == "local:LineageProcess"
    assert node["config"]["generations"] == 1
    assert node["config"]["experiment_id"] == "twovar"
    assert "ecoli-metabolism.kcat" in node["config"]["config_overrides"]


def test_register_workflow_processes_resolves_address():
    core = build_core()
    register_workflow_processes(core)
    # local:LineageProcess must resolve after registration.
    from v2ecoli.workflow.lineage import LineageProcess
    assert core.process_registry.access("LineageProcess") is LineageProcess
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_meta_composite_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.meta_composite'`

- [ ] **Step 3: Write the implementation**

Create `v2ecoli/workflow/meta_composite.py`:

```python
"""Build the meta-composite document for a variants × seeds × generations sweep.

One LineageProcess node per (variant, seed) branch lives under
``state.branches[<branch-key>]``. The whole sweep is a single process-bigraph
document, saveable/inspectable via v2ecoli.pbg.save_pbg_doc.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.workflow.variants import expand_branches, BranchSpec


def register_workflow_processes(core) -> None:
    """Register workflow Processes/Steps so ``local:`` addresses resolve."""
    from v2ecoli.workflow.lineage import LineageProcess
    core.register_link("LineageProcess", LineageProcess)


def _branch_key(spec: BranchSpec) -> str:
    return f"variant={spec.variant_index}/seed={spec.seed}"


def _lineage_node(spec: BranchSpec, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lineage": {
            "_type": "process",
            "address": "local:LineageProcess",
            "interval": float(config.get("time_step", 1.0)),
            "config": {
                "cache_dir": config.get("cache_dir", "out/cache"),
                "seed": spec.seed,
                "lineage_seed": spec.seed,
                "variant_index": spec.variant_index,
                "variant_name": spec.variant_name,
                "config_overrides": dict(spec.overrides),
                "generations": int(config.get("generations", 1)),
                "single_daughters": bool(config.get("single_daughters", True)),
                "experiment_id": config.get("experiment_id", "default"),
                "out_dir": config.get("out_dir", "out/workflow"),
                "max_duration_per_gen": float(config.get("max_duration_per_gen", 3600.0)),
            },
            "inputs": {},
            "outputs": {
                "summary": ["summary"],
                "complete": ["complete"],
            },
        },
        "summary": {},
        "complete": False,
    }


def build_meta_composite(config: dict[str, Any]) -> dict[str, Any]:
    """Return a process-bigraph document for the full sweep described by ``config``."""
    branches = expand_branches(config)
    branch_state = {_branch_key(spec): _lineage_node(spec, config) for spec in branches}
    return {
        "state": {
            "global_time": 0.0,
            "branches": branch_state,
        },
        "skip_initial_steps": True,
        "sequential_steps": False,
    }
```

Note on `register_link` / `process_registry.access`: this mirrors `v2ecoli/colony.py:143` (`core.register_link('EcoliWCM', EcoliWCM)`). If `process_registry.access` is not the exact accessor in this `bigraph-schema` version, adjust the assertion in the test to match how `colony.py` verifies resolution (grep `process_registry` in the installed `bigraph_schema`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_meta_composite_build.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/meta_composite.py tests/test_meta_composite_build.py
git commit -m "feat(workflow): build_meta_composite + process registration"
```

---

## Task 6: Driver loop + CLI

**Files:**
- Create: `v2ecoli/workflow/run.py`
- Modify: `pyproject.toml` (console script)
- Modify: `v2ecoli/workflow/__init__.py` (re-exports)
- Test: `tests/test_workflow_smoke.py`

- [ ] **Step 1: Write the failing smoke test (cache-gated)**

Create `tests/test_workflow_smoke.py`:

```python
import os
import json
import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def test_tiny_sweep_runs_to_completion(tmp_path):
    from v2ecoli.workflow.run import run_workflow

    config = {
        "experiment_id": "smoke",
        "n_init_sims": 1,
        "generations": 1,
        "single_daughters": True,
        "cache_dir": CACHE,
        "out_dir": str(tmp_path / "parquet"),
        "variants": {},
        # Cap so the test ends fast even though a real division is ~2500 s.
        "max_duration_per_gen": 5.0,
        "time_step": 1.0,
    }
    result = run_workflow(config, max_total_time=20.0, pbg_out=str(tmp_path / "sweep.pbg"))
    assert result["complete"] is True
    assert os.path.exists(str(tmp_path / "sweep.pbg"))
    # one branch, completed
    assert len(result["branches"]) == 1
    assert all(b["complete"] for b in result["branches"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && V2ECOLI_CACHE=out/cache python -m pytest tests/test_workflow_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.run'` (or SKIP without cache).

- [ ] **Step 3: Write the implementation**

Create `v2ecoli/workflow/run.py`:

```python
"""Driver + CLI for v2ecoli workflow sweeps.

Loads a vEcoli-style JSON config, builds the meta-composite, and ticks it
until every branch reports ``complete`` (or a global wall/sim cap is hit).
Saves the sweep document as a .pbg for inspection.

    v2ecoli-workflow --config configs/two_generations.json --out out/two_gen
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from process_bigraph import Composite

from v2ecoli.core import build_core
from v2ecoli.workflow.config import load_config_with_inheritance
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)


def _all_complete(composite) -> bool:
    branches = composite.state.get("branches", {})
    return bool(branches) and all(
        b.get("complete") for b in branches.values())


def run_workflow(config: dict[str, Any], *, max_total_time: float = 1e9,
                 pbg_out: str | None = None) -> dict[str, Any]:
    """Build and run the meta-composite for ``config``. Returns a result dict."""
    core = build_core()
    register_workflow_processes(core)

    doc = build_meta_composite(config)
    composite = Composite(doc, core=core)

    dt = float(config.get("time_step", 1.0))
    elapsed = 0.0
    while not _all_complete(composite) and elapsed < max_total_time:
        composite.run(dt)
        elapsed += dt

    if pbg_out:
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)

    branches = composite.state.get("branches", {})
    return {
        "complete": _all_complete(composite),
        "elapsed": elapsed,
        "branches": {k: {"complete": v.get("complete"), "summary": v.get("summary")}
                     for k, v in branches.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a workflow config JSON.")
    parser.add_argument("--out", default=None,
                        help="Output dir for parquet + sweep.pbg (default: out/<experiment_id>).")
    parser.add_argument("--build-only", action="store_true",
                        help="Build + save the .pbg without running.")
    parser.add_argument("--max-total-time", type=float, default=1e9,
                        help="Global sim-time safety cap (seconds).")
    args = parser.parse_args()

    config = load_config_with_inheritance(args.config)
    exp = config.get("experiment_id") or "workflow"
    out_dir = args.out or os.path.join("out", exp)
    config.setdefault("out_dir", os.path.join(out_dir, "parquet"))
    os.makedirs(out_dir, exist_ok=True)
    pbg_out = os.path.join(out_dir, "sweep.pbg")

    if args.build_only:
        core = build_core()
        register_workflow_processes(core)
        composite = Composite(build_meta_composite(config), core=core)
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)
        print(f"Built {len(composite.state['branches'])} branches → {pbg_out}")
        return

    result = run_workflow(config, max_total_time=args.max_total_time, pbg_out=pbg_out)
    n = len(result["branches"])
    done = sum(1 for b in result["branches"].values() if b["complete"])
    print(f"Sweep '{exp}': {done}/{n} branches complete in {result['elapsed']:.0f} s sim.")
    print(f"Saved: {pbg_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Re-export from the package**

Replace `v2ecoli/workflow/__init__.py` contents with:

```python
"""v2ecoli workflow framework: meta-composite variants × seeds × generations sweeps."""

from v2ecoli.workflow.config import load_config_with_inheritance
from v2ecoli.workflow.variants import expand_branches, BranchSpec
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)
from v2ecoli.workflow.run import run_workflow

__all__ = [
    "load_config_with_inheritance",
    "expand_branches",
    "BranchSpec",
    "build_meta_composite",
    "register_workflow_processes",
    "run_workflow",
]
```

- [ ] **Step 5: Add the console script**

In `pyproject.toml`, under `[project.scripts]` (create the table if absent — check existing entries like `v2ecoli-parca` first to match format), add:

```toml
v2ecoli-workflow = "v2ecoli.workflow.run:main"
```

Then reinstall the console script: `cd /Users/eranagmon/code/v2ecoli && pip install -e . --no-deps -q`

- [ ] **Step 6: Run the smoke test**

Run: `cd /Users/eranagmon/code/v2ecoli && V2ECOLI_CACHE=out/cache python -m pytest tests/test_workflow_smoke.py -v`
Expected: PASS (or SKIP without cache). If it runs, it builds a 1-branch sweep, caps each generation at 5 s sim time, completes, and writes `sweep.pbg`.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/workflow/run.py v2ecoli/workflow/__init__.py pyproject.toml tests/test_workflow_smoke.py
git commit -m "feat(workflow): driver loop + v2ecoli-workflow CLI"
```

---

## Task 7: Analysis Steps — base, scale registry, one example

**Files:**
- Create: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_analysis.py`:

```python
from v2ecoli.workflow.analysis import (
    AnalysisStep, ANALYSIS_SCALES, MassFractionSummary)


def test_five_scales_registered():
    assert set(ANALYSIS_SCALES) == {
        "single", "multidaughter", "multigeneration", "multiseed", "multivariant"}


def test_mass_fraction_summary_is_single_scale():
    assert MassFractionSummary.scale == "single"
    assert issubclass(MassFractionSummary, AnalysisStep)


def test_mass_fraction_summary_computes_fractions():
    step = MassFractionSummary({}, core=None)
    rows = [
        {"listeners": {"mass": {"dry_mass": 100.0, "protein_mass": 60.0,
                                "rRna_mass": 20.0, "dna_mass": 20.0}}},
        {"listeners": {"mass": {"dry_mass": 200.0, "protein_mass": 120.0,
                                "rRna_mass": 40.0, "dna_mass": 40.0}}},
    ]
    out = step.analyze(rows)
    assert abs(out["protein_fraction_mean"] - 0.6) < 1e-9
    assert abs(out["rRna_fraction_mean"] - 0.2) < 1e-9
    assert out["n_rows"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.analysis'`

- [ ] **Step 3: Write the implementation**

Create `v2ecoli/workflow/analysis.py`:

```python
"""Analyses as process-bigraph Steps that run on simulation results.

Defines the AnalysisStep base, the five-scale registry mirroring vEcoli's
analysis hierarchy, and one worked example (MassFractionSummary, ``single``
scale). Each scale declares which slice of emitted results it reads:

    single          one cell's timeseries
    multidaughter   sister cells from one division
    multigeneration cells across a lineage's generations
    multiseed       cells across seeds (same variant)
    multivariant    cells across all variants

Porting the full vEcoli analysis library onto this base is a follow-up spec.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step


# scale name -> human description of the result slice it consumes
ANALYSIS_SCALES: dict[str, str] = {
    "single": "one cell's timeseries",
    "multidaughter": "sister cells from one division",
    "multigeneration": "cells across a lineage's generations",
    "multiseed": "cells across seeds of one variant",
    "multivariant": "cells across all variants",
}


class AnalysisStep(V2Step):
    """Base for result-consuming analysis Steps.

    Subclasses set ``scale`` (one of ANALYSIS_SCALES) and implement
    ``analyze(rows) -> dict``. ``rows`` is a list of emitted result records
    (dicts shaped like the partitioned parquet rows / in-state snapshots) for
    the slice this scale covers. The Step's update() reads ``results`` from
    state and writes the analysis output to ``analysis``.
    """

    scale: str = "single"
    config_schema = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.scale not in ANALYSIS_SCALES:
            raise ValueError(
                f"{cls.__name__}.scale={cls.scale!r} not in {sorted(ANALYSIS_SCALES)}")

    def inputs(self):
        return {"results": "list"}

    def outputs(self):
        return {"analysis": "map"}

    def analyze(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def update(self, state, interval=None):
        rows = state.get("results") or []
        return {"analysis": self.analyze(rows)}


class MassFractionSummary(AnalysisStep):
    """Single-scale example: mean mass fractions across a cell's timeseries."""

    name = "mass_fraction_summary"
    scale = "single"

    def analyze(self, rows):
        if not rows:
            return {"n_rows": 0}
        fractions = {"protein": [], "rRna": [], "dna": []}
        for r in rows:
            mass = (r.get("listeners", {}) or {}).get("mass", {}) or {}
            dry = float(mass.get("dry_mass", 0.0)) or 0.0
            if dry <= 0:
                continue
            fractions["protein"].append(float(mass.get("protein_mass", 0.0)) / dry)
            fractions["rRna"].append(float(mass.get("rRna_mass", 0.0)) / dry)
            fractions["dna"].append(float(mass.get("dna_mass", 0.0)) / dry)
        out: dict[str, Any] = {"n_rows": len(rows)}
        for name, vals in fractions.items():
            out[f"{name}_fraction_mean"] = (sum(vals) / len(vals)) if vals else 0.0
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_analysis.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(workflow): analysis Step base + five-scale registry + example"
```

---

## Task 8: Port example configs

**Files:**
- Create: `v2ecoli/configs/default.json`
- Create: `v2ecoli/configs/two_generations.json`
- Test: extend `tests/test_workflow_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_config.py`:

```python
import os
from v2ecoli.workflow.variants import expand_branches


def test_ported_two_generations_config_expands():
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "v2ecoli", "configs")
    cfg = load_config_with_inheritance(os.path.join(cfg_dir, "two_generations.json"))
    assert cfg["generations"] == 2
    assert cfg["n_init_sims"] == 2
    branches = expand_branches(cfg)
    # no variants block → baseline only × 2 seeds
    assert len(branches) == 2
    assert {b.seed for b in branches} == {0, 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_config.py::test_ported_two_generations_config_expands -v`
Expected: FAIL with `FileNotFoundError` (config not present).

- [ ] **Step 3: Create the configs**

Create `v2ecoli/configs/default.json`:

```json
{
    "experiment_id": "default",
    "generations": 1,
    "n_init_sims": 1,
    "single_daughters": true,
    "lineage_seed": 0,
    "different_seeds_per_variant": false,
    "skip_baseline": false,
    "cache_dir": "out/cache",
    "out_dir": "out/workflow",
    "time_step": 1.0,
    "max_duration_per_gen": 3600.0,
    "variants": {},
    "analysis_options": {}
}
```

Create `v2ecoli/configs/two_generations.json`:

```json
{
    "inherit_from": ["default.json"],
    "experiment_id": "two_generations",
    "generations": 2,
    "n_init_sims": 2,
    "single_daughters": true,
    "analysis_options": {
        "single": {"mass_fraction_summary": {}}
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_config.py::test_ported_two_generations_config_expands -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/configs/default.json v2ecoli/configs/two_generations.json tests/test_workflow_config.py
git commit -m "feat(workflow): port default + two_generations example configs"
```

---

## Task 9: Full suite + integration check

**Files:** none (verification task)

- [ ] **Step 1: Run the full workflow test suite**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_workflow_config.py tests/test_workflow_variants.py tests/test_workflow_lineage.py tests/test_meta_composite_build.py tests/test_workflow_analysis.py tests/test_baseline_overrides.py tests/test_workflow_smoke.py -v`
Expected: All PASS (cache-gated tests SKIP if `out/cache` absent — note which skipped).

- [ ] **Step 2: Verify the CLI end to end (if cache present)**

Run: `cd /Users/eranagmon/code/v2ecoli && V2ECOLI_CACHE=out/cache v2ecoli-workflow --config v2ecoli/configs/two_generations.json --build-only --out /tmp/two_gen`
Expected: prints `Built 2 branches → /tmp/two_gen/sweep.pbg`; the `.pbg` exists and contains a `branches` map with two `variant=0/seed=0` / `variant=0/seed=1` entries each holding a `local:LineageProcess` node.

- [ ] **Step 3: Confirm no regressions in existing composite tests**

Run: `cd /Users/eranagmon/code/v2ecoli && python -m pytest tests/test_build_composite.py -v`
Expected: PASS (baseline still builds with the new optional `config_overrides` param defaulting to none).

- [ ] **Step 4: Commit any fixups**

```bash
git add -A && git commit -m "test(workflow): full-suite verification fixups" || echo "nothing to commit"
```

---

## Self-Review Notes (addressed during planning)

- **Spec coverage:** config loader (T1) ✓, port vEcoli JSON (T8) ✓, declarative variant overrides (T2, T3) ✓, meta-composite branches as embedded sub-composites (T4, T5) ✓, generation control incl. single_daughters + stop-at-N (T4) ✓, partitioned emission with matching metadata (T4 via `parquet_vecoli`) ✓, analyses as Steps + five-scale registry + one example (T7) ✓, driver/CLI + `--build-only` inspection (T6) ✓.
- **Deferred (matches spec §Out of scope):** parallel/distributed execution, ParCa-recomputing variants, resume/caching, full analysis library, `single_daughters=false` binary tree, `nested` variant grammar. Each is guarded by an explicit error or documented default, not a silent gap.
- **Type consistency:** `BranchSpec` fields, `LineageProcess` config keys, and `_lineage_node` config keys are aligned across T2/T4/T5; `analyze(rows)->dict` and `ANALYSIS_SCALES` keys aligned across T7.
- **Known verification dependency:** the smoke + override + CLI tests require a ParCa cache at `out/cache` (or `$V2ECOLI_CACHE`). They SKIP cleanly without it; on a cache-bearing machine they must pass. The `core.register_link` / `process_registry.access` accessor names should be confirmed against the installed `bigraph-schema` (mirrors `colony.py:141-143`).
