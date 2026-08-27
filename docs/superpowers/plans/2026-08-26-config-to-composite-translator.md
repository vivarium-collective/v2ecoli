# Config → Composite Translator (Phase 1: v2ecoli) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate a vEcoli-style config's declared layer into a `process_bigraph.Composite`-executable v2 document — one artifact viewable in the loom and runnable by `Composite`.

**Architecture:** A standalone `v2ecoli/library/config_to_composite.py` maps `add_processes`/`swap_processes`/`exclude`/`spatial`/`variants` → a `{schema, state}` document: a `_type:"process"` node per declared process, `address: local:<ClassName>`, `config` from the config's own `process_configs`, wiring from `topology`/`topology_registry`. A companion registration hook wraps each declared vivarium process via the existing `vivarium_bridge` adapter and registers it under `local:<ClassName>` so the addresses resolve at realize time. The Phase-1a viewer transform (`config_bigraph.py`) is re-homed here as the shared node/wiring core.

**Tech Stack:** Python 3.12, `process_bigraph`, `bigraph_schema`, vivarium-core (via the adapter), pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-config-to-composite-translator-design.md`

## Global Constraints

- **Home:** all code under `v2ecoli/library/`; imported as `v2ecoli.library.*`. Reaches sms-ecoli by `scripts/sync_upstream.sh` — **no `descope/extensions.yaml` entry** (it is upstream code).
- **Worktree:** author in `~/code/v2ecoli--config-to-composite` (branch `feat/config-to-composite-translator`, off `origin/main`). Never touch canonical `~/code/v2ecoli`.
- **Run tests with:** `PYTHONPATH=<worktree> ~/code/sms-ecoli/.venv/bin/python -m pytest` (the worktree has no `.venv`; the sms-ecoli venv has all deps). Verify `python -c "import v2ecoli; print(v2ecoli.__file__)"` resolves to the worktree.
- **Fork:** class-address / `topology_registry` / adapter lookups need the vEcoli fork on the path — `V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli-private`, and `import ecoli.processes` **before** importing the translator (import-order: the fork's registry must load first).
- **No AI attribution** in commit messages.
- **Scope:** the config's DECLARED layer only. Do NOT reproduce baseline processes, `sim_data` configs, bulk/unique state, partitioned processes, RNG, or division — those are deferred (spec §5).

---

### Task 1: Re-home the Phase-1a viewer transform into v2ecoli

Brings the already-written structural transform (currently on an sms-ecoli worktree) upstream as the shared node/wiring core.

**Files:**
- Create: `v2ecoli/library/config_bigraph.py` (copy from `~/code/sms-ecoli--config-bigraph/v2ecoli/library/config_bigraph.py`)
- Test: `tests/test_config_bigraph.py` (copy from `~/code/sms-ecoli--config-bigraph/tests/test_config_bigraph.py`)

**Interfaces:**
- Produces: `config_to_document(config: dict, *, fork_dir="") -> {"state": dict, "summary": dict}`; helpers `_process_node`, `_normalize_path`, `_resolve_process_meta`, `_fork_registries`, `_spatial_node`, `_variants_node`.

- [ ] **Step 1: Copy the module and test verbatim**

```bash
cp ~/code/sms-ecoli--config-bigraph/v2ecoli/library/config_bigraph.py \
   ~/code/v2ecoli--config-to-composite/v2ecoli/library/config_bigraph.py
cp ~/code/sms-ecoli--config-bigraph/tests/test_config_bigraph.py \
   ~/code/v2ecoli--config-to-composite/tests/test_config_bigraph.py
```

- [ ] **Step 2: Run the re-homed tests**

Run: `cd ~/code/v2ecoli--config-to-composite && PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_bigraph.py -q`
Expected: PASS (9 passed).

- [ ] **Step 3: Commit**

```bash
git add v2ecoli/library/config_bigraph.py tests/test_config_bigraph.py
git commit -m "feat(config-bigraph): re-home the config→loom-document viewer transform upstream"
```

---

### Task 2: `config_to_composite` — executable-shape process nodes

New module producing the `{schema, state}` document with `address`-based process nodes and `config` from `process_configs`.

**Files:**
- Create: `v2ecoli/library/config_to_composite.py`
- Test: `tests/test_config_to_composite.py`

**Interfaces:**
- Consumes: `config_bigraph._normalize_path`, `_resolve_process_meta`, `_spatial_node`, `_variants_node`.
- Produces: `config_to_composite(config: dict, *, fork_dir="") -> {"schema": dict, "state": dict}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_to_composite.py
from v2ecoli.library.config_to_composite import config_to_composite

def _cfg():
    return {
        "add_processes": ["proc_a"],
        "swap_processes": {"old_m": "new_m"},
        "process_configs": {"proc_a": {"rate": 3}},
        "topology": {"proc_a": {"bulk": ["bulk"]}, "new_m": {"flux": ["metabolites"]}},
    }

def test_process_nodes_are_address_based_and_executable_shape():
    doc = config_to_composite(_cfg())
    assert set(doc) == {"schema", "state"}
    node = doc["state"]["proc_a"]
    assert node["_type"] == "process"
    assert node["address"] == "local:proc_a"      # local:<name> when no fork enriches
    assert node["config"] == {"rate": 3}
    assert "_draft" not in node                    # executable, not a draft view

def test_swap_node_annotated_and_present():
    state = config_to_composite(_cfg())["state"]
    assert state["new_m"]["_type"] == "process"
    assert state["new_m"]["_contract"]["swap_replaces"] == "old_m"

def test_store_nodes_exist_for_wire_targets():
    state = config_to_composite(_cfg())["state"]
    assert state["bulk"] == {}
    assert state["metabolites"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_to_composite.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/library/config_to_composite.py
"""vEcoli-style config → a process_bigraph.Composite-executable v2 document.

Standalone translator of the config's DECLARED layer (spec
docs/superpowers/specs/2026-08-26-config-to-composite-translator-design.md).
Emits address-based process nodes (config from the config's own process_configs,
wiring from topology / topology_registry) that realize once each
``local:<ClassName>`` address is registered (see register_declared_processes).
"""
from __future__ import annotations

from typing import Any

from v2ecoli.library.config_bigraph import (
    _normalize_path, _resolve_process_meta, _spatial_node, _variants_node,
)


def _node(name, topology, config, fork_dir):
    address, _desc, registry_topology = _resolve_process_meta(name, fork_dir)
    ports = dict(registry_topology)
    ports.update(topology or {})
    inputs, outputs, targets = {}, {}, set()
    for port, path in ports.items():
        norm = _normalize_path(path)
        if norm is not None:
            inputs[port] = norm            # bidirectional default (Task 3 refines)
            outputs[port] = norm
            targets.add(norm[0])
    node = {
        "_type": "process",
        "address": address,
        "config": dict(config or {}),
        "inputs": inputs,
        "outputs": outputs,
        "interval": 1.0,
    }
    return node, targets


def config_to_composite(config: dict, *, fork_dir: str = "") -> dict:
    add = list(config.get("add_processes") or [])
    swap = dict(config.get("swap_processes") or {})
    exclude = list(config.get("exclude_processes") or [])
    topology = dict(config.get("topology") or {})
    process_configs = dict(config.get("process_configs") or {})
    spatial = config.get("spatial_environment_config")
    variants = dict(config.get("variants") or {})

    state: dict[str, Any] = {}
    targets: set[str] = set()

    for name in add:
        node, t = _node(name, topology.get(name), process_configs.get(name), fork_dir)
        state[name] = node
        targets |= t
    for old, new in swap.items():
        node, t = _node(new, topology.get(new), process_configs.get(new), fork_dir)
        node.setdefault("_contract", {})["swap_replaces"] = old
        state[new] = node
        targets |= t

    for store in sorted(targets):
        state.setdefault(store, {})
    if exclude:
        state["excluded_processes"] = {e: {} for e in exclude}
    if isinstance(spatial, dict) and spatial:
        state["environment"] = _spatial_node(spatial)
    if variants:
        state["variants"] = _variants_node(variants)

    return {"schema": {}, "state": state}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_to_composite.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/config_to_composite.py tests/test_config_to_composite.py
git commit -m "feat(config-to-composite): address-based executable-shape process nodes"
```

---

### Task 3: Fork enrichment — real class addresses + registry-topology ports

With the fork loaded, addresses become `local:<ClassName>` and processes the config leaves un-wired (pg-shape) get ports from `topology_registry`. This is already handled by `_resolve_process_meta`; this task adds fork-backed coverage and confirms tuple-path handling.

**Files:**
- Modify: `tests/test_config_to_composite.py`

**Interfaces:**
- Consumes: `config_to_composite` (Task 2), the fork's `ecoli.processes` registry.

- [ ] **Step 1: Write the failing (fork-backed) test**

```python
import os, sys, pytest

FORK = "/Users/eranagmon/code/vEcoli-private"

@pytest.mark.skipif(not os.path.isdir(FORK), reason="vEcoli-private fork absent")
def test_fork_enriches_address_and_registry_ports():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    import ecoli.processes  # noqa: F401 — fork registry must load first
    from v2ecoli.library.config_to_composite import config_to_composite
    cfg = {"add_processes": ["pg-shape"], "topology": {}}  # no config topology
    node = config_to_composite(cfg, fork_dir=FORK)["state"]["pg-shape"]
    assert node["address"] == "local:PGShape"                 # real class name
    assert set(node["inputs"]) == {"bulk", "environment", "listeners"}  # from registry
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `V2E_VECOLI_DIR=$FORK PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_to_composite.py -q -k fork`
Expected: PASS if the registry-topology fallback + tuple normalization from Task 1 already flow through; FAIL points at a `_normalize_path`/`_resolve_process_meta` gap to fix in `config_bigraph.py`.

- [ ] **Step 3: Fix any gap** (only if Step 2 failed)

If FAIL, ensure `config_bigraph._normalize_path` accepts tuples (`isinstance(path, (list, tuple))`) and `_resolve_process_meta` returns the `topology_registry` map — both already present in the re-homed Phase-1a code; adjust if the copy drifted.

- [ ] **Step 4: Run to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_to_composite.py v2ecoli/library/config_bigraph.py
git commit -m "test(config-to-composite): fork-backed address + registry-topology ports"
```

---

### Task 4: `register_declared_processes` — make addresses resolve

Registers each declared process's adapter-wrapped class under `local:<ClassName>` in a given core, so the document's `address` resolves at realize time. This is the "executable" hinge (spec §3.2).

**Files:**
- Modify: `v2ecoli/library/config_to_composite.py`
- Test: `tests/test_config_to_composite.py`

**Interfaces:**
- Consumes: `vivarium_bridge.wrap_vivarium_process(v1_cls, *, name=...)`; `v2ecoli.core.build_core`; the fork `process_registry`.
- Produces: `register_declared_processes(core, config: dict, *, fork_dir="") -> list[str]` (returns the `local:` names registered).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.skipif(not os.path.isdir(FORK), reason="vEcoli-private fork absent")
def test_register_declared_processes_makes_addresses_resolvable():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    import ecoli.processes  # noqa: F401
    from v2ecoli.core import build_core
    from v2ecoli.library.config_to_composite import (
        config_to_composite, register_declared_processes)
    core = build_core()
    cfg = {"add_processes": ["pg-shape"], "topology": {}}
    names = register_declared_processes(core, cfg, fork_dir=FORK)
    assert "PGShape" in names
    # the registered address resolves through the core's link registry
    assert core.link_registry.access("PGShape") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `V2E_VECOLI_DIR=$FORK PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_to_composite.py -q -k register`
Expected: FAIL (`register_declared_processes` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `config_to_composite.py`. NOTE: match the exact registration call to `v2ecoli.core.register_ecoli_core` (grep it for `register_process` vs `link_registry.register`); the snippet below assumes `core.register_process(name, cls)` per the pbg convention — adjust to the repo's actual API.

```python
def _declared_process_names(config: dict) -> list[str]:
    names = list(config.get("add_processes") or [])
    names += list((config.get("swap_processes") or {}).values())
    return names

def register_declared_processes(core, config: dict, *, fork_dir: str = "") -> list[str]:
    """Wrap each declared vivarium process via the adapter and register it under
    ``local:<ClassName>`` in ``core`` so the translated document's addresses
    resolve. Returns the list of registered class names. Best-effort per name:
    an unresolvable/unwrappable process is skipped (kept out of the return)."""
    import os, sys
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    fork = fork_dir or os.environ.get("V2E_VECOLI_DIR", "")
    if fork and fork not in sys.path:
        sys.path.insert(0, fork)
    try:
        import ecoli.processes  # noqa: F401
        from vivarium.core.registry import process_registry
    except Exception:
        return []
    registered: list[str] = []
    for name in _declared_process_names(config):
        try:
            v1_cls = process_registry.access(name)
            if v1_cls is None:
                continue
            wrapped = wrap_vivarium_process(v1_cls, name=name)
            cls_name = getattr(v1_cls, "__name__", name)
            core.register_process(cls_name, wrapped)   # match register_ecoli_core
            registered.append(cls_name)
        except Exception:
            continue
    return registered
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS. (If `core.link_registry.access`/`register_process` names differ, align with `register_ecoli_core`.)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/config_to_composite.py tests/test_config_to_composite.py
git commit -m "feat(config-to-composite): register adapter-wrapped procs under local:<ClassName>"
```

---

### Task 5: Executability — `Composite` realizes the declared-layer document

The end-to-end check: register, translate, and confirm `process_bigraph.Composite({schema, state}, core)` realizes without error.

**Files:**
- Test: `tests/test_config_to_composite.py`

**Interfaces:**
- Consumes: `config_to_composite`, `register_declared_processes`, `v2ecoli.core.build_core`, `process_bigraph.Composite`.

- [ ] **Step 1: Write the failing/verifying test**

```python
@pytest.mark.skipif(not os.path.isdir(FORK), reason="vEcoli-private fork absent")
def test_declared_layer_document_realizes_in_composite():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    import ecoli.processes  # noqa: F401
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.library.config_to_composite import (
        config_to_composite, register_declared_processes)
    core = build_core()
    cfg = {"add_processes": ["pg-shape"], "topology": {}}
    register_declared_processes(core, cfg, fork_dir=FORK)
    doc = config_to_composite(cfg, fork_dir=FORK)
    comp = Composite(doc, core=core)          # must not raise: address resolves + realizes
    assert comp is not None
```

- [ ] **Step 2: Run the test**

Run: `V2E_VECOLI_DIR=$FORK PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_to_composite.py -q -k realizes`
Expected: PASS. If it raises `no link found at address`, the registered name ≠ the node `address` name — reconcile `_resolve_process_meta`'s `local:<name>` with `register_declared_processes`'s `cls_name` (both must use the class `__name__`). Fix in whichever is inconsistent.

- [ ] **Step 3: Reconcile address/name if needed** (only if Step 2 raised)

Ensure the node `address` (from `_resolve_process_meta`) and the registered name (`cls_name`) are identical — both the class `__name__`. Adjust `_resolve_process_meta` or the registration to match; re-run Step 2 to green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_config_to_composite.py v2ecoli/library/config_to_composite.py v2ecoli/library/config_bigraph.py
git commit -m "test(config-to-composite): Composite realizes the declared-layer document"
```

---

### Task 6: Live verification against real configs (no new code)

Confirms real antibiotic configs translate, render (loom document shape), and realize; records the numbers in the plan's PR.

**Files:** none (verification only).

- [ ] **Step 1: Translate + realize the real configs**

Run:
```bash
cd ~/code/v2ecoli--config-to-composite
V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli-private PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python - <<'PY'
import sys, os, json
FORK="/Users/eranagmon/code/vEcoli-private"; sys.path.insert(0, FORK)
import ecoli.processes
from process_bigraph import Composite
from v2ecoli.core import build_core
from v2ecoli.library.config_to_composite import config_to_composite, register_declared_processes
for name in ["final_mec.json","mecillinam_shape.json"]:
    cfg=json.load(open(os.path.join(FORK,"configs",name)))
    core=build_core(); register_declared_processes(core,cfg,fork_dir=FORK)
    doc=config_to_composite(cfg,fork_dir=FORK)
    procs=[k for k,v in doc["state"].items() if isinstance(v,dict) and v.get("_type")=="process"]
    Composite(doc, core=core)  # realizes
    print(name, "procs:", len(procs), "realized OK")
PY
```
Expected: both print `realized OK`; `final_mec.json` ≥5 procs, `mecillinam_shape.json` ≥7 procs (incl. `pg-shape`).

- [ ] **Step 2: Full test suite green**

Run: `V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli-private PYTHONPATH=$PWD ~/code/sms-ecoli/.venv/bin/python -m pytest tests/test_config_bigraph.py tests/test_config_to_composite.py -q`
Expected: all PASS.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin feat/config-to-composite-translator
gh pr create --repo vivarium-collective/v2ecoli \
  --title "feat: config→composite translator (declared-layer, Composite-executable)" \
  --body "Implements docs/superpowers/specs/2026-08-26-config-to-composite-translator-design.md (Phase 1). Standalone translator: vEcoli config declared layer → {schema,state} realized by process_bigraph.Composite via local:<ClassName> registration through the vivarium_bridge adapter. Verified: final_mec.json / mecillinam_shape.json translate + realize."
```

---

## Follow-on (separate plans, NOT this plan)

- **Phase 2 — Workbench glue:** env-worker `config_to_composite` method + `POST /api/config-to-composite` route + the `Apply` unification in `ConfigPanel.tsx` (loom Vite build). Written after Phase 1's contract lands.
- **Adapter-derived port direction:** narrow the bidirectional default to the adapter's `inputs()`/`outputs()` classification.
- **Referenced initial-state seed** so the declared-layer document runs standalone; and the reuse-EcoliSim whole-cell pure-JSON path.
- **Config inheritance:** fix `v2ecoli.workflow.config._merge_configs` list-of-lists crash so `load_config_with_inheritance` resolves `final_mec.json`.
