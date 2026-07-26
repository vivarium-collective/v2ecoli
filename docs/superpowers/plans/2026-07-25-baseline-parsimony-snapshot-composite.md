# baseline + parsimony snapshot composite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `baseline_parsimony` v2ecoli composite that runs `baseline` live and writes 3D `.pack.json` snapshots at declared sim-times (default: initial ≈10 s + pre-division), which the workbench Parsimony Viewer renders.

**Architecture:** Revive the packing machinery from the unmerged `v2e-3d` worktree into `v2ecoli/structural/`; factor a load-state-free packing entry; add an `EcoliPackStep` that packs the live state when a declared time arrives; add a generator that wraps `baseline()` and appends the step as a final execution layer.

**Tech Stack:** Python 3, process-bigraph (`Step`, `@composite_generator`), pbg-parsimony (`Capsule`, `Chromosome`, `Ingredient`, `build_pack`), v2ecoli (`baseline`, `cell_shape`), pytest.

## Global Constraints

- Reference source for the revive: `/Users/eranagmon/code/v2e-3d/v2ecoli/structural/` (branch `feat/3d-structural-model`) — `build.py`, `data/` (`ecoli_k12_genes.csv`, `uniprot_map.json`, `v2ecoli_state.npz`), `__init__.py`. Do NOT copy `composite.py`/`webapp/` (the pre-baked-state variant) — the live packing is new here.
- Pack files MUST be written to `studies/<study>/viz/3d/<name>.pack.json` and named so the workbench gallery sorts the initial state first: use names `initial` and `pre-division` (the viewer's `_pack_name_rank` ranks `initial`/`birth`/`10s` first).
- Pack schema is `parsimony.pack.v1` (from `pbg_parsimony.build_pack`); top-level keys: `format`, `bounds`, `compartments`, `ingredients`, `placements`.
- `pbg-parsimony` is already a dependency (`pyproject.toml:16`); do not re-pin it.
- Geometry comes from `Capsule.from_volume_fl(volume_fl)`; `volume_fl` is a `['shape']` store float key emitted by `ShapeStep` (`cell_shape.py`). Do NOT read a `Capsule` object from any store (the object is excluded from the emitted shape store).
- Append `EcoliPackStep` as a FINAL execution layer exactly as `baseline` appends `shape_step` (`v2ecoli/composites/baseline.py:895–919`) — never edit baseline's own flow.
- Run all tests with the worktree's interpreter: `/Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest …` from `/Users/eranagmon/code/v2e-3d-snapshots` (bare `python` may lack `unum`/deps). If that venv resolves the wrong checkout, prepend `PYTHONPATH=/Users/eranagmon/code/v2e-3d-snapshots`.
- Keep any full-generation / real-`build_pack` run behind an opt-in `@pytest.mark.slow` marker — default test runs must stay fast.

---

### Task 1: Revive the `v2ecoli/structural/` packing package

**Files:**
- Create: `v2ecoli/structural/__init__.py`, `v2ecoli/structural/build.py`, `v2ecoli/structural/data/{ecoli_k12_genes.csv,uniprot_map.json,v2ecoli_state.npz}` (copied from the `v2e-3d` worktree)
- Test: `tests/test_structural_revive.py`

**Interfaces:**
- Produces: `from v2ecoli.structural.build import select_ingredients, build_model, load_state, DATA` — `select_ingredients(counts: dict, *, top_n=40, lipid_count=40000) -> list[Ingredient]`.

- [ ] **Step 1: Copy the package from the v2e-3d worktree**

```bash
mkdir -p v2ecoli/structural/data
cp /Users/eranagmon/code/v2e-3d/v2ecoli/structural/__init__.py v2ecoli/structural/__init__.py
cp /Users/eranagmon/code/v2e-3d/v2ecoli/structural/build.py v2ecoli/structural/build.py
cp /Users/eranagmon/code/v2e-3d/v2ecoli/structural/data/ecoli_k12_genes.csv v2ecoli/structural/data/
cp /Users/eranagmon/code/v2e-3d/v2ecoli/structural/data/uniprot_map.json v2ecoli/structural/data/
cp /Users/eranagmon/code/v2e-3d/v2ecoli/structural/data/v2ecoli_state.npz v2ecoli/structural/data/
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_structural_revive.py
def test_structural_imports_and_selects():
    from v2ecoli.structural.build import select_ingredients, DATA
    assert DATA.is_dir()
    # a couple of known abundant species → non-empty ingredient list
    ings = select_ingredients({"EG10893-MONOMER": 5000, "CPLX0-3964": 500}, top_n=2)
    assert isinstance(ings, list) and len(ings) >= 1
```

- [ ] **Step 3: Run it — fix imports until it passes**

Run: `PYTHONPATH=/Users/eranagmon/code/v2e-3d-snapshots /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/test_structural_revive.py -v`
Expected first run: FAIL (module/import errors). Fix: ensure `__init__.py` doesn't import the omitted `composite.py`; `DATA` resolves to the copied `data/` dir; `_flat_dir()` resolves `reconstruction.ecoli.flat` (present in v2ecoli). If `build.py`'s top-level imports pull in `composite`/`webapp`, trim them. Iterate until PASS.

- [ ] **Step 4: Verify pristine test output**

Run the same command; Expected: PASS, no import warnings.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/structural tests/test_structural_revive.py
git commit -m "feat: revive v2ecoli/structural packing package from v2e-3d"
```

---

### Task 2: Load-state-free packing entry

**Files:**
- Modify: `v2ecoli/structural/build.py`
- Test: `tests/test_pack_from_state.py`

**Interfaces:**
- Consumes: `select_ingredients`, `Capsule.from_volume_fl`, `Chromosome`, `build_pack`, `DATA` (all in build.py).
- Produces:
  - `bulk_to_counts(bulk) -> dict[str, int]` — a live `['bulk']` store (structured array with `id`/`count` fields) → `{ecocyc_id: summed_count}`, compartment tags stripped. Factored from `load_state`'s tail (build.py ~lines 64–74).
  - `pack_from_state(out_dir, name, counts, volume_fl, *, top_n=40, scale=0.3, proxy_lod=2) -> dict` — everything `build_model` does after `load_state` (build.py ~lines 208–216): `select_ingredients` → `Capsule.from_volume_fl` → `Chromosome(...)` → `build_pack(...)`. Returns `build_pack`'s result dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pack_from_state.py
import json
from v2ecoli.structural.build import bulk_to_counts, pack_from_state

def test_bulk_to_counts_strips_compartments_and_sums():
    import numpy as np
    bulk = np.array([("EG10893-MONOMER[c]", 10), ("EG10893-MONOMER[m]", 5),
                     ("CPLX0-3964[c]", 3)],
                    dtype=[("id", "U40"), ("count", "i8")])
    counts = bulk_to_counts(bulk)
    assert counts["EG10893-MONOMER"] == 15   # summed across compartments
    assert counts["CPLX0-3964"] == 3

def test_pack_from_state_writes_valid_pack(tmp_path):
    counts = {"EG10893-MONOMER": 5000, "CPLX0-3964": 500}
    pack_from_state(str(tmp_path), "initial", counts, volume_fl=1.0, top_n=2)
    pack = json.loads((tmp_path / "initial.pack.json").read_text())
    assert pack["format"] == "parsimony.pack.v1"
    assert "ingredients" in pack and "placements" in pack
```

- [ ] **Step 2: Run — verify it fails**

Run: `… -m pytest tests/test_pack_from_state.py -v`
Expected: FAIL (`bulk_to_counts`/`pack_from_state` undefined).

- [ ] **Step 3: Implement**

In `build.py`, extract the tag-strip+sum loop from `load_state` into `bulk_to_counts(bulk)` (reuse the exact `id.split("[")[0]` strip and `int(count)` sum), and have `load_state` call it. Extract the packing tail from `build_model` into `pack_from_state(...)`, and have `build_model` call `pack_from_state(out_dir, name, *load_state(state_source), top_n=…, scale=…, proxy_lod=…)`. `pack_from_state` body:

```python
def pack_from_state(out_dir, name, counts, volume_fl, *, top_n=40, scale=0.3, proxy_lod=2):
    ingredients = select_ingredients(counts, top_n=top_n)
    capsule = Capsule.from_volume_fl(volume_fl)
    chromosome = Chromosome(
        beads=34000, spacing=135.0, bead_radius=12.0,
        genome_csv=str(DATA / "ecoli_k12_genes.csv"),
        segment=StructureRef("pdb", "1BNA"),
        supercoil={"radius": 90.0, "pitch": 130.0, "domains": 200})
    return build_pack(ingredients, capsule, chromosome,
                      out_dir=out_dir, name=name, scale=scale, proxy_lod=proxy_lod)
```

- [ ] **Step 4: Run — verify it passes**

Run: `… -m pytest tests/test_pack_from_state.py -v` → PASS. Also re-run Task 1's test (no regression).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/structural/build.py tests/test_pack_from_state.py
git commit -m "feat: pack_from_state + bulk_to_counts (packing without load_state)"
```

---

### Task 3: `EcoliPackStep` — declared-time snapshot packing

**Files:**
- Create: `v2ecoli/structural/pack_step.py`
- Test: `tests/test_pack_step.py`

**Interfaces:**
- Consumes: `pack_from_state`, `bulk_to_counts` (Task 2).
- Produces: `EcoliPackStep(Step)` with `config`: `snapshots` (dict `name -> time_spec`, where `time_spec` is a float sim-time or the string `"division_time"`), `study` (str), `out_dir` (str, default derived from `study`), `top_n` (int, default 40), `scale` (float, default 0.3), `epsilon_s` (float, default 1.0). `inputs()` ports: `bulk`, `shape`, `global_time`, `full_chromosomes`. `outputs()`: `pack_status` (`map[float]`). Packs each snapshot once, when its time arrives.

- [ ] **Step 1: Write the failing test (stubbed packer)**

```python
# tests/test_pack_step.py
from v2ecoli.structural import pack_step as ps

def _step(monkeypatch, snapshots, calls):
    monkeypatch.setattr(ps, "pack_from_state",
        lambda out_dir, name, counts, volume_fl, **k: calls.append((name, volume_fl)) or {"placements": [1]})
    monkeypatch.setattr(ps, "bulk_to_counts", lambda bulk: {"X": 1})
    return ps.EcoliPackStep(config={"snapshots": snapshots, "study": "s",
                                     "out_dir": "/tmp/o", "epsilon_s": 1.0})

def _state(t, division_time=None):
    fc = {"division_time": division_time} if division_time is not None else {}
    return {"bulk": [], "shape": {"volume_fl": 2.0}, "global_time": t, "full_chromosomes": fc}

def test_fixed_time_snapshot_fires_once(monkeypatch):
    calls = []
    step = _step(monkeypatch, {"initial": 10.0}, calls)
    step.update(_state(5.0));  assert calls == []          # before the time
    step.update(_state(10.0)); assert [c[0] for c in calls] == ["initial"]  # at/after
    step.update(_state(20.0)); assert [c[0] for c in calls] == ["initial"]  # not re-fired

def test_pre_division_uses_division_time(monkeypatch):
    calls = []
    step = _step(monkeypatch, {"pre-division": "division_time"}, calls)
    step.update(_state(30.0, division_time=None));  assert calls == []   # not scheduled yet
    step.update(_state(30.0, division_time=100.0)); assert calls == []   # scheduled, not near
    step.update(_state(99.5, division_time=100.0)); assert [c[0] for c in calls] == ["pre-division"]  # within epsilon
```

- [ ] **Step 2: Run — verify it fails**

Run: `… -m pytest tests/test_pack_step.py -v` → FAIL (module undefined).

- [ ] **Step 3: Implement**

```python
# v2ecoli/structural/pack_step.py
"""EcoliPackStep — write a parsimony 3D pack of the live cell at declared
simulation times. Appended as a final execution layer to a baseline run; it
runs every tick, packs a snapshot the first time its scheduled time arrives."""
from process_bigraph import Step

from v2ecoli.structural.build import pack_from_state, bulk_to_counts


class EcoliPackStep(Step):
    config_schema = {
        "snapshots": "tree[any]",          # {name: float sim-time | "division_time"}
        "study": "string",
        "out_dir": "string",
        "top_n": {"_type": "integer", "_default": 40},
        "scale": {"_type": "float", "_default": 0.3},
        "epsilon_s": {"_type": "float", "_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._fired = set()

    def inputs(self):
        return {"bulk": "any", "shape": "tree[any]",
                "global_time": "float", "full_chromosomes": "tree[any]"}

    def outputs(self):
        return {"pack_status": "map[float]"}

    def _due(self, name, spec, t, states):
        if name in self._fired:
            return False
        if isinstance(spec, str) and spec == "division_time":
            dt = (states.get("full_chromosomes") or {}).get("division_time")
            if not dt:                     # not scheduled yet
                return False
            return t >= float(dt) - self.config["epsilon_s"]
        return t >= float(spec)            # fixed sim-time

    def update(self, state, interval=None):
        t = float(state.get("global_time") or 0.0)
        status = {}
        for name, spec in (self.config.get("snapshots") or {}).items():
            if not self._due(name, spec, t, state):
                continue
            counts = bulk_to_counts(state.get("bulk"))
            volume_fl = float((state.get("shape") or {}).get("volume_fl") or 0.0)
            res = pack_from_state(self.config["out_dir"], name, counts, volume_fl,
                                  top_n=self.config["top_n"], scale=self.config["scale"])
            self._fired.add(name)
            status[name] = float(len((res or {}).get("placements") or []))
        return {"pack_status": status} if status else {}
```

(If `Step.config_schema` typing differs on this pbg version — check an existing step like `cell_shape.ShapeStep` — mirror its schema style; the behavior above is what matters.)

- [ ] **Step 4: Run — verify it passes**

Run: `… -m pytest tests/test_pack_step.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/structural/pack_step.py tests/test_pack_step.py
git commit -m "feat: EcoliPackStep — pack live cell at declared sim-times"
```

---

### Task 4: `baseline_parsimony` composite generator

**Files:**
- Create: `v2ecoli/composites/baseline_parsimony.py`
- Modify: `v2ecoli/composites/__init__.py`
- Test: `tests/test_baseline_parsimony_composite.py`

**Interfaces:**
- Consumes: `baseline` (the generator), `EcoliPackStep`.
- Produces: a registered `@composite_generator(name="baseline_parsimony")` returning a pbg document whose per-agent `flow_order` ends with `pack_step`, wired to `['bulk']`, `['shape']`, `['global_time']`, `['full_chromosomes']`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_parsimony_composite.py
def test_generator_appends_pack_step(tmp_path):
    import v2ecoli
    from v2ecoli.core import build_core
    core = build_core()
    doc = v2ecoli.build_composite("ecoli_structural", core=core, seed=0,
                                  cache_dir="out/cache", study="ecoli-3d",
                                  emitter="null")  # doc-shape only; do not run
    # find the per-agent cell state
    agent = next(iter(doc.composition["state"]["agents"].values())) \
        if hasattr(doc, "composition") else None
    # tolerate Composite vs raw-doc return: assert via the built state tree
    state = doc.state if hasattr(doc, "state") else doc["state"]
    cell = next(iter(state["agents"].values()))
    assert "pack_step" in cell
    ps = cell["pack_step"]
    assert ps["address"].endswith("EcoliPackStep")
    assert ps["inputs"]["bulk"] == ["bulk"]
    assert ps["inputs"]["shape"] == ["shape"]
    assert ps["inputs"]["full_chromosomes"] == ["full_chromosomes"]
```

(If `build_composite` returns a `Composite`, read `.state`; adjust the accessor to the real return per `v2ecoli/__init__.py:build_composite` — it wraps `Composite(doc, core=core)`. The assertion targets the wired step node.)

- [ ] **Step 2: Run — verify it fails**

Run: `… -m pytest tests/test_baseline_parsimony_composite.py -v` → FAIL (`baseline_parsimony` not registered).

- [ ] **Step 3: Implement the generator**

```python
# v2ecoli/composites/baseline_parsimony.py
"""baseline_parsimony — the baseline whole-cell model plus a parsimony 3D
packing step that writes snapshot packs at declared simulation times."""
from __future__ import annotations

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites.ecoli_baseline import baseline


@composite_generator(name="baseline_parsimony", default_n_steps=2700)
def baseline_parsimony(core=None, *, study: str = "ecoli-3d",
                       snapshots: dict | None = None, top_n: int = 40,
                       scale: float = 0.3, **kwargs) -> dict:
    doc = baseline(core=core, **kwargs)
    if core is None:
        return doc
    from v2ecoli.structural.pack_step import EcoliPackStep
    core.register_link("EcoliPackStep", EcoliPackStep)
    snaps = snapshots or {"initial": 10.0, "pre-division": "division_time"}
    out_dir = f"studies/{study}/viz/3d"
    for agent_id, cell in doc["state"]["agents"].items():
        cell["pack_status"] = {}
        cell["pack_step"] = {
            "_type": "step", "address": "local:EcoliPackStep",
            "config": {"snapshots": snaps, "study": study, "out_dir": out_dir,
                       "top_n": top_n, "scale": scale},
            "inputs": {"bulk": ["bulk"], "shape": ["shape"],
                       "global_time": ["global_time"],
                       "full_chromosomes": ["full_chromosomes"]},
            "outputs": {"pack_status": ["pack_status"]},
        }
        # append as a final execution layer + rewire flow (mirror baseline's
        # shape_step append, baseline.py:895-919)
        _append_final_step(cell, "pack_step")
    return doc
```

Implement `_append_final_step(cell, step_name)` to reproduce baseline's final-layer append: read the cell's execution layers / `flow_order`, append `[step_name]`, and call `inject_flow_dependencies(cell, flow_order, layers=...)`. If baseline exposes its layers on the returned doc, reuse them; otherwise import `inject_flow_dependencies` from where baseline uses it and rebuild `flow_order` from the doc's existing steps + the appended step. **Verify against `baseline.py:895-919`** and match its exact call.

Register in `v2ecoli/composites/__init__.py`: add `from v2ecoli.composites import baseline_parsimony  # noqa: F401` next to the other generator imports, and reconcile the `__all__` entry (replace/duplicate the dangling `"parsimony_ecoli"` stub, `__init__.py:74`, with `"baseline_parsimony"`).

- [ ] **Step 4: Run — verify it passes**

Run: `… -m pytest tests/test_baseline_parsimony_composite.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/baseline_parsimony.py v2ecoli/composites/__init__.py tests/test_baseline_parsimony_composite.py
git commit -m "feat: baseline_parsimony generator (baseline + EcoliPackStep)"
```

---

### Task 5: Integration — a real snapshot end-to-end (fast + opt-in slow)

**Files:**
- Test: `tests/test_baseline_parsimony_integration.py`

**Interfaces:**
- Consumes: the registered `baseline_parsimony` generator + a real (small) `build_pack`.

- [ ] **Step 1: Write the fast integration test (reaches the `initial` snapshot only)**

```python
# tests/test_baseline_parsimony_integration.py
import json, pytest

def test_initial_snapshot_pack_written(tmp_path, monkeypatch):
    """Run baseline_parsimony far enough to cross the 'initial' time and assert a
    valid initial.pack.json lands under the study's viz/3d. Uses a tiny top_n and
    an early initial time so it stays fast; pre-division is NOT exercised here."""
    import v2ecoli
    from v2ecoli.core import build_core
    monkeypatch.chdir(tmp_path)  # study dir is written relative to CWD
    core = build_core()
    comp = v2ecoli.build_composite(
        "ecoli_structural", core=core, seed=0, cache_dir="out/cache",
        study="itest", snapshots={"initial": 2.0}, top_n=2, emitter="null")
    comp.run(4.0)  # cross global_time=2.0
    pack = tmp_path / "studies" / "itest" / "viz" / "3d" / "initial.pack.json"
    assert pack.is_file()
    doc = json.loads(pack.read_text())
    assert doc["format"] == "parsimony.pack.v1"

@pytest.mark.slow
def test_pre_division_snapshot(tmp_path, monkeypatch):
    """Full generation to division → both packs. Opt-in (slow): runs a real
    baseline generation and packs ~1.3M molecules twice."""
    import v2ecoli
    from v2ecoli.core import build_core
    monkeypatch.chdir(tmp_path)
    core = build_core()
    comp = v2ecoli.build_composite("ecoli_structural", core=core, seed=0,
                                   cache_dir="out/cache", study="itest", emitter="null")
    comp.run(3000.0)  # to/through division
    d = tmp_path / "studies" / "itest" / "viz" / "3d"
    assert (d / "initial.pack.json").is_file()
    assert (d / "pre-division.pack.json").is_file()
```

- [ ] **Step 2: Run the fast test**

Run: `… -m pytest tests/test_baseline_parsimony_integration.py -v -m "not slow"`
Expected: PASS (initial pack written). If the run needs a ParCa cache, symlink `out/cache` per the v2ecoli worktree-cache convention before running (see memory `v2ecoli worktree cache symlink`); if the fast run is still too slow (>~2 min), lower `snapshots={"initial": 0.5}` and the `comp.run` horizon.

- [ ] **Step 3: Confirm the slow test is opt-in**

Run: `… -m pytest tests/test_baseline_parsimony_integration.py -m slow --collect-only` — shows `test_pre_division_snapshot` collected only under `-m slow`. Register the `slow` marker in `pyproject.toml`/`pytest.ini` if not already present.

- [ ] **Step 4: One manual real end-to-end (out of the test suite)**

Run the slow test once manually (or `build_composite(...).run(...)` in a scratch script) to confirm BOTH packs render in the workbench Parsimony Viewer gallery (Initial · Pre-division) for the `ecoli-3d` study. Record the run time in the PR.

- [ ] **Step 5: Commit**

```bash
git add tests/test_baseline_parsimony_integration.py pyproject.toml
git commit -m "test: baseline_parsimony integration (fast initial + opt-in pre-division)"
```

---

## Self-Review

**Spec coverage:**
- `baseline_parsimony` generator wrapping baseline + final-layer step → Task 4. ✔
- Declared-time gating (initial fixed time; pre-division via `full_chromosomes.division_time`) → Task 3. ✔
- Geometry from `volume_fl` via `Capsule.from_volume_fl` (not a stored object) → Task 2 (`pack_from_state`). ✔
- Live bulk → counts conversion → Task 2 (`bulk_to_counts`). ✔
- Packs to `studies/<study>/viz/3d/{initial,pre-division}.pack.json`, gallery-named → Tasks 3–4 (`out_dir`, snapshot names). ✔
- Revive `v2e-3d` packing machinery into `v2ecoli/structural/` → Task 1. ✔
- Fast-gating + opt-in-slow tests → Tasks 3, 5. ✔
- Viewer contract (names) → Global Constraints + Task 5 manual check. ✔

**Placeholder scan:** No TBD/TODO. Two spots flag "verify against the real code" with the exact anchor to match (`_append_final_step` vs `baseline.py:895-919`; `config_schema` style vs `ShapeStep`) — these are verification directives with a named target, not vague requirements.

**Type consistency:** `bulk_to_counts(bulk) -> dict` and `pack_from_state(out_dir, name, counts, volume_fl, *, top_n, scale, proxy_lod)` defined in Task 2, consumed identically in Task 3. `EcoliPackStep` config keys (`snapshots`, `study`, `out_dir`, `top_n`, `scale`, `epsilon_s`) defined in Task 3, set identically in Task 4's step node. Input ports (`bulk`/`shape`/`global_time`/`full_chromosomes`) match between Task 3 `inputs()` and Task 4's wiring.
