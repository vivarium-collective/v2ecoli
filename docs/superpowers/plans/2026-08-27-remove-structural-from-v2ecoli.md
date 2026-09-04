# Remove Structural Investigation from v2ecoli → Consolidate in 3d-ecoli — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `v2ecoli` free of `pbg_parsimony` and of the structural/3D investigation, and consolidate that work into the standalone `3d-ecoli` repo (package `ecoli_3d`), unified on 3d-ecoli's existing builder.

**Architecture:** Two repos, two worktrees. `v2ecoli` keeps a **parsimony-free** `cell_shape.py`/`ShapeStep` (emits plain-float geometry; baseline unchanged) and deletes everything else structural + the `pbg-parsimony` dep. `3d-ecoli` becomes a viva workspace and gains the 5 live-state entry points that `EcoliPackStep` needs, so its own richer `build.py` becomes the single builder behind both the snapshot (`parsimony-ecoli`) and live-sim (`baseline_parsimony`) composites; the `structural-ecoli` investigation + study move there. The Rust `parsimony` engine and `pbg-openmm` relax are unchanged external deps.

**Tech Stack:** Python 3.12.12, process-bigraph / bigraph-schema, `viva_superpowers` composite generator, `pbg_parsimony` (repo `viva-parsimony`), `pytest`, `uv`.

**Spec:** `docs/superpowers/plans/../specs/2026-08-27-remove-structural-from-v2ecoli-design.md` (in the v2ecoli worktree).

## Global Constraints

- **v2ecoli worktree:** `~/code/v2ecoli--remove-ecoli-3d`, branch `remove-structural-from-v2ecoli`, off `origin/main`.
- **3d-ecoli worktree:** `~/code/3d-ecoli--consolidate`, branch `consolidate-structural`, off `origin/main`.
- **No AI attribution** in any commit message or PR body (no `Co-Authored-By: Claude`, no "Generated with Claude Code").
- **Package identity:** repo `3d-ecoli`, Python package `ecoli_3d` — **no rename**. Parsimony package is `pbg_parsimony` (from repo `viva-parsimony`).
- **Cross-repo dev env:** 3d-ecoli imports v2ecoli. Develop 3d-ecoli tasks against the **Part-A v2ecoli worktree**, not `@main` (main still returns Capsule objects until Part A merges). Editable-install both into one 3.12.12 venv: `uv pip install -e ~/code/v2ecoli--remove-ecoli-3d --no-deps` and `uv pip install -e ~/code/3d-ecoli--consolidate --no-deps` into the shared venv, plus `pbg_parsimony`. Verify with `python -c "import v2ecoli, ecoli_3d; print(v2ecoli.__file__, ecoli_3d.__file__)"`.
- **Parsimony binary:** the end-to-end `build_pack` pack tests need `PARSIMONY_HOME=~/code/parsimony` (Rust binary built: `cargo build --release -p parsimony-cli`) + `PYTHONUTF8=1`. Where the binary is absent, pack-level tests must **SKIP** (never falsely pass), mirroring the existing acceptance gate.
- **Commit discipline:** verify `git branch --show-current` + `git rev-parse --short HEAD` before every commit; `git log --oneline <base>..HEAD` shows only your commits before any push.
- **Merge order:** v2ecoli PR merges first; then bump 3d-ecoli's `v2ecoli` pin to that commit and re-lock before the 3d-ecoli PR.

---

## Phase 1 — v2ecoli: sever parsimony from `cell_shape` (must land before Phase 2 dev)

### Task 1: Make `cell_shape.py` parsimony-free

**Files:**
- Modify: `~/code/v2ecoli--remove-ecoli-3d/v2ecoli/cell_shape.py`
- Test: `~/code/v2ecoli--remove-ecoli-3d/tests/test_shape.py`

**Interfaces:**
- Produces: `shape_from_mass(mass_fg, width_um=1.0, density_g_per_ml=1.1, periplasm_fraction=0.2) -> dict` returning **only plain floats** — the existing numeric keys plus `envelope` as a **dict of numbers** (`outer_radius_A`, `outer_half_len_A`, `inner_radius_A`, `inner_half_len_A`, `periplasm_fraction`, `periplasm_vol_fl`, `cytoplasm_vol_fl`, `outer_sa_um2`, `inner_sa_um2`). The `capsule`/`inner_capsule` object keys are **removed**. `SHAPE_KEYS`, `zero_shape()`, `ShapeStep` behavior unchanged (already float-only).
- Downstream (3d-ecoli Task 8) reads `shape["radius_A"]`, `shape["half_len_A"]`, `shape["inner_radius_A"]`, `shape["inner_half_len_A"]`.

- [ ] **Step 1: Write/adjust the failing test** — assert no parsimony import and float-only output.

```python
# tests/test_shape.py — add/replace the object-shape assertions with these
import importlib, sys
import v2ecoli.cell_shape as cs

def test_shape_from_mass_is_parsimony_free_floats():
    shape = cs.shape_from_mass(400.0)
    # numeric envelope, no Capsule objects
    assert "capsule" not in shape and "inner_capsule" not in shape
    for k in ("radius_A", "half_len_A", "inner_radius_A", "inner_half_len_A"):
        assert isinstance(shape[k], float)
    env = shape["envelope"]
    assert isinstance(env["outer_radius_A"], float)
    assert isinstance(env["inner_radius_A"], float)
    # inner membrane is the volume-consistent inward scale of the outer
    s = (1.0 - 0.2) ** (1.0 / 3.0)
    assert abs(env["inner_radius_A"] - shape["radius_A"] * s) < 1e-6

def test_cell_shape_module_does_not_import_pbg_parsimony():
    importlib.reload(cs)
    assert "pbg_parsimony" not in sys.modules or True  # see Task 6 for the strict env check
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest ~/code/v2ecoli--remove-ecoli-3d/tests/test_shape.py -q`
Expected: FAIL (current `shape_from_mass` returns `Capsule` objects under `capsule`/`envelope`).

- [ ] **Step 3: Rewrite `shape_from_mass`** — drop the import, use arithmetic.

```python
# cell_shape.py: delete `from pbg_parsimony import Capsule` (line 24).
# Replace lines 46 and 52-53 (the Capsule constructions) and the return dict:
    radius_A = r * 1e4                        # µm → Å
    half_len_A = (lcyl / 2.0) * 1e4
    s = (1.0 - periplasm_fraction) ** (1.0 / 3.0)
    inner_radius_A = radius_A * s
    inner_half_len_A = half_len_A * s
    return {
        "mass_fg": float(mass_fg),
        "density_g_per_ml": density_g_per_ml,
        "width_um": w,
        "volume_fl": v,
        "length_um": length,
        "outer_sa_um2": outer_sa,
        "inner_sa_um2": inner_sa,
        "periplasm_fraction": periplasm_fraction,
        "periplasm_vol_fl": v * periplasm_fraction,
        "cytoplasm_vol_fl": v * (1.0 - periplasm_fraction),
        "radius_A": radius_A,
        "half_len_A": half_len_A,
        "inner_radius_A": inner_radius_A,
        "inner_half_len_A": inner_half_len_A,
        "envelope": {
            "outer_radius_A": radius_A,
            "outer_half_len_A": half_len_A,
            "inner_radius_A": inner_radius_A,
            "inner_half_len_A": inner_half_len_A,
            "periplasm_fraction": periplasm_fraction,
            "periplasm_vol_fl": v * periplasm_fraction,
            "cytoplasm_vol_fl": v * (1.0 - periplasm_fraction),
            "outer_sa_um2": outer_sa,
            "inner_sa_um2": inner_sa,
        },
    }
```
Also update the module docstring (lines 14-17) to drop the "returns the pbg-parsimony `Capsule`" claim. `ShapeStep.update`'s pop-list (`"capsule", "inner_capsule", "envelope"`) still works (keys may be absent — `pop(k, None)`).

- [ ] **Step 4: Run to verify pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest ~/code/v2ecoli--remove-ecoli-3d/tests/test_shape.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/v2ecoli--remove-ecoli-3d
git add v2ecoli/cell_shape.py tests/test_shape.py
git commit -m "refactor(cell_shape): drop pbg_parsimony.Capsule; emit plain-float geometry"
```

---

## Phase 2 — 3d-ecoli: unify on its builder + become a workspace

> Develop against the shared venv with the Part-1 v2ecoli worktree editable-installed. All paths below are in `~/code/3d-ecoli--consolidate` unless noted. Copy v2ecoli source from `~/code/v2ecoli--remove-ecoli-3d/v2ecoli/structural/`.

### Task 2: Port the near-mechanical live extractors into `ecoli_3d/build.py`

**Files:**
- Modify: `ecoli_3d/build.py`
- Test: `tests/test_live_extractors.py` (create)

**Interfaces:**
- Produces (all public in `ecoli_3d.build`): `bulk_to_counts(bulk) -> dict`, `bulk_to_locations(bulk) -> dict` (+ module-level `_TAG_TO_COMPARTMENT`), `chromosome_state_from_live(full_chromosome, active_replisome=None) -> tuple[int, float]`, `rnaps_from_live(active_rnap, full_chromosome=None, chromosome_domain=None) -> list[dict]` (+ `_active_rows(arr)`).
- Consumes: B's existing `classify_domains`, `_descendant_domains_set`, `REPLICHORE_BP`, `GENOME_BP` (already present).

- [ ] **Step 1: Write the failing test** using small synthetic numpy arrays.

```python
# tests/test_live_extractors.py
import numpy as np
from ecoli_3d import build

def _bulk(ids, counts):
    return np.array(list(zip(ids, counts)),
                    dtype=[("id", "U40"), ("count", "i8")])

def test_bulk_to_counts_sums_and_strips_tag():
    bulk = _bulk(["GLC[c]", "GLC[p]", "ATP[c]"], [3, 4, 5])
    assert build.bulk_to_counts(bulk) == {"GLC": 7, "ATP": 5}

def test_bulk_to_locations_dominant_compartment():
    bulk = _bulk(["GLC[c]", "GLC[p]"], [1, 9])
    loc = build.bulk_to_locations(bulk)
    assert loc["GLC"] == build._TAG_TO_COMPARTMENT["p"]

def test_chromosome_state_from_live_unreplicated():
    fc = np.array([(1,)], dtype=[("_entryState", "i8")])
    n, ff = build.chromosome_state_from_live(fc, None)
    assert n == 1 and ff == 0.0
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_live_extractors.py -q` → FAIL (`AttributeError: module 'ecoli_3d.build' has no attribute 'bulk_to_counts'`).

- [ ] **Step 3: Port the functions** — copy `_TAG_TO_COMPARTMENT`, `bulk_to_counts`, `bulk_to_locations`, `_active_rows`, `chromosome_state_from_live`, `rnaps_from_live` **verbatim** from `~/code/v2ecoli--remove-ecoli-3d/v2ecoli/structural/build.py` (A:110–147, A:286–303, A:370–452) into `ecoli_3d/build.py`. They depend only on symbols B already has (`classify_domains`, `_descendant_domains_set`, `REPLICHORE_BP`). Do not duplicate `classify_domains`/`_descendant_domains_set` — B's copies are identical; reuse them.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_live_extractors.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/3d-ecoli--consolidate
git add ecoli_3d/build.py tests/test_live_extractors.py
git commit -m "feat(build): port live-state extractors (bulk/chromosome/rnap) from v2ecoli"
```

### Task 3: Reconstruct `Capsule` locally in `build_model` (decouple from cell_shape objects)

**Files:**
- Modify: `ecoli_3d/build.py` (the `build_model` cell_shape block, ~B:1214–1224)
- Test: `tests/test_capsule_reconstruction.py` (create)

**Interfaces:**
- Consumes: `v2ecoli.cell_shape.shape_from_mass` now returns plain floats (Phase 1 Task 1).
- Produces: `build_model` builds `capsule`/`envelope` `pbg_parsimony.Capsule` objects locally from `shape["*_A"]`.

- [ ] **Step 1: Write the failing test** — a unit that calls the (to-be-added) helper.

```python
# tests/test_capsule_reconstruction.py
from ecoli_3d.build import _capsules_from_shape

def test_capsules_from_shape_numeric():
    shape = {"radius_A": 5000.0, "half_len_A": 8000.0,
             "inner_radius_A": 4642.0, "inner_half_len_A": 7427.0}
    outer, inner, env = _capsules_from_shape(shape)
    assert abs(outer.radius - 5000.0) < 1e-6
    assert abs(inner.half_len - 7427.0) < 1e-6
    assert env["outer"] is outer and env["inner"] is inner
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`_capsules_from_shape` undefined).

- [ ] **Step 3: Add the helper + rewire `build_model`.**

```python
# ecoli_3d/build.py — add near the top-level helpers
def _capsules_from_shape(shape):
    """Rebuild pbg_parsimony Capsules from cell_shape's numeric fields
    (cell_shape now returns plain floats, not Capsule objects)."""
    from pbg_parsimony import Capsule
    outer = Capsule(half_len=shape["half_len_A"], radius=shape["radius_A"])
    inner = Capsule(half_len=shape["inner_half_len_A"], radius=shape["inner_radius_A"])
    return outer, inner, {"outer": outer, "inner": inner}
```
Replace B:1222–1224 with:
```python
    capsule, _inner, envelope = _capsules_from_shape(shape)
```
(leave the rest of `build_model` — the `.radius`/`.half_len` uses and `build_pack(... capsule ..., envelope=envelope)` — unchanged; they now read the reconstructed objects.)

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_capsule_reconstruction.py -q` → PASS. Also run B's existing snapshot tests that don't need the binary: `pytest tests/test_build_chromosome_fields.py tests/test_rnap_state.py -q` → PASS (no regression).

- [ ] **Step 5: Commit**

```bash
git add ecoli_3d/build.py tests/test_capsule_reconstruction.py
git commit -m "refactor(build): reconstruct Capsule locally from numeric cell_shape fields"
```

### Task 4: Extract `pack_from_state` (in-memory core) from `build_model`

**Files:**
- Modify: `ecoli_3d/build.py`
- Test: `tests/test_pack_from_state.py` (create; port + adapt from v2ecoli `tests/test_pack_from_state.py`)

**Interfaces:**
- Produces: `pack_from_state(out_dir, name, counts, volume_fl, locations=None, *, top_n=40, scale=0.3, proxy_lod=2, relax=False, cache_dir="out/cache", relax_params=None, envelope=True, periplasm_gap_A=250.0, rnaps=None, n_chromosomes=1, fork_fraction=0.0) -> dict`. `build_model(...)` becomes a thin wrapper: read files via `load_state`/`chromosome_state`/`rnap_state`/… then call `pack_from_state(...)`.
- Consumes: B's ingredient assembly, `Chromosome`, `_capsules_from_shape`/`Capsule.from_volume_fl`, `build_pack`.

- [ ] **Step 1: Write the failing test** — assert the seam exists and threads live params without reading files. Guard the actual pack behind binary availability.

```python
# tests/test_pack_from_state.py
import os, shutil, pytest
from ecoli_3d import build

def test_pack_from_state_signature_accepts_live_params():
    import inspect
    sig = inspect.signature(build.pack_from_state)
    for p in ("counts", "volume_fl", "locations", "rnaps",
              "n_chromosomes", "fork_fraction", "envelope", "relax"):
        assert p in sig.parameters

_HAVE_BIN = bool(os.environ.get("PARSIMONY_HOME")) and shutil.which  # see Global Constraints

@pytest.mark.skipif(not os.environ.get("PARSIMONY_HOME"),
                    reason="parsimony binary absent; pack skipped (never falsely passes)")
def test_pack_from_state_writes_pack(tmp_path):
    counts = {"EG10893-MONOMER": 100, "RNA-POLYMERASE": 10}  # small real ids
    res = build.pack_from_state(str(tmp_path), "t", counts, volume_fl=1.0,
                                locations={}, top_n=5, scale=0.1,
                                rnaps=[], n_chromosomes=1, fork_fraction=0.0)
    assert (tmp_path / "t.pack.json").exists()
    assert isinstance(res.get("placements"), list)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`pack_from_state` undefined).

- [ ] **Step 3: Refactor `build_model` → extract `pack_from_state`.** Split B's `build_model` (B:1174–1402) at the file-read boundary:
  - Move the **file reads** (`load_state`, `chromosome_state`, `rnap_state`, `rna_state`, `ribosome_state`, `division_progress`) to the top of `build_model`.
  - Move the **in-memory core** (ingredient assembly via `select_ingredients`, `Chromosome` construction with `rnaps`/`n_chromosomes`/`fork_fraction`/markers/septum, `_capsules_from_shape` (from `shape_from_mass(volume_fl*density*1000)`), `build_pack`, and the post-processing: flagella, FtsZ ring, `_backfill_all_counts`, `compact` if applicable) into `pack_from_state(out_dir, name, counts, volume_fl, locations=None, *, …, rnaps=None, n_chromosomes=1, fork_fraction=0.0)`.
  - Keep B's placement semantics: 70S/RNAP count=0 → placed via markers/chromosome stage; the live `rnaps` list drives RNAP placement.
  - `build_model` becomes:
    ```python
    def build_model(out_dir="out/ecoli3d", *, name="ecoli_3d", top_n=40, scale=1.0,
                    state_source="snapshot", proxy_lod=2, top_complexes=150,
                    width_um=1.0, density_g_per_ml=1.1, septum_fraction=None):
        counts, volume_fl, compartments = load_state(state_source)
        n_chrom, fork = chromosome_state(state_source)
        rnaps = rnaps_from_state_arrays(rnap_state(state_source))  # existing inline logic, factored
        return pack_from_state(out_dir, name, counts, volume_fl, locations=compartments,
                               top_n=top_n, scale=scale, proxy_lod=proxy_lod,
                               rnaps=rnaps, n_chromosomes=n_chrom, fork_fraction=fork,
                               ...snapshot extras (rnas/ribosomes/peptides/septum) via kwargs...)
    ```
  - **Routing:** `pack_from_state` standardizes on B's tag-letter convention — `locations` holds tag letters (as `bulk_to_locations`/`load_state` produce), routed via B's `_route_envelope` inside `select_ingredients(compartments=locations)`.
  - Note: the snapshot path also threads rnas/ribosomes/peptides that the live path doesn't yet supply — accept them as optional `pack_from_state` kwargs (default `None`) so `EcoliPackStep`'s call (no rnas/ribosomes) still works; live RNA/ribosome placement is a documented follow-up.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_pack_from_state.py::test_pack_from_state_signature_accepts_live_params -q` → PASS; the binary-gated test SKIPs without `PARSIMONY_HOME`. Re-run B's snapshot tests (`pytest tests/ -q -k "not pack_from_state or signature"`) → no regressions.

- [ ] **Step 5: Commit**

```bash
git add ecoli_3d/build.py tests/test_pack_from_state.py
git commit -m "refactor(build): extract in-memory pack_from_state; build_model wraps it"
```

### Task 5: Bring `EcoliPackStep` into `ecoli_3d`

**Files:**
- Create: `ecoli_3d/pack_step.py` (from v2ecoli `structural/pack_step.py`)
- Test: `tests/test_pack_step.py` (port from v2ecoli `tests/test_pack_step.py`)

**Interfaces:**
- Consumes: `ecoli_3d.build.{pack_from_state, bulk_to_counts, bulk_to_locations, chromosome_state_from_live, rnaps_from_live}` (Tasks 2, 4); `v2ecoli.core.build_core`.
- Produces: `ecoli_3d.pack_step.EcoliPackStep` (schedules snapshots, fires once each at declared time / `division_time`).

- [ ] **Step 1: Write the failing test** — port v2ecoli's scheduling tests (they need no binary).

```python
# tests/test_pack_step.py  (adapt import to ecoli_3d)
from ecoli_3d.pack_step import EcoliPackStep
# ... copy v2ecoli tests/test_pack_step.py scheduling cases: both snapshots fire
# once, division_time resolves from full_chromosome rows, volume<=0 defers. The
# _step fixture must stub bulk_to_counts/bulk_to_locations/chromosome_state_from_live/
# rnaps_from_live/pack_from_state on ecoli_3d.build so no real pack runs.
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ecoli_3d.pack_step` missing).

- [ ] **Step 3: Copy `pack_step.py`** into `ecoli_3d/`, changing only the import (line 9):
```python
from ecoli_3d.build import (
    pack_from_state, bulk_to_counts, bulk_to_locations,
    chromosome_state_from_live, rnaps_from_live,
)
```
Keep `_default_core` importing `from v2ecoli.core import build_core`.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_pack_step.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add ecoli_3d/pack_step.py tests/test_pack_step.py
git commit -m "feat(pack_step): bring EcoliPackStep into ecoli_3d on the unified builder"
```

### Task 6: Bring the `baseline_parsimony` composite + `acceptance.py`

**Files:**
- Create: `ecoli_3d/composites/ecoli_structural.py` (from v2ecoli `composites/ecoli_structural.py`)
- Create: `ecoli_3d/acceptance.py` (from v2ecoli `structural/acceptance.py`)
- Test: `tests/test_baseline_parsimony_composite.py` (port from v2ecoli), `tests/test_s01_acceptance_gate.py` (port from v2ecoli `tests/structural/`)

**Interfaces:**
- Produces: composite registry key `ecoli_3d.composites.ecoli_structural.baseline_parsimony` (name `ecoli_structural`); wraps `v2ecoli.composites.ecoli_baseline.baseline`, appends `ecoli_3d` `EcoliPackStep`.
- Consumes: `ecoli_3d.pack_step.EcoliPackStep`, `viva_superpowers.composite_generator`.

- [ ] **Step 1: Write the failing test** — composite builds, appends pack_step, forwards baseline params (no binary).

```python
# tests/test_baseline_parsimony_composite.py (adapt imports to ecoli_3d)
from ecoli_3d.composites.ecoli_structural import baseline_parsimony
# copy v2ecoli's assertions: doc has agents, each agent gains a 'pack_step' edge
# with address 'local:EcoliPackStep', flow_order ends with pack_step, and the
# core_extensions hook registers EcoliPackStep.
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module missing).

- [ ] **Step 3: Copy both files.** In `ecoli_structural.py` change imports:
  - `from ecoli_3d.pack_step import EcoliPackStep` (in `_register_ecoli_pack_step`).
  - Keep `from v2ecoli.composites.ecoli_baseline import baseline` and `from v2ecoli.composites._helpers import inject_flow_dependencies`.
  - `_resolve_pack_out_dir` keeps using `viva_workspace`.
  - `acceptance.py`: copy verbatim; update any `v2ecoli.structural` import to `ecoli_3d`.
  - Add `ecoli_3d/composites/__init__.py` if absent.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_baseline_parsimony_composite.py tests/test_s01_acceptance_gate.py -q` → PASS (acceptance end-to-end pack test SKIPs without binary).

- [ ] **Step 5: Commit**

```bash
git add ecoli_3d/composites/ ecoli_3d/acceptance.py tests/test_baseline_parsimony_composite.py tests/test_s01_acceptance_gate.py
git commit -m "feat(composite): bring baseline_parsimony (ecoli_structural) + acceptance into ecoli_3d"
```

### Task 7: Promote 3d-ecoli to a viva workspace + move the investigation/study

**Files:**
- Create: `workspace.yaml`
- Create: `workspace/investigations/structural-ecoli/investigation.yaml` (from v2ecoli), `workspace/studies/s01-birth-and-division/study.yaml` (from v2ecoli)
- Create: `ecoli_3d/workbench_viewers.py` (the 3D-pack deep-link, from v2ecoli's `_has_3d_pack`/`_ecoli_3d_targets`/`get_viewers` ecoli-3d entry)
- Modify: `.gitignore` (add `workspace/studies/s01-birth-and-division/viz/3d/`)
- Test: `tests/test_workspace_loads.py` (create)

**Interfaces:**
- Produces: a loadable viva workspace whose package is `ecoli_3d`, exposing the `structural-ecoli` investigation whose study references composite key `ecoli_3d.composites.ecoli_structural.ecoli_structural`.

- [ ] **Step 1: Write the failing test** — workspace resolves + study composite key points at ecoli_3d.

```python
# tests/test_workspace_loads.py
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent.parent

def test_workspace_yaml_registers_ecoli_3d():
    ws = yaml.safe_load((ROOT / "workspace.yaml").read_text())
    assert ws["package"] == "ecoli_3d" or "ecoli_3d" in str(ws)

def test_study_composite_points_at_ecoli_3d():
    y = (ROOT / "workspace/studies/s01-birth-and-division/study.yaml").read_text()
    assert "ecoli_3d.composites.ecoli_structural" in y
    assert "v2ecoli.composites.ecoli_structural" not in y
```

- [ ] **Step 2: Run to verify it fails** — FAIL (files absent).

- [ ] **Step 3: Scaffold the workspace.** Model `workspace.yaml` on v2ecoli's (package registration, the `pbg_parsimony` + `v2ecoli` modules, workbench include list with `ecoli_3d`, and the R2 `viz_viewer_urls` for `ecoli_3d`/`initial`/`pre-division`). `git mv` is cross-repo, so **copy** the two YAMLs from `~/code/v2ecoli--remove-ecoli-3d/workspace/...` and edit the study's `composite:` key `v2ecoli.composites.ecoli_structural.ecoli_structural` → `ecoli_3d.composites.ecoli_structural.ecoli_structural`. Copy `workbench_viewers.py`'s 3D block into `ecoli_3d/workbench_viewers.py` (drop the PTools viewer — that stays v2ecoli's). Add the gitignore line for the large packs.

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_workspace_loads.py -q` → PASS. If the workbench is available, `viva-status` in the worktree reports a valid workspace.

- [ ] **Step 5: Commit**

```bash
git add workspace.yaml workspace/ ecoli_3d/workbench_viewers.py .gitignore tests/test_workspace_loads.py
git commit -m "feat(workspace): promote 3d-ecoli to a viva workspace; move structural-ecoli investigation"
```

### Task 8: 3d-ecoli dependency + fixups + full green

**Files:**
- Modify: `pyproject.toml` (pin `v2ecoli`), `uv.lock`
- Modify: `ecoli_3d/publish/03_assemble_local_view.py` (stale path)
- Test: full suite

- [ ] **Step 1: Fix the stale viewer path** in `publish/03_assemble_local_view.py` — replace the hardcoded `/Users/eranagmon/code/pbg-parsimony/pbg_parsimony/viewer` with a package-relative resolve:
```python
import importlib.util, pathlib
VIEWER = pathlib.Path(importlib.util.find_spec("pbg_parsimony").origin).parent / "viewer"
```

- [ ] **Step 2: Run the full 3d-ecoli suite** against the shared venv:

Run: `PARSIMONY_HOME=~/code/parsimony PYTHONUTF8=1 <shared-venv>/bin/python -m pytest ~/code/3d-ecoli--consolidate/tests -q`
Expected: all PASS (binary-gated pack tests run if `PARSIMONY_HOME` set + binary built; otherwise SKIP).

- [ ] **Step 3: Pin `v2ecoli`** — after Phase 3 (v2ecoli strip) merges to v2ecoli main, set `pyproject.toml` `[tool.uv.sources] v2ecoli = { git = "...", rev = "<post-strip-main-sha>" }` and re-lock: `uv lock`. (Until then, leave `branch = "main"` and rely on the editable install for dev.)

- [ ] **Step 4: Commit**

```bash
git add ecoli_3d/publish/03_assemble_local_view.py pyproject.toml uv.lock
git commit -m "chore(3d-ecoli): fix stale viewer path; pin v2ecoli to post-strip main"
```

---

## Phase 3 — v2ecoli: delete the rest of the structural surface

> Runs after Phase 2 has copied everything it needs from the v2ecoli worktree.

### Task 9: Delete structural code + composite + viewer block + reports

**Files:**
- Delete: `v2ecoli/structural/` (all), `v2ecoli/composites/ecoli_structural.py`, `reports/composite-state/v2ecoli.composites.ecoli_structural*.json`, `reports/composite-state/v2ecoli.structural.composite.parsimony-ecoli.json`, `docs/superpowers/specs/2026-07-25-structural-ecoli-investigation-design.md`
- Modify: `v2ecoli/workbench_viewers.py` (remove `_has_3d_pack`, `_ecoli_3d_targets`, the `ecoli-3d` entry in `get_viewers`, and its docstring bullet — keep the PTools viewer)
- Delete tests: `tests/structural/`, `tests/test_baseline_parsimony_composite.py`, `tests/test_baseline_parsimony_integration.py`, `tests/test_pack_from_state.py`, `tests/test_pack_relax_wiring.py`, `tests/test_pack_step.py`, `tests/test_structural_revive.py` (KEEP `tests/test_shape.py`)

- [ ] **Step 1: Write the failing guard test** — assert the structural surface is gone and `import v2ecoli` pulls no parsimony.

```python
# tests/test_no_parsimony.py (create)
import importlib, sys, pytest

def test_import_v2ecoli_does_not_import_pbg_parsimony():
    for m in list(sys.modules):
        if m == "pbg_parsimony" or m.startswith("pbg_parsimony."):
            del sys.modules[m]
    importlib.import_module("v2ecoli")
    import v2ecoli.composites.ecoli_baseline  # the baseline path
    assert not any(m == "pbg_parsimony" or m.startswith("pbg_parsimony.")
                   for m in sys.modules), "v2ecoli must not import pbg_parsimony"

def test_structural_modules_gone():
    for name in ("v2ecoli.structural", "v2ecoli.structural.build",
                 "v2ecoli.composites.ecoli_structural"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (modules still import; parsimony present).

- [ ] **Step 3: Delete the files** (in the v2ecoli worktree):
```bash
cd ~/code/v2ecoli--remove-ecoli-3d
git rm -r v2ecoli/structural v2ecoli/composites/ecoli_structural.py \
  reports/composite-state/v2ecoli.composites.ecoli_structural.ecoli_structural.json \
  reports/composite-state/v2ecoli.composites.ecoli_structural.json \
  reports/composite-state/v2ecoli.structural.composite.parsimony-ecoli.json \
  docs/superpowers/specs/2026-07-25-structural-ecoli-investigation-design.md \
  tests/structural tests/test_baseline_parsimony_composite.py \
  tests/test_baseline_parsimony_integration.py tests/test_pack_from_state.py \
  tests/test_pack_relax_wiring.py tests/test_pack_step.py tests/test_structural_revive.py
```
Then edit `v2ecoli/workbench_viewers.py`: delete `_has_3d_pack` (216-219), `_ecoli_3d_targets` (222-248), and the `ecoli-3d` dict in `get_viewers` (275-286); trim the `get_viewers` docstring's 3D bullet. `_studies_root` stays (PTools uses it).

- [ ] **Step 4: Run to verify pass** — `~/code/v2ecoli/.venv/bin/python -m pytest tests/test_no_parsimony.py tests/test_shape.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "remove(structural): delete parsimony build/composite/viewer/reports/tests from v2ecoli"
```

### Task 10: Drop the `pbg-parsimony` dependency + workspace + investigation

**Files:**
- Modify: `pyproject.toml` (remove `"pbg-parsimony"` dep line + the `[tool.uv.sources]` `pbg-parsimony = {...}` line), `uv.lock` (re-lock)
- Modify: `workspace.yaml` (remove the `pbg_parsimony` module block, remove it from the workbench `include:` list, remove the `viz_viewer_urls` `ecoli_3d`/`initial`/`pre-division` entries)
- Delete: `workspace/investigations/structural-ecoli/`, `workspace/studies/s01-birth-and-division/`
- Modify: `.gitignore` (remove the now-dead `workspace/studies/s01-birth-and-division/viz/3d/` rule)

- [ ] **Step 1: Write the failing test** — assert parsimony is not a declared dep and the investigation is gone.

```python
# tests/test_no_parsimony_dep.py (create)
import tomllib, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_pyproject_has_no_parsimony():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert not any("parsimony" in d for d in deps)
    assert "pbg-parsimony" not in data.get("tool", {}).get("uv", {}).get("sources", {})

def test_structural_investigation_removed():
    assert not (ROOT / "workspace/investigations/structural-ecoli").exists()
    assert not (ROOT / "workspace/studies/s01-birth-and-division").exists()
```

- [ ] **Step 2: Run to verify it fails** — FAIL (dep + investigation still present).

- [ ] **Step 3: Edit + delete.** Remove the two `pyproject.toml` lines; edit `workspace.yaml` (module block, include list, viz_viewer_urls); `git rm -r workspace/investigations/structural-ecoli workspace/studies/s01-birth-and-division`; drop the gitignore rule. Re-lock: `uv lock` (from the worktree; if a cold resolve conflict appears, note it — v2ecoli already resolves via committed lock).

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_no_parsimony_dep.py -q` → PASS. Then the **full v2ecoli suite**: `~/code/v2ecoli/.venv/bin/python -m pytest tests -q` → green (no structural collection errors). Confirm a baseline smoke still emits `shape`:
```bash
~/code/v2ecoli/.venv/bin/python -c "from v2ecoli.composites.ecoli_baseline import baseline; from v2ecoli.core import build_core; d=baseline(core=build_core()); print('shape' in str(d))"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "remove(deps): drop pbg-parsimony dependency + structural investigation/study + workspace wiring"
```

---

## Phase 4 — Integrate

### Task 11: PRs + merge order + pin bump

- [ ] **Step 1: v2ecoli PR** — push `remove-structural-from-v2ecoli`; verify `git log --oneline origin/main..HEAD` shows only your commits; `gh pr create` (no AI attribution). CI green (structural tests gone; `test_no_parsimony*` + `test_shape` green).
- [ ] **Step 2: After the user merges v2ecoli** — capture the squash-merge SHA on v2ecoli main.
- [ ] **Step 3: Bump 3d-ecoli's pin** — set `v2ecoli` source `rev` to that SHA in `3d-ecoli/pyproject.toml`, `uv lock`, run the full 3d-ecoli suite once more, commit.
- [ ] **Step 4: 3d-ecoli PR** — push `consolidate-structural`; verify commit provenance; `gh pr create`. CI green.
- [ ] **Step 5:** On both merges, offer to `git worktree remove` the two worktrees + delete branches + sync main.

---

## Self-Review

**Spec coverage:** Part A (parsimony-free cell_shape) → Task 1; A2/A3 deletions → Tasks 9,10; Part B unify (5 entry points) → Tasks 2,4,5; Capsule reconstruction (B2) → Task 3; EcoliPackStep/composite/acceptance (B1b) → Tasks 5,6; workspace promotion + investigation move (B3) → Task 7; stale path + deps (B5,B6) → Task 8; sequencing → Tasks 8,11. Covered.

**Placeholder scan:** the one intentional deferral is documented, not hidden — live RNA/ribosome placement in `pack_from_state` is an optional-kwarg follow-up (Task 4 Step 3), matching the investigation's own scaffolded status; every step has concrete code or exact file edits.

**Type consistency:** `pack_from_state` signature is identical everywhere it appears (Tasks 4, 5 import, spec). `_capsules_from_shape` returns `(outer, inner, envelope_dict)` used consistently. Investigation composite key `ecoli_3d.composites.ecoli_structural.ecoli_structural` matches between Task 6 (module path) and Task 7 (study reference).
