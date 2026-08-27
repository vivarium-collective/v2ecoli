# Design: Move the structural (3D-model) investigation out of v2ecoli into 3d-ecoli

**Date:** 2026-08-27
**Status:** Draft for review
**Repos touched:** `v2ecoli` (strip), `3d-ecoli` (consolidate). No changes to `viva-parsimony` / `parsimony`.

## Goal

`v2ecoli` should **not** import `viva-parsimony` (`pbg_parsimony`) and should **no
longer carry the structural / 3D-model investigation**. All of that work is
consolidated into the standalone **`3d-ecoli`** repo (package `ecoli_3d`), which
imports `v2ecoli` and `pbg_parsimony` to build the 3D structural model. `3d-ecoli`
is promoted to a viva workspace so the `structural-ecoli` investigation and its
study run there through the workbench.

Repo/package names are kept **as-is** (repo `3d-ecoli`, package `ecoli_3d`) — no
rename churn.

## Current state (verified 2026-08-27, off `origin/main`)

- **The two repos diverged into two paradigms — neither is simply "stale".**
  - **3d-ecoli** (`main`, package `ecoli_3d`) is a complete, working standalone
    3D-model repo: a **richer** `ecoli_3d/build.py` (~1440 lines: full
    transcription/translation/septum/flagella; last touched 2026-06-27), a
    **snapshot** composite (`parsimony-ecoli` / `EcoliStructuralStep` — reads a
    committed `.npz` state and packs once via `build_model`), committed state
    snapshots, a publish→R2 pipeline, and a webapp. It is **not** yet a viva
    workspace (no `workspace.yaml`).
  - **v2ecoli `origin/main`** (`326f42ab`) carries a **different, slimmer**
    structural stack (`structural/build.py` ~519 lines; last committed
    2026-07-25): `EcoliPackStep` / `baseline_parsimony` — which packs **during a
    live baseline sim** at declared snapshot times (`pack_from_state` + live
    extractors `bulk_to_counts`/`bulk_to_locations`/`chromosome_state_from_live`/
    `rnaps_from_live`). This is what the `structural-ecoli` investigation runs on.
  - The `structural-ecoli` investigation's own `findings` say its live-sim results
    are **unverified/scaffolded** — packs were never committed, the pipeline can't
    run in the canonical env (parsimony binary absent), and its acceptance gate is
    wired to **SKIP**. Its real deliverable was the acceptance-gate scaffolding.
- **The only coupling that makes the *plain* baseline pull in `pbg_parsimony`:**
  `v2ecoli/composites/ecoli_baseline.py` (~L1166–1179) unconditionally registers
  `ShapeStep` and appends a `shape_step` layer; `v2ecoli/cell_shape.py` does
  `from pbg_parsimony import Capsule` at module top.
- **`ShapeStep`'s emitted `shape` store is already pure floats.** `ShapeStep.update`
  pops `capsule`/`inner_capsule`/`envelope` (the only non-float, Capsule-bearing
  values) before emitting. `Capsule` is a thin arithmetic wrapper
  (`half_len`, `radius`); the numbers survive as `radius_A`/`half_len_A`/
  `inner_radius_A`/`inner_half_len_A`. **Nothing non-structural in v2ecoli reads the
  `shape` store** (grep-verified) — its only consumer is the parsimony 3D build.

## Decisions (settled)

1. **`cell_shape.py` / `ShapeStep` stays in v2ecoli**, rewritten to be
   parsimony-free (drop the `Capsule` import; compute the `*_A` numbers with plain
   arithmetic). The baseline keeps appending `shape_step` and keeps emitting cell
   geometry from mass. The **Capsule/envelope object construction moves to
   `3d-ecoli`**, which reads the plain numbers from `v2ecoli.cell_shape.shape_from_mass`
   and builds `pbg_parsimony.Capsule` itself for `build_pack`.
2. **`3d-ecoli` is promoted to a viva workspace** (add `workspace.yaml` + register
   `ecoli_3d` as the workspace package) so the `structural-ecoli` investigation and
   the `s01-birth-and-division` study move over intact and run via the workbench.
3. **Unify on 3d-ecoli's builder.** 3d-ecoli's richer `ecoli_3d/build.py` is THE
   single builder. v2ecoli's `structural/build.py` is **deleted**, not migrated.
   We bring v2ecoli's `EcoliPackStep` + `baseline_parsimony` composite +
   `acceptance.py` + the investigation/study into 3d-ecoli, and **re-point the pack
   step at `ecoli_3d.build`** by giving that module the 5 live-state entry points
   the step imports (`bulk_to_counts`, `bulk_to_locations`,
   `chromosome_state_from_live`, `rnaps_from_live`, `pack_from_state`).

## Design

### Part A — v2ecoli (strip), off `origin/main`

Worktree: `~/code/v2ecoli--remove-ecoli-3d` (branch `remove-structural-from-v2ecoli`).

**A1. Sever the parsimony import from `cell_shape.py` (keep the file).**
- Remove `from pbg_parsimony import Capsule`.
- In `shape_from_mass`, replace the two `Capsule(...)` constructions with direct
  arithmetic:
  - `radius_A = r * 1e4`, `half_len_A = (lcyl / 2.0) * 1e4`
  - `inner_radius_A = radius_A * s`, `inner_half_len_A = half_len_A * s`
- Return dict: keep all the existing float keys. For the previously-Capsule keys:
  drop `capsule`/`inner_capsule`; make `envelope` a **plain-number** dict
  (`outer_radius_A`/`outer_half_len_A`/`inner_radius_A`/`inner_half_len_A` + the
  existing float fields) so downstream can reconstruct Capsules without importing
  parsimony. `SHAPE_KEYS`, `zero_shape`, and `ShapeStep.update`'s float emission are
  unchanged (they already excluded the Capsule/envelope keys).
- `ecoli_baseline.py` shape_step block: **unchanged.** The store contract is float-only.
- Update `tests/test_shape.py` to the parsimony-free API (assert numbers, not
  `Capsule` objects). This test **stays** in v2ecoli.

**A2. Delete the rest of the structural surface from v2ecoli.**
- `v2ecoli/structural/` (`__init__.py`, `acceptance.py`, `build.py`, `pack_step.py`,
  `data/`).
- `v2ecoli/composites/ecoli_structural.py` (`baseline_parsimony`).
- `v2ecoli/workbench_viewers.py`: remove the 3D-pack deep-link (`_has_3d_pack`,
  the 3D-viewer target builder, ~L216–282 and its registration) — keep the rest of
  the module intact.
- `workspace/investigations/structural-ecoli/` + `workspace/studies/s01-birth-and-division/`
  (+ any committed `viz/3d` pack artifacts).
- `reports/composite-state/v2ecoli.composites.ecoli_structural*.json` and
  `...structural.composite.parsimony-ecoli.json`.
- `docs/superpowers/specs/2026-07-25-structural-ecoli-investigation-design.md`
  (moves to 3d-ecoli).
- Tests: `tests/structural/`, `tests/test_baseline_parsimony_composite.py`,
  `tests/test_baseline_parsimony_integration.py`, `tests/test_pack_from_state.py`,
  `tests/test_pack_relax_wiring.py`, `tests/test_pack_step.py`,
  `tests/test_structural_revive.py`. (Keep `tests/test_shape.py`.)

**A3. Drop the dependency.**
- `pyproject.toml`: remove `"pbg-parsimony"` (dep) and the `[tool.uv.sources]`
  `pbg-parsimony = {...}` line. Re-lock `uv.lock`.
- `workspace.yaml`: remove the `pbg_parsimony` module block, remove it from the
  workbench `include:` list, and remove the 3D `viz_viewer_urls` entries
  (`ecoli_3d`, `initial`, `pre-division`).

**A4. Verify.**
- `python -c "import v2ecoli"` imports with **no** `pbg_parsimony` on the path
  (test in an env without parsimony installed, or assert the module isn't imported).
- Baseline composite builds and runs; `shape` store still populated.
- Full v2ecoli test suite green.

### Part B — 3d-ecoli (consolidate), off `origin/main`

Worktree: `~/code/3d-ecoli--consolidate` (branch `consolidate-structural`).

**B1. Give `ecoli_3d/build.py` the live-state entry points (unify on B's builder).**
3d-ecoli's `build.py` stays THE builder; add the 5 public symbols
`EcoliPackStep` imports, with v2ecoli's exact signatures:
- **Near-mechanical ports** (lift verbatim from v2ecoli `structural/build.py`):
  `bulk_to_counts(bulk)`, `bulk_to_locations(bulk)` (+ its `_TAG_TO_COMPARTMENT`),
  `chromosome_state_from_live(full_chromosome, active_replisome=None)`,
  `rnaps_from_live(active_rnap, full_chromosome=None, chromosome_domain=None)`
  (+ the `_active_rows` helper). B already has identical `classify_domains` /
  `_descendant_domains_set` and `REPLICHORE_BP`, so these bind cleanly.
- **The real refactor** — `pack_from_state(out_dir, name, counts, volume_fl,
  locations=None, *, top_n=40, scale=0.3, proxy_lod=2, relax=False,
  cache_dir="out/cache", relax_params=None, envelope=True, periplasm_gap_A=250.0,
  rnaps=None, n_chromosomes=1, fork_fraction=0.0)`: extract B's in-memory core
  (ingredient assembly + `Chromosome` + `build_pack`) out of `build_model` so it
  is callable with **passed-in** `counts/volume_fl/locations/rnaps/
  n_chromosomes/fork_fraction`. `build_model` becomes the file-reading wrapper
  (`pack_from_state(*load_state(state_source), rnaps=…, n_chromosomes=…, …)`),
  preserving B's snapshot path unchanged. Keep B's richer placement semantics
  (70S/RNAP count=0 → placed via markers/chromosome stage), fed by the live
  `rnaps` list. Thread `relax`/`relax_params` through (port B a `relax_ingredients`
  step, or no-op when `relax=False` initially — relax is opt-in).
- **Reconcile the routing input:** B's `select_ingredients` routes by tag LETTER
  via `_route_envelope`; A's `pack_from_state` passes `locations` as
  parsimony-compartment NAMES. Standardize `pack_from_state` on B's convention
  (tag letters via `bulk_to_locations`→`_route_envelope`) so the one builder has a
  single routing path.

**B2. Reconstruct `Capsule` locally from numeric fields.** B's only `cell_shape`
consumption is `build_model` (B:1214–1224), which reads the Capsule OBJECTS
`shape["capsule"]` / `shape["envelope"]["outer_membrane"|"inner_membrane"]`. After
v2ecoli A1 makes `shape_from_mass` return plain numbers, rebuild locally:
```python
from pbg_parsimony import Capsule
capsule = Capsule(half_len=shape["half_len_A"], radius=shape["radius_A"])
inner   = Capsule(half_len=shape["inner_half_len_A"], radius=shape["inner_radius_A"])
envelope = {"outer": capsule, "inner": inner}
```
(`pack_from_state`'s own in-memory path builds the envelope from
`Capsule.from_volume_fl(volume_fl)` + `periplasm_gap_A`, as v2ecoli's A did.) These
are B's sole `pbg_parsimony.Capsule` construction sites — the import stays in
3d-ecoli, where parsimony belongs.

**B1b. Bring `EcoliPackStep`, the composite, and `acceptance.py`** into `ecoli_3d/`:
- `pack_step.py` → `ecoli_3d/pack_step.py`, import line changed to
  `from ecoli_3d.build import (pack_from_state, bulk_to_counts, bulk_to_locations,
  chromosome_state_from_live, rnaps_from_live)`; `_default_core` still uses
  `v2ecoli.core.build_core`.
- `ecoli_structural.py` (`baseline_parsimony`) → `ecoli_3d/`, wrapping
  `v2ecoli.composites.ecoli_baseline.baseline` and appending the `ecoli_3d`
  `EcoliPackStep` (registry key becomes `ecoli_3d.<module>.baseline_parsimony`).
- `structural/acceptance.py` → `ecoli_3d/acceptance.py`.
- 3d-ecoli keeps its existing `parsimony-ecoli` / `EcoliStructuralStep` snapshot
  composite too — both now sit on the one `build.py`.

**B3. Promote to a viva workspace.**
- Add `workspace.yaml` registering `ecoli_3d` as the workspace package + the
  `pbg_parsimony` and `v2ecoli` modules + the R2 `viz_viewer_urls`.
- Move `workspace/investigations/structural-ecoli/` and
  `workspace/studies/s01-birth-and-division/` + committed pack artifacts here.
  Repoint composite registry keys `v2ecoli.composites.ecoli_structural.*` →
  `ecoli_3d.<...>`.
- Move `workbench_viewers.py`'s 3D deep-link here (adapted to the ecoli_3d package).
- Move the `2026-07-25-structural-ecoli-investigation-design.md` spec here.

**B4. Bring the tests.** Port the v2ecoli structural tests (renamespaced to
`ecoli_3d`), merge with 3d-ecoli's existing `test_build_*`.

**B5. Fix stale bits.** `ecoli_3d/publish/03_assemble_local_view.py` hardcodes
`/Users/eranagmon/code/pbg-parsimony/...` → point at the `viva-parsimony` clone /
`pbg_parsimony` package data (`importlib.resources`).

**B6. Dependencies.** `3d-ecoli/pyproject.toml` already deps `v2ecoli` +
`pbg-parsimony`. Pin `v2ecoli` at the **post-strip** commit (after Part A merges to
v2ecoli main). Re-lock. For local dev, editable-install the post-strip v2ecoli +
this repo + `pbg_parsimony` into one venv.

### Sequencing (cross-repo dependency)

`3d-ecoli` imports `v2ecoli`, and B2 depends on A1 (cell_shape returning plain
numbers). Order:

1. **v2ecoli Part A** in its worktree; land tests green. (Can develop against a
   local editable install for 3d-ecoli in parallel.)
2. **3d-ecoli Part B** developed against the Part-A v2ecoli worktree
   (`PYTHONPATH` / editable install), tests green independently.
3. Merge v2ecoli PR to main first, then bump `3d-ecoli`'s `v2ecoli` pin to that
   commit, re-lock, merge 3d-ecoli.

Until v2ecoli Part A is on main, `3d-ecoli`'s `v2ecoli @ main` pin still resolves
to the *old* v2ecoli (which still has `cell_shape` returning Capsules) — B2's
number-based path must be written to work against the **new** cell_shape, so
develop/test B against the Part-A worktree, not `@ main`.

## Testing

- **v2ecoli:** parsimony-free import assertion; `test_shape.py` adapted; full suite
  green; a baseline smoke run still emits `shape`.
- **3d-ecoli:** existing `test_build_*` + ported structural tests green; a
  `build_pack` smoke build produces a non-empty pack; the workspace loads and the
  `structural-ecoli` investigation resolves.

## Risks / notes

- **Committed pack/mesh artifacts** may be large; confirm what's actually tracked
  under the study `viz/3d` before `git mv` (packs were historically gitignored;
  R2 hosts the big ones).
- **`workbench_viewers.py`** is a shared module in v2ecoli — excise only the 3D
  deep-link, leave the generic viewer machinery.
- **Reports** referencing the old composite key must be regenerated in 3d-ecoli or
  dropped, not carried with a dead `v2ecoli.composites.ecoli_structural` key.
- The v2ecoli `main` branch — not the stale `study-registry-migration` working tree
  — is the correct base (already used for the worktree).

## Out of scope

- No changes to `viva-parsimony` / `parsimony` (the Rust engine).
- No re-run/re-publish of the R2-hosted viewer (external; unchanged).
- No package/repo rename.
