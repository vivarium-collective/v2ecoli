# baseline + parsimony snapshot composite (Sub-project B)

**Date:** 2026-07-25
**Repo:** `v2ecoli` (worktree `/Users/eranagmon/code/v2e-3d-snapshots`, branch `feat/baseline-parsimony-snapshots`, off `origin/main`).
**Depends on / feeds:** Sub-project A (vivarium-workbench PR #578) — the Parsimony Viewer + `3d_pack` capability + per-study `models.json` gallery render whatever packs this composite writes. No workbench changes here.

## Problem

v2ecoli can pack a 3D molecular scene of the cell (parsimony), but only from pre-baked `.npz` states in the unmerged `v2e-3d` worktree — there is no composite that runs `baseline` live and emits 3D snapshot packs at meaningful simulation times. Goal: a composite that takes `baseline` as a component, adds a parsimony packing Step, and writes named `.pack.json` snapshots at declared times — default an **initial** (~10 s, post-equilibration) and a **pre-division** state — that the workbench's Parsimony Viewer renders as a saved-views gallery.

## Goals

1. A `baseline_parsimony` composite generator: `baseline()`'s live document + a packing Step appended as a final execution layer.
2. The Step fires at **declared simulation times** (default: initial ≈ 10 s, and pre-division) and writes one named pack per fire.
3. Packs land at `studies/<study>/viz/3d/<name>.pack.json` (`initial.pack.json`, `pre-division.pack.json`), where the workbench viewer already finds and gallery-renders them.
4. Revive the packing machinery from the `v2e-3d` worktree into `v2ecoli/structural/` (currently an empty stub on main).

## Non-goals

- No workbench/viewer changes (Sub-project A already renders packs; `initial` sorts first via the viewer's `_pack_name_rank`).
- Not optimizing run time — one run does a real generation to division and packs ~1.3 M molecules inline at each declared time; it is inherently slow (see "Cost").
- No new packing algorithm — reuse `v2e-3d`'s ingredient selection and `pbg_parsimony.build_pack` as-is.
- Not removing the pre-baked-state `parsimony_ecoli` path if it later lands; this is the live-run counterpart.

## Architecture

### Composite generator — `v2ecoli/composites/baseline_parsimony.py`
A `@composite_generator(name="baseline_parsimony", ...)` that:
1. Builds the base document via `baseline(core=core, **kwargs)` (the live whole-cell model, incl. `ShapeStep`).
2. Appends `EcoliPackStep` as a **final execution layer**, mirroring exactly how `baseline` appends `shape_step` (`v2ecoli/composites/baseline.py:895–919`):
   ```python
   core.register_link("EcoliPackStep", EcoliPackStep)
   cell_state['pack_step'] = {
       '_type': 'step', 'address': 'local:EcoliPackStep',
       'config': {'snapshots': {...}, 'study': ..., 'top_n': 40, 'scale': 0.3},
       'inputs': {
           'bulk':            ['bulk'],
           'shape':           ['shape'],
           'mass':            ['listeners', 'mass'],
           'global_time':     ['global_time'],
           'full_chromosomes':['full_chromosomes'],  # for division_time
       },
       'outputs': {'pack_status': ['pack_status']},   # small status store, pre-seeded
   }
   execution_layers = execution_layers + [['pack_step']]
   ```
   Register it in `v2ecoli/composites/__init__.py` (the `__all__` already lists a `parsimony`-family entry stub).

### Packing Step — `v2ecoli/structural/` (revived from `v2e-3d`)
Bring `v2e-3d/v2ecoli/structural/build.py` (`build_model`, `select_ingredients`, `load_state`) into `v2ecoli/structural/`, and add `EcoliPackStep(Step)`:

- **Inputs** as above. **`update_condition(timestep, states)`** returns True only when `global_time` has crossed the next un-fired declared time — the same idiom several processes use (`metabolism.py:509` gates `next_update_time <= global_time`). Each declared time fires at most once.
- **Declared times** (`config['snapshots']`, configurable), default two:
  - `"initial"` → fixed `global_time ≈ 10.0 s` (post-equilibration birth state).
  - `"pre-division"` → `full_chromosomes.division_time − ε`. `division_time` (= `global_time + D_period`) is written per full chromosome at `chromosome_replication.py:692`; reading it gives the scheduled division time in advance, so the Step can snapshot just before it. (Chosen over a max-mass proxy for precision.)
- **On fire:** reconstruct the cell geometry from the current shape — the emitted `['shape']` store holds only shape *floats* (`volume_fl`, `length_um`, …); the `Capsule`/`envelope` objects are intentionally excluded (`cell_shape.py:86`, a `map[float]` store can't hold objects). So the Step recomputes the `Capsule` + gram-negative `envelope` from `volume_fl` (or `mass`) via `v2ecoli.cell_shape.shape_from_mass` — the same computation `ShapeStep` runs, no geometry duplicated in a store. Then call `build_model(...)` → `pbg_parsimony.build_pack(counts=<bulk>, capsule=…, envelope=…, top_n, scale, out_dir=studies/<study>/viz/3d, name=<snapshot-name>)`, writing `<name>.pack.json` (+ `.meta.json`).
- **`update()`** returns a small `pack_status` (e.g. `{name: n_placed}`) for observability; the durable artifact is the pack file.

### Ingredient selection
Reuse `v2e-3d/structural/build.py` `select_ingredients(counts, top_n=40)` verbatim (curated assemblies — 70S ribosome, RNA polymerase, … — + top-N AlphaFold monomers). Default `top_n=40`, `scale=0.3`, matching `v2e-3d`.

## Data flow

```
baseline() live document  ── ShapeStep → ['shape'] floats (volume_fl, length_um, …)
   + EcoliPackStep (final layer)
       reads ['bulk'], ['shape'], ['full_chromosomes'], ['global_time']
       update_condition fires @ global_time≈10s  and  @ division_time−ε
         → shape_from_mass(volume/mass) → Capsule + envelope
         → build_model → pbg_parsimony.build_pack
         → studies/<study>/viz/3d/{initial,pre-division}.pack.json
                                   │
        (Sub-project A) workbench Parsimony Viewer + /api/study/<study>/3d/models.json
                                   → gallery: Initial · Pre-division
```

## File-by-file

**New**
- `v2ecoli/composites/baseline_parsimony.py` — the `baseline_parsimony` generator (wraps `baseline`, appends `EcoliPackStep`).
- `v2ecoli/structural/__init__.py`, `v2ecoli/structural/build.py` — revived from `v2e-3d` (`build_model`, `select_ingredients`, ingredient/structure tables).
- `v2ecoli/structural/pack_step.py` — `EcoliPackStep(Step)` (declared-time gating + pack-on-fire).

**Changed**
- `v2ecoli/composites/__init__.py` — import + register `baseline_parsimony` (reconcile the existing `parsimony_ecoli`/`__all__` stub, `composites/__init__.py:74`).
- `pyproject.toml` — `pbg-parsimony` is already a dependency (`pyproject.toml:16`); confirm the `build_pack`/AlphaFold-fetch extras the revived `build.py` needs are present.

## Testing

- **Gating (fast, stubbed packer):** unit-test `EcoliPackStep.update_condition` — fires once when `global_time` crosses each declared time; fires for `pre-division` when `global_time ≥ division_time − ε` given a `full_chromosomes.division_time`; never re-fires a spent snapshot.
- **Pack-on-fire (stubbed `build_pack`):** `update()` at a declared time calls the packer with reconstructed capsule + bulk counts and writes to `studies/<study>/viz/3d/<name>.pack.json`.
- **Integration (small, real packer):** build `baseline_parsimony` with `top_n=2` and a short run that reaches the `initial` snapshot; assert `initial.pack.json` exists and validates as `parsimony.pack.v1` (top-level `format`, `bounds`, `ingredients`, `placements`). Gate the full pre-division run behind a slow/opt-in marker.
- **Viewer contract:** assert the two pack filenames are what Sub-project A's gallery expects (`initial` ranks first).

## Cost

One run executes a **real generation to division** (tens of minutes) and packs ~1.3 M molecules inline at each declared time (a heavy Rust `build_pack` per snapshot). Expect a single run in the tens-of-minutes range. This is inherent to producing real snapshots from a live baseline run to pre-division (the chosen approach over pre-baked states or decoupled dump-then-pack). Keep the full pre-division integration test opt-in; use pre-baked `.npz` only if a fast deterministic fixture is later needed.

## Sequencing

This spec → implementation plan → implement in the worktree → verify (fast gating + integration tests; one real end-to-end run to confirm both packs render in the workbench) → PR to `v2ecoli` main.
