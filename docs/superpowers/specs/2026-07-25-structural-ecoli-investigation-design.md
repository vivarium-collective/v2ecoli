# structural-ecoli investigation + snapshot study (Sub-project 1)

**Date:** 2026-07-25
**Repo:** `v2ecoli` (worktree `/Users/eranagmon/code/v2e-structural`, branch `feat/structural-ecoli-investigation`, off `origin/main`).
**Builds on:** `baseline_parsimony` composite (PR #371, on main) + the Parsimony Viewer / `3d_pack` capability in vivarium-workbench (PR #578, on main).

## Goal

Create a `structural-ecoli` investigation in the v2ecoli workspace with **one study** that runs `baseline_parsimony`, saving 3D snapshot packs at **~10 s** (post-equilibration initial) and **right before division**, wired so the **Parsimony Viewer** under Analysis Tools loads those packs for this study. This is the first real end-to-end demonstration of the just-merged composite + viewer.

## Non-goals

- No workbench code changes (the viewer + capability matching already ship on main).
- No rerun feature — that's Sub-project 2 (its own spec).
- Not committing the generated packs/meshes by default (they're large; see "Artifacts").

## Verified mechanism (why this works with no code changes)

- The workbench launches a study run detached with **`cwd = workspace root`** (`run_registry.spawn_detached`, `cwd=str(workspace)`).
- `baseline_parsimony._studies_root()` reads `./workspace.yaml`'s `layout.studies` relative to that CWD → `workspace/studies`; `_resolve_pack_out_dir(None, study)` → `workspace/studies/<study>/viz/3d`.
- The Parsimony Viewer discovers packs by scanning each study dir's `viz/3d/*.pack.json` (`saved_visualizations.build_saved_visualizations` → `iter_study_dirs()`), associating a pack to the study by directory.
- **Therefore:** if the study's `baseline_parsimony` config sets `study: <this-study-slug>`, its `EcoliPackStep` writes `workspace/studies/<slug>/viz/3d/{initial,pre-division}.pack.json`, and the viewer attributes them to this study. No new glue required.

## Components

### 1. Investigation — `workspace/investigations/structural-ecoli/investigation.yaml`
Minimal, extensible:
```yaml
name: structural-ecoli
title: Structural E. coli
question: >
  What does the whole-cell molecular state look like in real 3D space at key
  points in the cell cycle — just after birth vs. right before division?
studies:
  - s01-birth-and-division
status: active
```
Created via `POST /api/investigation-create` (or written directly + validated by `investigations.load_spec`).

### 2. Study — `workspace/studies/s01-birth-and-division/study.yaml`
v4/v3 shape (must carry a non-empty `baseline:` block or the study-detail page 500s):
```yaml
name: s01-birth-and-division
title: Birth and pre-division 3D snapshots
question: >
  Pack the simulated molecular state into a 3D cell at ~10 s (post-equilibration
  birth) and right before division, for the Parsimony Viewer.
baseline:
  - name: baseline_parsimony
    composite: v2ecoli.composites.baseline_parsimony.baseline_parsimony
    params:
      study: s01-birth-and-division      # MUST equal this study's slug (drives pack out_dir)
      snapshots:                          # explicit (matches the composite default)
        initial: 10.0
        pre-division: division_time
      top_n: 40
      scale: 0.3
      seed: 0
      cache_dir: out/cache
      n_steps: 2700                       # a full generation; run stops at division
status: active
```
Notes:
- `study` param = the slug so packs land in this study's `viz/3d/`.
- `snapshots` is stated explicitly for clarity even though it equals the composite default.
- `n_steps` = one generation; `baseline` stops cleanly at division (`run_with_division`), so the `pre-division` snapshot (fired at `division_time − ε`) is captured and the run ends near division.

### 3. Run + view
Run via the workbench (`POST /api/study-run-baseline {study: s01-birth-and-division}`) or the study-detail "Run" button. On completion, `studies/s01-birth-and-division/viz/3d/initial.pack.json` and `pre-division.pack.json` exist → the Parsimony Viewer's card lists this study with an Initial · Pre-division gallery.

## Build / verification plan (implemented directly — content + a run, no SDD)

1. **Create** the investigation + study YAMLs; validate they load (`investigations.load_spec` doesn't raise; study-detail page renders).
2. **Seam smoke (fast):** run `baseline_parsimony` for this study with a tiny `top_n` (e.g. 2) and `snapshots: {initial: 2.0}` only, a short horizon → assert `workspace/studies/s01-birth-and-division/viz/3d/initial.pack.json` appears and is `parsimony.pack.v1`, and that `saved_visualizations`/the Parsimony Viewer associates it with this study. This confirms the CWD→out_dir→viewer chain before the long run.
3. **Full run (background):** run the study with the real config (`top_n=40`, both snapshots, `n_steps=2700`) detached; poll for completion via git/on-disk packs (not the buffered log). Report when both packs exist.
4. **Verify in the viewer:** serve the workbench (from the v2ecoli venv so composites resolve) and confirm the Parsimony Viewer shows this study with Initial + Pre-division, Open renders each.

## Artifacts / git

- **Commit:** `investigation.yaml` + `study.yaml` (+ this spec).
- **Do NOT commit by default:** the generated `viz/3d/*.pack.json` + `.meta.json` + `meshes/*.obj` — top_n=40 packs + LOD meshes are large. They live on disk and the local viewer serves them. (The existing `ecoli-3d` packs ARE committed; if we later want this study's packs hosted/committed, that's a separate step — likely publish to R2 like ecoli-3d.) Add `workspace/studies/s01-birth-and-division/viz/3d/` to `.gitignore` (or rely on an existing viz ignore) so the run's heavy output isn't accidentally committed.

## Risks

- **Run length:** one generation to division + packing two top_n=40 snapshots ≈ tens of minutes (run in background).
- **Packer environment:** `build_pack` shells to the Rust `parsimony` CLI (`PARSIMONY_HOME=/Users/eranagmon/code/parsimony`) and fetches AlphaFold structures; the run must have those available. The known non-ASCII `UnicodeDecodeError` in `pbg_parsimony.engine` needs `PYTHONUTF8=1` in the run environment (the pack is written before the crash, but set it to be safe).
- **ParCa cache:** the run reuses `out/cache`; ensure it's present/symlinked in the worktree.
