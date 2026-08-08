# Whole-Cell Model Comparison — reusable Vivarium Investigation

**Date:** 2026-08-01
**Status:** design (approved directionally; pending written-spec review)
**Repo/worktree:** `v2ecoli` @ `origin/main`, worktree `v2ecoli--compare-harness` (branch `compare-harness`)

## Purpose

Turn the existing `v2ecoli-vecoli-comparison` investigation into a **reusable
Vivarium Investigation for comparing whole-cell models**. You point it at a
reference model repo and a list of configurations; it runs a fixed candidate
model against the reference through each config and emits an objective,
per-config evaluation — report cards, gating verdicts, and visualizations — with
**no investigation history** (no before/after, no fix narrative).

The candidate model is fixed as **v2ecoli** (the process-bigraph whole-cell
model; the framework lives in its repo). The **reference repo and the config
list are the parameters**. A configuration is the single unit of comparison:
media, nutrient condition, and any process swap are all fields of one reference
config.

## Non-goals

- Full A-vs-B generality (arbitrary candidate). The candidate is always
  v2ecoli. A `reference.kind` seam is left open for a future non-vEcoli
  reference, but no second adapter is built here.
- Reporting on fixes, root causes, or the historical divergence investigation.
  That content is removed, not relocated.
- Changing the grading model. Gates remain `{parca, statistical}`.

## Current state (origin/main, what we are changing)

- Harness code: `scripts/_compare/` (config-driven per study) + `v2e-compare`
  CLI (`scripts/compare_cli.py`).
- **Hardcoded to the v2ecoli↔vEcoli pair:** `orchestrator.py` bakes in
  `VECOLI_REPO = "/Users/eranagmon/code/vEcoli"` and `VECOLI_PYTHON`;
  `runner._run_engines` runs `run_comparison_ensemble.py` twice
  (`--composite v2ecoli` / `--composite vecoli`).
- **Two config concepts:** plain baseline studies pass `--condition`; process
  swaps pass a separate `from_vecoli_config` (metabolism_redux).
- Investigation `whole-cell-model-comparison`'s predecessor
  (`v2ecoli-vecoli-comparison`) blends base comparison studies and
  `metabolism_redux_*` swap variants as 12 members.
- Executive prose and several member studies (`with_aa`, `no_oxygen`,
  `metabolism_redux_*`) carry before/after fix and root-cause narrative.
- Report-card modules on origin/main: `config`, `parca`, `standard`,
  `statistical`, `composition`, `distribution`, `metabolism`, `trajectory`.

## Design

### 1. Identity — advertise the reusable framework

Rename the investigation `v2ecoli-vecoli-comparison` → **`whole-cell-model-comparison`**.

- **Title:** "Whole-Cell Model Comparison"
- **Question:** "Does a candidate whole-cell model reproduce a reference
  implementation across a set of configurations?"
- **Executive / how-to-read / biological_story:** describe the framework first —
  inputs (a reference repo + a list of configs), what it measures
  (matched-initial-state per-observable deltas over N seeds), how it grades
  (`parca` t=0 + multi-seed `statistical` Welch t-test) — then state *this
  instance's* binding: candidate = v2ecoli, reference = vEcoli. Strictly
  current-state; no history.
- **Path ripple (in scope):** `docs/report_cards/v2ecoli-vecoli-comparison/`
  and everything keyed on `invest_name` (via `card_root(spec)`) move to
  `whole-cell-model-comparison`. Update any dashboard/registry references and
  investigation-member back-references.

### 2. Framework spec — the `comparison:` block is the reusable interface

The investigation YAML remains the single source of truth. The `comparison:`
block is extended to fully declare the reference engine and the config list:

```yaml
comparison:
  candidate: v2ecoli
  reference:
    repo: env:V2E_VECOLI_DIR       # path, or env:VAR indirection
    kind: vecoli                   # run-interface convention: parca.py + runscripts.workflow
  defaults:
    seeds: 4
    gens: 1
    cards: [parca, statistical, standard, trajectory]
  configs:                         # ← the list of configs; ONE report card each
    - { name: basal,       config: basal }                # condition → standard reference config
    - { name: with_aa,     config: with_aa }
    - { name: acetate,     config: acetate }
    - { name: succinate,   config: succinate }
    - { name: no_oxygen,   config: no_oxygen }
    - { name: redux_basal, config: configs/metabolism_redux.json, condition: basal }  # swap = a config
    # ... remaining redux_* variants as configs
```

**Config is the unit.** Each `configs[]` entry resolves to exactly one reference
config: either a **condition name** (framework expands it to the standard
reference config for that condition) or a **path** to a reference config JSON
(which may carry a process swap). Per-entry overrides (`seeds`, `gens`, `cards`,
`condition`) fall back to `defaults`. `from_vecoli_config` is retired — a swap is
just a config whose JSON declares it.

### 3. Reference-engine descriptor (generalize the hardcoded paths)

- New small module `scripts/_compare/reference.py`: a `ReferenceEngine`
  dataclass built from the `reference` block — `repo` (resolving `env:VAR`),
  `python` (`<repo>/.venv/bin/python`), and `kind`. `kind: vecoli` provides the
  run commands (`runscripts/parca.py`, `-m runscripts.workflow`) and the
  vEcoli-family PATH shim currently in `orchestrator._vecoli_env`.
- `orchestrator.py`: drop the module-level `VECOLI_REPO`/`VECOLI_PYTHON`
  constants; the run wrappers take a `ReferenceEngine`. The vEcoli-specific
  behavior lives behind `kind == "vecoli"`, isolated so a new kind is additive.
- `study_spec.py`: `StudySpec` gains `config: str` (path or condition) and
  `reference: ReferenceEngine`; `from_vecoli_config` field removed. The
  investigation context carries the parsed `reference` and `configs`.
- `runner._run_engines(spec)`: for the study's single config, run candidate then
  reference through **that same config**. The candidate side converts/injects
  the reference config via existing `inject.py` / `config_adapter.py`; the
  reference side applies it natively (path → `--config`; condition → standard).
- Study generation: `configs[]` → one materialized study per entry (name =
  `configs[].name`), replacing the hand-maintained `members`/study set.

### 4. Cleanup — objective evaluation only

- Rewrite `investigation.yaml` `executive`, `how_to_read`, `biological_story`,
  `glossary` to describe the framework and the current result only. Remove all
  before/after, "we fixed", root-cause (`rpoBC`, `exp_free`), and
  found-and-fixed prose.
- Strip the same from every member study.yaml (`with_aa`, `no_oxygen`,
  `metabolism_redux_*`, `acetate`, `succinate`, `basal`, `statistical`, `parca`).
  Each study states: the config compared, the matched per-observable deltas, and
  the verdict.
- `materialize.py` finding statements are already current-state; audit to ensure
  no fix-history phrasing is emitted. Findings are objective facts about the
  latest run.

### 5. Report cards & gating

Gating is **unchanged**: **`{parca, statistical}`** — multi-seed Welch t-test
over ≥4 seeds plus the t=0 initial-state match. `standard` (single-seed) stays
illustrative-only (single-seed cannot separate a port divergence from seed
noise). Verdict → outcome map unchanged: `within_tol→PASS`, `drift→PARTIAL`,
`mismatch→FAIL`. Cards per config come from `defaults.cards` (overridable).

### 5a. Visualization overhaul (all studies, both surfaces)

Redesign the report cards and charts. Both output surfaces get it: the
standalone HTML report (`docs/report_cards/`) **and** the workbench study
report-card views. Follow the `dataviz` method (form → color-by-job → validate
palette with `scripts/validate_palette.js` → mark specs → hover → a11y pass).

- **Unified visual system.** Today `report.py` and `plotly_helpers.py` each carry
  an ad-hoc palette (engine colors indigo/amber; status green/amber/red). Extract
  ONE token set — surfaces, ink, a validated 2-hue **categorical** pair for the
  two engines (candidate vs reference), and a reserved **status** palette
  (within_tol / drift / mismatch, each with glyph + label, never color-alone) —
  into a single shared module (`scripts/_compare/theme.py`) consumed by every
  card and by `report.py`. Seed the tokens from the workbench design system
  (`vivarium_workbench/static/style.css` `:root`) so cards read as the new
  workbench look; run the validator (light + dark) and snap to passing steps.
  Cards render **theme-aware** (light/dark).
- **Richer per-observable charts.** Candidate-vs-reference overlays with a
  cross-seed mean line + confidence band (not one line per seed); a per-observable
  Δ panel with tolerance shading; inline KS / Welch-t annotation. One axis per
  chart (never dual-axis); crosshair+tooltip hover.
- **Study-level summary card.** New `summary` card at the top of every study: the
  verdict pill strip, a per-observable |Δ| status row (heat row, status palette),
  seed count, and gate status — the at-a-glance objective summary before the
  detailed cards. It is informational (no new gate).
- **Cross-config overview.** New investigation-level view: a configs × observables
  matrix of verdict/|Δ| (status fill + value label), so reproduction across the
  whole config set reads at once. Rendered on both the HTML report index and the
  investigation dashboard view.

Non-gating: the visuals present the same verdicts; they do not change grading.

### 6. CLI

- New `v2e-compare init --reference <repo> --configs <dir-or-list> [-o <name>]`:
  scaffolds an investigation.yaml with the `comparison:` block above (candidate
  v2ecoli, the given reference, one `configs[]` entry per supplied config or
  condition), then materializes its studies. No sims run.
- `v2e-compare run <investigation>` / `study <name>` / `scaffold` unchanged in
  spirit; they consume the new descriptor + configs.

### 7. Re-run current models (final phase)

- After the framework + cleanup + tests land, **verify end-to-end on one config**
  locally (`v2e-compare study basal`), confirming the report card + verdict
  render objectively.
- Then run the full config set on the mini via Ray (`v2e-compare run
  whole-cell-model-comparison --ray`) against current v2ecoli (`origin/main`)
  and current vEcoli (`main`). Regenerate all report cards + visualizations.
  This is the only heavy-compute step (full ParCa + Nextflow 2-gen lineages per
  config; hours) and is sequenced last to avoid re-running if output layout
  changes.
- **Render vs. run are separate.** The heavy step produces cached per-engine
  zarr stores; rendering cards/charts from them is seconds
  (`v2e-compare run … --render-only`). So the viz overhaul (§5a) iterates against
  already-cached stores without re-running engines — do the full run once, then
  re-render freely.

### 8. Testing

Extend `tests/compare/`:
- **Manifest parsing:** `reference` descriptor (`env:VAR` resolution, `kind`) and
  `configs[]` (condition vs path, per-entry overrides).
- **Config-is-the-unit generation:** one study per config; `from_vecoli_config`
  removed cleanly (studies that used it now carry a `config:` path).
- **Reference-engine seam:** `kind: vecoli` yields the expected run commands /
  PATH shim; an unknown kind raises a clear error.
- **Objective-narrative lint:** rendered cards + executive contain no
  fix-history keywords (`before`, `after fix`, `root cause`, `we fixed`,
  `rpoBC`).
- **Theme/palette:** the shared `theme.py` categorical (engine) pair + status
  palette PASS `scripts/validate_palette.js` in light and dark.
- **Summary card:** builds a per-observable status row + verdict strip from a
  verdict fixture; status is glyph+label, not color-alone.
- **Cross-config overview:** builds a configs × observables matrix from a
  multi-study verdict set.
- E2E cross-engine run stays gated behind `COMPARE_E2E=1`.

## Risks / notes

- **Rename churn:** report-card paths and any hardcoded `v2ecoli-vecoli-comparison`
  references must all move; a stale reference silently drops a study from the
  dashboard. Grep the workspace + dashboard for the old id.
- **Editable-install gap:** the serving/running venv's editable `v2ecoli` points
  at the canonical checkout. Run/verify with the worktree on PATH per the repo's
  worktree rule so the executed code matches `origin/main`.
- **Compute cost:** the re-run is hours; the plan front-loads a single-config
  verification to de-risk it.
- **Config expansion for conditions:** the framework must produce a standard
  reference config for a bare condition name identically to how the current
  ensemble does (`--condition`), so baselines are unchanged by the unification.
